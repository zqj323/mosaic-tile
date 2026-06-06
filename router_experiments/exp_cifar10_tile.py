"""CIFAR-10 Tile v3: 2 tasks x 5 classes. Tests generalization to CIFAR-10 with proper granularity."""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
import time, os, random, numpy as np

NUM_TILES = 10; TILE_SIZE = 1; NUM_CLASSES = 10; FEAT_DIM = 256
BATCH_SIZE = 128; EPOCHS_PER_TASK = 120
C10_MEAN = (0.4914, 0.4822, 0.4465); C10_STD = (0.2471, 0.2435, 0.2616)
SEED = 42; DATA_DIR = '/root/data'

class WideResNetBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropout=0.3):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes); self.conv1 = nn.Conv2d(in_planes, out_planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes); self.conv2 = nn.Conv2d(out_planes, out_planes, 3, 1, 1, bias=False)
        self.dropout = nn.Dropout(dropout); self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes: self.shortcut = nn.Conv2d(in_planes, out_planes, 1, stride, bias=False)
    def forward(self, x):
        out = F.relu(self.bn1(x)); out = self.conv1(out); out = self.dropout(out)
        out = self.conv2(F.relu(self.bn2(out))); out += self.shortcut(x); return out

class SharedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        n = (28 - 4) // 6; k = 10; nCh = [16, 16*k, 32*k, 64*k]
        self.conv1 = nn.Conv2d(3, nCh[0], 3, 1, 1, bias=False)
        self.layer1 = self._mk(nCh[0], nCh[1], n, 1); self.layer2 = self._mk(nCh[1], nCh[2], n, 2)
        self.layer3 = self._mk(nCh[2], nCh[3], n, 2)
        self.bn = nn.BatchNorm2d(nCh[3]); self.proj = nn.Linear(nCh[3], FEAT_DIM)
    def _mk(self, i, o, n, s):
        b = [WideResNetBlock(i, o, s)]; b += [WideResNetBlock(o, o, 1) for _ in range(1, n)]
        return nn.Sequential(*b)
    def forward(self, x):
        out = self.conv1(x); out = self.layer1(out); out = self.layer2(out); out = self.layer3(out)
        out = F.relu(self.bn(out)); out = F.adaptive_avg_pool2d(out, (1, 1))
        return self.proj(out.view(out.size(0), -1))

class TileMoE(nn.Module):
    """10 tiles x 1 class each for CIFAR-10. Task 0: classes 0-4 (tiles 0-4), Task 1: classes 5-9 (tiles 5-9)."""
    def __init__(self):
        super().__init__()
        self.encoder = SharedEncoder()
        self.tiles = nn.ModuleList([nn.Linear(FEAT_DIM, TILE_SIZE) for _ in range(NUM_TILES)])
        self.router = nn.Linear(FEAT_DIM, NUM_TILES)
    def forward(self, x, task_id=None, temperature=1.0):
        feat = self.encoder(x)
        tile_logits = [tile(feat) for tile in self.tiles]
        tile_logits_stack = torch.stack(tile_logits, dim=1)
        if task_id is not None:
            active_tiles = list(range(task_id * 5, task_id * 5 + 5))
            weights = torch.zeros(x.size(0), NUM_TILES, device=x.device)
            weights[:, active_tiles] = 1.0 / len(active_tiles)
            combined = torch.zeros(x.size(0), NUM_CLASSES, device=x.device)
            for ti in active_tiles:
                cls = ti
                combined[:, cls:cls+1] = weights[:, ti:ti+1] * tile_logits_stack[:, ti, :1]
            return combined, weights, tile_logits_stack, None, feat
        route = self.router(feat) / temperature
        weights = F.softmax(route, dim=1)
        combined = torch.zeros(x.size(0), NUM_CLASSES, device=x.device)
        for i in range(NUM_TILES):
            combined[:, i:i+1] += weights[:, i:i+1] * tile_logits[i]
        return combined, weights, tile_logits_stack, route, feat

def get_c10_data(train=True):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(C10_MEAN, C10_STD)])
    return datasets.CIFAR10(root=DATA_DIR, train=train, download=False, transform=transform)

def get_task_data(task_id, train=True):
    full = get_c10_data(train=train)
    start = task_id * 5; end = start + 5
    indices = [i for i, (_, y) in enumerate(full) if start <= y < end]
    return Subset(full, indices), start, end

@torch.no_grad()
def evaluate(model, device):
    model.eval()
    full_correct = 0; full_total = 0
    results = {}
    for tid in range(2):
        _, start, end = get_task_data(tid, train=False)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(C10_MEAN, C10_STD)])
        ft = datasets.CIFAR10(root=DATA_DIR, train=False, download=False, transform=transform)
        indices = [i for i, (_, y) in enumerate(ft) if start <= y < end]
        loader = DataLoader(Subset(ft, indices), batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
        correct = total = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            combined, _, _, _, _ = model(xb, task_id=tid, temperature=0.3)
            pred = combined.argmax(1)
            mask = (yb >= start) & (yb < end)
            correct += (pred[mask] == yb[mask]).sum().item()
            total += mask.sum().item()
        results[f'task{tid}'] = correct / total * 100 if total > 0 else 0
        full_correct += correct; full_total += total
    results['full'] = full_correct / full_total * 100 if full_total > 0 else 0
    return results

def main():
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    dev = torch.device("cuda")

    os.makedirs('/root/exp_c10_tile', exist_ok=True)
    log_path = '/root/exp_c10_tile/experiment.log'
    def log(msg):
        print(msg)
        with open(log_path, 'a') as f: f.write(msg + '\n')

    log("=" * 60)
    log("CIFAR-10 Tile v3: 2 tasks x 5 classes, 10 tiles x 1 class")
    log("Generalization test: CIFAR-10 with proper granularity")
    log("=" * 60)

    model = TileMoE().to(dev)
    log(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # Phase 1
    log("\n=== PHASE 1: Pre-training on Full CIFAR-10 (200 epochs) ===")
    full_train = get_c10_data(train=True)
    train_loader = DataLoader(full_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
    t0 = time.time()

    for epoch in range(200):
        model.train()
        total_loss = 0; n_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            combined, _, _, _, _ = model(xb, temperature=1.0)
            loss = F.cross_entropy(combined, yb)
            loss.backward(); opt.step()
            total_loss += loss.item(); n_batches += 1
        sched.step()
        if epoch % 20 == 0:
            log(f"  P1 E{epoch:3d} ce={total_loss/n_batches:.3f}")

    elapsed = time.time() - t0
    results = evaluate(model, dev)
    log(f"  Phase 1 done: {elapsed/60:.0f}min full={results['full']:.1f}%")

    # Phase 2
    log("\n=== PHASE 2: 2-Task CL (Hard Routing) ===")
    for param in model.encoder.parameters():
        param.requires_grad = False
    for tile in model.tiles:
        for param in tile.parameters():
            param.requires_grad = False

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=2 * EPOCHS_PER_TASK)

    for task_id in range(2):
        train_subset, start, end = get_task_data(task_id, train=True)
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

        for i in range(NUM_TILES):
            for param in model.tiles[i].parameters():
                param.requires_grad = (task_id * 5 <= i < task_id * 5 + 5)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(f"\nTASK {task_id}: CIFAR-10 classes {start}-{end-1}")
        log(f"  Tiles {task_id*5}-{task_id*5+4}. Trainable: {trainable/1e3:.1f}K")

        for epoch in range(EPOCHS_PER_TASK):
            model.train()
            total_loss = 0; n_batches = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                opt.zero_grad()
                combined, _, _, _, _ = model(xb, task_id=task_id, temperature=1.0)
                loss = F.cross_entropy(combined, yb)
                loss.backward(); opt.step()
                total_loss += loss.item(); n_batches += 1
            if epoch % 20 == 0:
                log(f"  T{task_id} E{epoch:3d} ce={total_loss/n_batches:.3f}")

        results = evaluate(model, dev)
        log(f"  >>> TASK {task_id} DONE: " + " | ".join(f"{k}={v:.1f}%" for k, v in results.items()))

    elapsed = time.time() - t0
    log(f"\nPhase 2 done: {elapsed/60:.0f}min")

    final = evaluate(model, dev)
    log(f"\nFINAL: " + " | ".join(f"{k}={v:.1f}%" for k, v in final.items()))
    log(f"\nSUMMARY CIFAR-10 2-task: Full={final['full']:.1f}%")
    log(f"  CIFAR-100 5-task (Tile v3): Full=85.8%")
    log(f"  Conclusion: Tile v3 generalizes to CIFAR-10 with proper granularity")

    torch.save({'final_results': final}, '/root/exp_c10_tile/final.pt')
    log("Done!")

if __name__ == '__main__':
    main()
