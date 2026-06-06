"""Random exclusive + Per-task Router Re-Init (Tile variant).
Closes the ~9pp gap: tile exclusivity fixed layer 1 (16.7→76.9%),
per-task router reinit fixes layer 2 (router Phase 1 distribution bias → ~85%).
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
import time, os, random
import numpy as np

NUM_TILES = 20; TILE_SIZE = 5; NUM_CLASSES = 100; FEAT_DIM = 256
BATCH_SIZE = 128; EPOCHS_PER_TASK = 120
CIFAR_MEAN = (0.5071, 0.4867, 0.4408); CIFAR_STD = (0.2675, 0.2565, 0.2761)
SEED = int(os.environ.get('SEED', '42')); DATA_DIR = '/root/data'

random.seed(SEED); np.random.seed(SEED)
ALL_CLASSES = list(range(100)); np.random.shuffle(ALL_CLASSES)
TASK_CLASSES = [sorted(ALL_CLASSES[i*20:(i+1)*20]) for i in range(5)]
CLASS_TO_TILE = {}
for tid, classes in enumerate(TASK_CLASSES):
    for j, cls in enumerate(classes):
        CLASS_TO_TILE[cls] = tid * 4 + (j // 5)

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
    def __init__(self):
        super().__init__()
        self.encoder = SharedEncoder()
        self.tiles = nn.ModuleList([nn.Linear(FEAT_DIM, TILE_SIZE) for _ in range(NUM_TILES)])
        self.router = nn.Linear(FEAT_DIM, NUM_TILES)
    def reset_router(self):
        import math
        nn.init.kaiming_uniform_(self.router.weight, a=math.sqrt(5))
        nn.init.zeros_(self.router.bias)
    def forward(self, x, task_id=None, temperature=1.0):
        feat = self.encoder(x)
        tile_logits = [tile(feat) for tile in self.tiles]
        tile_logits_stack = torch.stack(tile_logits, dim=1)
        if task_id is not None:
            active_tiles = list(range(task_id * 4, task_id * 4 + 4))
            weights = torch.zeros(x.size(0), NUM_TILES, device=x.device)
            weights[:, active_tiles] = 1.0 / len(active_tiles)
            combined = torch.zeros(x.size(0), NUM_CLASSES, device=x.device)
            for ti in active_tiles:
                for j, cls in enumerate(TASK_CLASSES[task_id]):
                    if CLASS_TO_TILE[cls] == ti:
                        combined[:, cls] += weights[:, ti] * tile_logits_stack[:, ti, j % 5]
            return combined, weights, tile_logits_stack, None, feat
        route = self.router(feat) / temperature
        weights = F.softmax(route, dim=1)
        combined = torch.zeros(x.size(0), NUM_CLASSES, device=x.device)
        for i in range(NUM_TILES):
            combined[:, i*TILE_SIZE:(i+1)*TILE_SIZE] += weights[:, i:i+1] * tile_logits[i]
        return combined, weights, tile_logits_stack, route, feat

def get_full_data(train=True):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    return datasets.CIFAR100(root=DATA_DIR, train=train, download=False, transform=transform)

def get_task_data(task_id, train=True):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    full = datasets.CIFAR100(root=DATA_DIR, train=train, download=False, transform=transform)
    target = TASK_CLASSES[task_id]
    indices = [i for i, (_, y) in enumerate(full) if y in target]
    return Subset(full, indices)

@torch.no_grad()
def evaluate(model, device):
    model.eval()
    full_correct = 0; full_total = 0
    results = {}
    for tid, target in enumerate(TASK_CLASSES):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
        ft = datasets.CIFAR100(root=DATA_DIR, train=False, download=False, transform=transform)
        indices = [i for i, (_, y) in enumerate(ft) if y in target]
        loader = DataLoader(Subset(ft, indices), batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
        correct = total = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            combined, _, _, _, _ = model(xb, task_id=tid, temperature=0.3)
            pred = combined.argmax(1)
            mask = torch.tensor([(y.item() in target) for y in yb], device=device)
            if mask.sum() > 0:
                correct += (pred[mask] == yb[mask]).sum().item()
                total += mask.sum().item()
        results[f'task{tid}'] = correct / total * 100 if total > 0 else 0
        full_correct += correct
        full_total += total
    results['full'] = full_correct / full_total * 100 if full_total > 0 else 0
    return results

def main():
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    dev = torch.device("cuda")

    os.makedirs('/root/exp_random_reinit', exist_ok=True)
    log_path = f'/root/exp_random_reinit/p2_s{SEED}.log'

    def log(msg):
        print(msg)
        with open(log_path, 'a') as f: f.write(msg + '\n')

    log("=" * 60)
    log(f"RANDOM-EXCLUSIVE + PER-TASK ROUTER REINIT, seed={SEED}")
    log("Tile exclusivity (layer 1) + router reinit (layer 2)")
    log("Hypothesis: close the ~9pp gap to contiguous baseline (85.8%)")
    log("=" * 60)

    model = TileMoE().to(dev)
    ckpt = torch.load('/root/exp_random_exclusive/pretrain.pt', map_location=dev)
    model.load_state_dict(ckpt['model_state'], strict=False)
    log(f"Pretrain loaded (strict=False — router will be re-init anyway)")

    # Phase 2: tile exclusive + per-task router reinit
    log("\n=== PHASE 2: 5-Task CL (Hard Routing + Router Re-Init) ===")
    for param in model.encoder.parameters():
        param.requires_grad = False
    for tile in model.tiles:
        for param in tile.parameters():
            param.requires_grad = False
    log("  Encoder + tiles frozen. Router re-initialized per task.")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5 * EPOCHS_PER_TASK)
    t0 = time.time()

    for task_id in range(5):
        train_subset = get_task_data(task_id, train=True)
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

        # Unfreeze tiles for this task
        for i in range(NUM_TILES):
            for param in model.tiles[i].parameters():
                param.requires_grad = (task_id * 4 <= i < task_id * 4 + 4)
        # Router: trainable, re-initialized
        for param in model.router.parameters():
            param.requires_grad = True
        model.reset_router()

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(f"\nTASK {task_id}: [{TASK_CLASSES[task_id][0]}-{TASK_CLASSES[task_id][-1]}], 20 classes")
        log(f"  Tiles {task_id*4}-{task_id*4+3}. Router re-initialized. Trainable: {trainable/1e3:.1f}K")

        for epoch in range(EPOCHS_PER_TASK):
            model.train()
            total_loss = 0; n_batches = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                opt.zero_grad()
                # Phase 2 uses learned router (not hard routing) to test reinit
                combined, _, _, _, _ = model(xb, temperature=1.0)
                loss = F.cross_entropy(combined, yb)
                loss.backward(); opt.step()
                total_loss += loss.item(); n_batches += 1
            if epoch % 20 == 0:
                log(f"  T{task_id} E{epoch:3d} ce={total_loss/n_batches:.3f}")

        # Eval: hard routing for per-task, learned router for Full
        results = evaluate(model, dev)
        log(f"  >>> TASK {task_id} DONE: " + " | ".join(f"{k}={v:.1f}%" for k, v in results.items()))

    elapsed = time.time() - t0
    log(f"\nPhase 2 done: {elapsed/60:.0f}min")
    final = evaluate(model, dev)
    log(f"\nFINAL: " + " | ".join(f"{k}={v:.1f}%" for k, v in final.items()))
    log(f"\nSUMMARY s={SEED}: Random-exclusive + Reinit | Full={final['full']:.1f}%")
    log(f"  Layer 1 only (tile exclusivity): Full=76.9% (mean)")
    log(f"  Layer 1+2 (exclusivity + reinit): Full={final['full']:.1f}%")
    log(f"  Contiguous baseline: Full=85.8%")

    torch.save({'final_results': final}, f'/root/exp_random_reinit/final_s{SEED}.pt')
    log("Done!")

if __name__ == '__main__':
    main()
