"""Random mutually-exclusive class grouping for Tile v3.
Tests: "Does Tile v3 require consecutive classes, or just tile-exclusive task mappings?"
100 classes randomly shuffled → 5 disjunct groups of 20.
Each task's 20 classes → sorted → evenly split across 4 tiles (5 classes each).
Hard routing: class→tile via lookup table (not floor division).
Hypothesis: if tiles are exclusive, Tile v3 works regardless of class ordering.
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
import math, time, os, random
import numpy as np

NUM_TILES = 20; TILE_SIZE = 5; NUM_CLASSES = 100; FEAT_DIM = 256
BATCH_SIZE = 128; EPOCHS_PER_TASK = 120
CIFAR_MEAN = (0.5071, 0.4867, 0.4408); CIFAR_STD = (0.2675, 0.2565, 0.2761)
SEED = 42; DATA_DIR = '/root/data'

# ── Random mutually-exclusive class assignment ──
random.seed(SEED); np.random.seed(SEED)
ALL_CLASSES = list(range(100))
np.random.shuffle(ALL_CLASSES)
# 5 mutually exclusive groups of 20 random classes
TASK_CLASSES = [sorted(ALL_CLASSES[i*20:(i+1)*20]) for i in range(5)]
# Each task: 20 classes → 4 tiles × 5 classes
# Build class→tile lookup: (task_id, class_id) → tile index
CLASS_TO_TILE = {}
for tid, classes in enumerate(TASK_CLASSES):
    for j, cls in enumerate(classes):
        tile_idx = tid * 4 + (j // 5)  # 4 tiles per task
        CLASS_TO_TILE[cls] = tile_idx

# ── Model (same as Tile v3, but with lookup-based hard routing) ──
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

class TileWithLookupRouting(nn.Module):
    """Tile MoE with lookup-based hard routing for random class assignments."""
    def __init__(self):
        super().__init__()
        self.encoder = SharedEncoder()
        self.tiles = nn.ModuleList([nn.Linear(FEAT_DIM, TILE_SIZE) for _ in range(NUM_TILES)])
        self.router = nn.Linear(FEAT_DIM, NUM_TILES)

    def forward(self, x, task_id=None, temperature=1.0):
        feat = self.encoder(x)
        tile_logits = []
        for tile in self.tiles:
            tile_logits.append(tile(feat))  # each [B, 5]
        tile_logits_stack = torch.stack(tile_logits, dim=1)  # [B, 20, 5]

        if task_id is not None:
            # Hard routing: only active tiles for this task
            active_tiles = list(range(task_id * 4, task_id * 4 + 4))
            weights = torch.zeros(x.size(0), NUM_TILES, device=x.device)
            weights[:, active_tiles] = 1.0 / len(active_tiles)
            combined = torch.zeros(x.size(0), NUM_CLASSES, device=x.device)
            for ti in active_tiles:
                cls_start = TASK_CLASSES[task_id][0]
                for j, cls in enumerate(TASK_CLASSES[task_id]):
                    if CLASS_TO_TILE[cls] == ti:
                        combined[:, cls] += weights[:, ti] * tile_logits_stack[:, ti, j % 5]
            return combined, weights, tile_logits_stack, None, feat

        # Phase 1: learned soft routing
        route = self.router(feat) / temperature
        weights = F.softmax(route, dim=1)
        combined = torch.zeros(x.size(0), NUM_CLASSES, device=x.device)
        for i in range(NUM_TILES):
            start = i * TILE_SIZE
            combined[:, start:start+TILE_SIZE] += weights[:, i:i+1] * tile_logits[i]
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
def evaluate_split(model, device):
    model.eval()
    results = {}
    full_correct = 0; full_total = 0
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

@torch.no_grad()
def compute_sim(model, loader, device):
    model.eval()
    all_logits = []
    for xb, yb in loader:
        _, _, logits_list, _, _ = model(xb.to(device), temperature=1.0)
        # logits_list is [B, 20, 5] — already stacked
        all_logits.append(logits_list.cpu())
    logits = torch.cat(all_logits, 0)  # [N, 20, 5]
    tile_means = logits.mean(dim=0)  # [20, 5]
    return np.mean([F.cosine_similarity(tile_means[i].flatten().unsqueeze(0),
                    tile_means[j].flatten().unsqueeze(0)).item()
                    for i in range(NUM_TILES) for j in range(i+1, NUM_TILES)])

def main():
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    dev = torch.device("cuda")

    os.makedirs('/root/exp_random_exclusive', exist_ok=True)
    log_path = '/root/exp_random_exclusive/experiment.log'

    def log(msg):
        print(msg)
        with open(log_path, 'a') as f: f.write(msg + '\n')

    log("=" * 60)
    log("RANDOM MUTUALLY-EXCLUSIVE TASK GROUPING FOR TILE v3")
    log("100 classes randomly shuffled → 5 disjunct groups of 20")
    log("Tiles: strictly exclusive (4 per task). Hypothesis: works.")
    log("=" * 60)
    for tid, classes in enumerate(TASK_CLASSES):
        log(f"  T{tid}: classes {classes[:5]}...{classes[-5:]} ({len(classes)} classes)")

    model = TileWithLookupRouting().to(dev)
    total_p = sum(p.numel() for p in model.parameters())
    log(f"\nTotal params: {total_p/1e6:.1f}M")

    SKIP_PHASE1 = os.environ.get('SKIP_PHASE1', '0') == '1'

    if SKIP_PHASE1 and os.path.exists('/root/exp_random_exclusive/pretrain.pt'):
        log("\n=== PHASE 1: SKIPPED (loading existing pretrain checkpoint) ===")
        ckpt = torch.load('/root/exp_random_exclusive/pretrain.pt', map_location=dev)
        model.load_state_dict(ckpt['model_state'])
        test_set = get_full_data(train=False)
        sim_loader = DataLoader(Subset(test_set, np.random.choice(10000, 2000, replace=False)),
                               batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
        mean_s = compute_sim(model, sim_loader, dev)
        log(f"  Loaded pretrain. sim={mean_s:.3f}")
    else:

    # Phase 1: Pre-training on full CIFAR-100 (regular order, no AR loss)
    log("\n=== PHASE 1: Pre-training on Full CIFAR-100 (200 epochs, no AR loss) ===")
    full_train = get_full_data(train=True)
    train_loader = DataLoader(full_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_set = get_full_data(train=False)
    sim_loader = DataLoader(Subset(test_set, np.random.choice(10000, 2000, replace=False)),
                           batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

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
            mean_s = compute_sim(model, sim_loader, dev)
            log(f"  P1 E{epoch:3d} ce={total_loss/n_batches:.3f} sim={mean_s:.3f}")

    elapsed = time.time() - t0
    mean_s = compute_sim(model, sim_loader, dev)
    results = evaluate_split(model, dev)
    log(f"  Phase 1 done: {elapsed/60:.0f}min full={results['full']:.1f}% sim={mean_s:.3f}")

    # Save pre-train checkpoint
    torch.save({'model_state': {k: v.clone() for k, v in model.state_dict().items()}},
               '/root/exp_random_exclusive/pretrain.pt')

    # Phase 2: 5-task CL with hard routing + mutually exclusive tiles
    log("\n=== PHASE 2: 5-Task CL (Mutually Exclusive Tiles, Random Classes) ===")
    for param in model.encoder.parameters():
        param.requires_grad = False
    for i, tile in enumerate(model.tiles):
        # All tiles frozen initially
        for param in tile.parameters():
            param.requires_grad = False
    log("  Encoder + all tiles frozen. Hard routing (lookup-based).")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5 * EPOCHS_PER_TASK)
    t0 = time.time()

    for task_id in range(5):
        train_subset = get_task_data(task_id, train=True)
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

        # Unfreeze only this task's 4 tiles
        for i in range(NUM_TILES):
            for param in model.tiles[i].parameters():
                param.requires_grad = (task_id * 4 <= i < task_id * 4 + 4)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(f"\nTASK {task_id}: classes {TASK_CLASSES[task_id][:3]}...{TASK_CLASSES[task_id][-3:]}")
        log(f"  Active tiles: {task_id*4}-{task_id*4+3}. Trainable: {trainable/1e3:.1f}K params")

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
                mean_s = compute_sim(model, sim_loader, dev)
                log(f"  T{task_id} E{epoch:3d} ce={total_loss/n_batches:.3f} sim={mean_s:.3f}")

        results = evaluate_split(model, dev)
        mean_s = compute_sim(model, sim_loader, dev)
        log(f"  >>> TASK {task_id} DONE: " + " | ".join(f"{k}={v:.1f}%" for k, v in results.items()))
        log(f"  sim={mean_s:.3f}")

    elapsed = time.time() - t0
    log(f"\nPhase 2 done: {elapsed/60:.0f}min total")

    final = evaluate_split(model, dev)
    mean_s = compute_sim(model, sim_loader, dev)
    log(f"\nFINAL: " + " | ".join(f"{k}={v:.1f}%" for k, v in final.items()))
    log(f"FINAL sim={mean_s:.3f}")
    log(f"\nSUMMARY: Random-exclusive | Full={final['full']:.1f}% | sim={mean_s:.3f}")
    log(f"  Contiguous baseline (Tile v3): Full=85.8% | Original shuffle (shared tiles): Full=16.7%")
    log(f"  Prediction: mutually-exclusive → Full close to 85.8% (tile exclusivity satisfied)")

    torch.save({'model_state': model.state_dict(), 'final_results': final},
               '/root/exp_random_exclusive/final.pt')
    log("Done!")

if __name__ == '__main__':
    main()
