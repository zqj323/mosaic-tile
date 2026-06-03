"""Mosaic Tile v3 — Shuffled variant: classes randomly assigned to tiles.
Tests whether hard routing requires contiguous class ranges or just a known mapping.
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
import math, time, os, random, argparse
import numpy as np

NUM_TILES = 20; CLASSES_PER_TILE = 5; NUM_CLASSES = 100
FEAT_DIM = 256; BATCH_SIZE = 128; EPOCHS_PER_TASK = 120

CIFAR_MEAN = (0.5071, 0.4867, 0.4408); CIFAR_STD = (0.2675, 0.2565, 0.2761)

# --- Shuffle class-to-tile mapping ---
# Each tile gets 5 random classes. Mapping is FIXED after shuffle.
tile_to_classes = {}  # tile_id -> set of class ids
class_to_tile = {}     # class_id -> tile_id

def init_shuffled_mapping(seed=42):
    """Assign each of 100 classes to exactly 1 of 20 tiles (5 classes/tile)."""
    rng = random.Random(seed + 999)  # different from model seed
    all_classes = list(range(NUM_CLASSES))
    rng.shuffle(all_classes)
    for tile_id in range(NUM_TILES):
        start = tile_id * CLASSES_PER_TILE
        tile_to_classes[tile_id] = set(all_classes[start:start + CLASSES_PER_TILE])
        for cls in tile_to_classes[tile_id]:
            class_to_tile[cls] = tile_id

def tiles_for_task(task_classes):
    """Return tile IDs covering the given set of class IDs."""
    tiles = set()
    for cls in task_classes:
        tiles.add(class_to_tile[cls])
    return sorted(tiles)

def get_task_ranges(num_tasks, seed=42):
    """Standard split: 100 classes divided evenly across tasks. Shuffled mapping inside."""
    rng = random.Random(seed)
    all_classes = list(range(NUM_CLASSES))
    rng.shuffle(all_classes)
    total = NUM_CLASSES; base = total // num_tasks; remainder = total % num_tasks
    ranges = []; cur = 0
    for t in range(num_tasks):
        n = base + (1 if t < remainder else 0)
        ranges.append(all_classes[cur:cur + n]); cur += n
    return ranges

# --- WRN-28-10 ---
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
    def __init__(self, depth=28, widen=10):
        super().__init__()
        n = (depth - 4) // 6; k = widen; nCh = [16, 16*k, 32*k, 64*k]
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

class MosaicModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.tiles = nn.ModuleList([nn.Linear(FEAT_DIM, CLASSES_PER_TILE) for _ in range(NUM_TILES)])
        self.router = nn.Linear(FEAT_DIM, NUM_TILES)
        self.frozen_tile_ids = set()
        self.active_tiles = None
        # Build tile-order → class-order remap index
        self.register_buffer('remap_idx', torch.zeros(NUM_CLASSES, dtype=torch.long))
        for tile_id in range(NUM_TILES):
            classes_in_tile = sorted(tile_to_classes[tile_id])
            for j, cls in enumerate(classes_in_tile):
                self.remap_idx[tile_id * CLASSES_PER_TILE + j] = cls

    def set_active_tiles(self, tile_ids):
        self.active_tiles = set(tile_ids) if tile_ids is not None else None

    def forward(self, x, temperature=1.0):
        feat = self.encoder(x)
        if self.active_tiles is not None:
            tile_logits = []
            for i, tile in enumerate(self.tiles):
                if i in self.active_tiles:
                    tile_logits.append(tile(feat))
                else:
                    tile_logits.append(torch.full((feat.size(0), CLASSES_PER_TILE), float('-inf'), device=feat.device))
            logits_tile_order = torch.cat(tile_logits, dim=1)  # (B, 100) in tile order
            # Remap to class order for CE loss compatibility
            logits = logits_tile_order[:, self.remap_idx]
            route_weights = torch.zeros(feat.size(0), NUM_TILES, device=feat.device)
            for i in self.active_tiles: route_weights[:, i] = 1.0 / len(self.active_tiles)
            return logits, route_weights, feat
        else:
            route_weights = F.softmax(self.router(feat) / temperature, dim=1)
            tile_logits = [tile(feat) * route_weights[:, i:i+1] for i, tile in enumerate(self.tiles)]
            logits_tile_order = torch.cat(tile_logits, dim=1)  # (B, 100) in tile order
            logits = logits_tile_order[:, self.remap_idx]  # remap to class order
            return logits, route_weights, feat

    def freeze_encoder(self):
        for p in self.encoder.parameters(): p.requires_grad = False

    def freeze_all_tiles(self):
        for i, tile in enumerate(self.tiles):
            for p in tile.parameters(): p.requires_grad = False
            self.frozen_tile_ids.add(i)

    def unfreeze_tiles(self, tile_ids):
        for tid in tile_ids:
            if tid in self.frozen_tile_ids:
                for p in self.tiles[tid].parameters(): p.requires_grad = True
                self.frozen_tile_ids.discard(tid)

    def get_active_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_tile_sim(self):
        weights = [tile.weight.detach().flatten() for tile in self.tiles]
        W = torch.stack(weights); W_norm = F.normalize(W, dim=1)
        sim_matrix = W_norm @ W_norm.T
        mask = ~torch.eye(NUM_TILES, dtype=torch.bool, device=W.device)
        return sim_matrix[mask].mean().item()

# --- Data ---
def get_loaders():
    tf = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    tf_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    train_set = datasets.CIFAR100(root='/root/data', train=True, download=False, transform=tf)
    test_set = datasets.CIFAR100(root='/root/data', train=False, download=False, transform=tf_test)
    return train_set, test_set

def get_task_data(train_set, test_set, task_classes, batch_size=BATCH_SIZE):
    train_idx = [i for i, (_, c) in enumerate(train_set) if c in task_classes]
    test_idx = [i for i, (_, c) in enumerate(test_set) if c in task_classes]
    tl = DataLoader(Subset(train_set, train_idx), batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    vl = DataLoader(Subset(test_set, test_idx), batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return tl, vl

# --- Remap logits: tile outputs in tile order, but classes within each tile are fixed ---
def logits_to_pred(logits, batch_size):
    """Convert tile logits (tile 0: 5 classes, tile 1: 5 classes, ...) to class predictions.
    The ith position in logits corresponds to the class assigned to that tile-position.
    """
    # Build reverse mapping: position_in_logits -> class_id
    pos_to_class = {}
    for tile_id in range(NUM_TILES):
        classes_in_tile = sorted(tile_to_classes[tile_id])
        for j, cls in enumerate(classes_in_tile):
            pos_to_class[tile_id * CLASSES_PER_TILE + j] = cls

    # Reorder logits so position matches class_id
    B = logits.size(0)
    class_logits = torch.zeros(B, NUM_CLASSES, device=logits.device)
    for pos in range(NUM_CLASSES):
        cls = pos_to_class[pos]
        class_logits[:, cls] = logits[:, pos]
    return class_logits

def evaluate(model, test_set, task_class_sets, device):
    model.eval()
    correct = [0] * len(task_class_sets); total = [0] * len(task_class_sets)
    all_correct = all_total = 0
    loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            for ti, task_classes in enumerate(task_class_sets):
                mask = torch.isin(y, torch.tensor(task_classes, device=device))
                if mask.any():
                    active = tiles_for_task(task_classes)
                    model.set_active_tiles(active)
                    logits, _, _ = model(x)  # already in class order
                    pred = logits.argmax(dim=1)
                    correct[ti] += (pred[mask] == y[mask]).sum().item()
                    total[ti] += mask.sum().item()
                    all_correct += (pred[mask] == y[mask]).sum().item()
                    all_total += mask.sum().item()
    accs = [correct[i]/max(total[i],1)*100 for i in range(len(task_class_sets))]
    return accs, all_correct/max(all_total,1)*100

def phase1_pretrain(model, train_loader, device, epochs=200):
    print(f"PHASE 1: Pre-training ({epochs} epochs, no AR loss)")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    for epoch in range(epochs):
        model.train(); total_ce = n = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits, _, _ = model(x)
            # Phase 1 uses soft router. Classes still shuffled but router learns the mapping.
            ce = F.cross_entropy(logits, y)
            optimizer.zero_grad(); ce.backward(); optimizer.step()
            total_ce += ce.item(); n += 1
        scheduler.step()
        if epoch % 20 == 0: print(f"  P1 E{epoch:3d} ce={total_ce/n:.3f} sim={model.get_tile_sim():+.3f}")
    print(f"  Phase 1 done. sim={model.get_tile_sim():+.3f}")

def phase2_hard(model, train_set, test_set, task_class_sets, device):
    print(f"\nPHASE 2 (shuffled, hard routing): {len(task_class_sets)} tasks")
    tile_counts = [len(tiles_for_task(tcs)) for tcs in task_class_sets]
    print(f"  Tiles per task: {tile_counts}")

    model.freeze_encoder(); model.freeze_all_tiles()

    results = []
    for task_id, task_classes in enumerate(task_class_sets):
        active_tiles = tiles_for_task(task_classes)
        model.unfreeze_tiles(active_tiles); model.set_active_tiles(active_tiles)
        n_params = model.get_active_params()
        print(f"\n  TASK {task_id}: {len(task_classes)} classes, tiles {active_tiles} ({n_params/1e6:.3f}M params)")
        print(f"    Task classes: {sorted(task_classes)[:10]}{'...' if len(task_classes)>10 else ''}")
        print(f"    Tile->classes: {{{ {t: sorted(tile_to_classes[t]) for t in active_tiles} }}}")

        train_loader, _ = get_task_data(train_set, test_set, task_classes)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)

        for epoch in range(EPOCHS_PER_TASK):
            model.train(); total_ce = n = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                logits, _, _ = model(x)
                ce = F.cross_entropy(logits, y)
                optimizer.zero_grad(); ce.backward(); optimizer.step()
                total_ce += ce.item(); n += 1
            if epoch % 20 == 0:
                print(f"    T{task_id} E{epoch:3d} ce={total_ce/n:.3f} sim={model.get_tile_sim():+.3f}")

        accs, full = evaluate(model, test_set, task_class_sets, device)
        acc_str = ' | '.join([f'task{i}={accs[i]:.1f}%' for i in range(len(task_class_sets))])
        print(f"    >>> TASK {task_id} DONE: {acc_str} | full={full:.1f}%")

        for tid in active_tiles:
            for p in model.tiles[tid].parameters(): p.requires_grad = False
            model.frozen_tile_ids.add(tid)
        results.append((accs, full))

    print(f"\n{'='*60}")
    print(f"FINAL (shuffled):")
    for ti, (accs, full) in enumerate(results):
        acc_str = ' | '.join([f'task{i}={accs[i]:.1f}%' for i in range(len(task_class_sets))])
        print(f"  After T{ti}: {acc_str} | full={full:.1f}%")
    return results[-1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_tasks', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pretrain_epochs', type=int, default=200)
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    init_shuffled_mapping(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  Seed: {args.seed}  Tasks: {args.num_tasks}")
    print(f"Shuffled mapping: each tile → 5 random classes")
    print(f"Sample: tile 0 → {sorted(tile_to_classes[0])}")
    print(f"Sample: tile 10 → {sorted(tile_to_classes[10])}")

    train_set, test_set = get_loaders()
    encoder = SharedEncoder(); model = MosaicModel(encoder).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    task_class_sets = get_task_ranges(args.num_tasks, args.seed)
    for i, tcs in enumerate(task_class_sets):
        tiles = tiles_for_task(tcs)
        print(f"  T{i}: {len(tcs)} classes → tiles {tiles}")

    full_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    phase1_pretrain(model, full_loader, device, epochs=args.pretrain_epochs)
    torch.save({'model': model.state_dict(), 'seed': args.seed}, f'/root/tile_shuffle_p{args.pretrain_epochs}_s{args.seed}_prediff.pt')

    accs, full = phase2_hard(model, train_set, test_set, task_class_sets, device)
    torch.save({'model': model.state_dict(), 'accs': accs, 'full': full}, f'/root/tile_shuffle_t{args.num_tasks}_s{args.seed}_final.pt')
    print(f"\nFinal full accuracy: {full:.1f}%")

if __name__ == '__main__':
    main()
