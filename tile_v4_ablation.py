"""Mosaic Tile v4: soft router unfrozen in Phase 2.
Phase 1: same (20 tiles + router, no AR loss, structural anti-clone)
Phase 2: freeze encoder + inactive tiles. Active tiles + ROUTER unfrozen.
Key hypothesis: unfrozen router adapts to task distribution shift,
learning to down-weight stale tiles and focus on active ones.
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
import math, time, os, json, copy, random, argparse
import numpy as np

NUM_TILES = 20; CLASSES_PER_TILE = 5; NUM_CLASSES = 100
FEAT_DIM = 256; BATCH_SIZE = 128; EPOCHS_PER_TASK = 120
T_END = 0.3

CIFAR_MEAN = (0.5071, 0.4867, 0.4408); CIFAR_STD = (0.2675, 0.2565, 0.2761)
tile_class_ranges = [(i * CLASSES_PER_TILE, (i + 1) * CLASSES_PER_TILE) for i in range(NUM_TILES)]

def get_task_ranges(num_tasks):
    total = NUM_CLASSES; base = total // num_tasks; remainder = total % num_tasks
    ranges = []; cur = 0
    for t in range(num_tasks):
        n = base + (1 if t < remainder else 0)
        ranges.append((cur, cur + n)); cur += n
    return ranges

def tiles_for_task(task_range):
    start, end = task_range
    return [i for i, (ts, te) in enumerate(tile_class_ranges) if ts < end and te > start]

# --- WRN-28-10 Encoder ---
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

# --- Mosaic Tile Model v4 ---
class MosaicModelV4(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.tiles = nn.ModuleList([nn.Linear(FEAT_DIM, CLASSES_PER_TILE) for _ in range(NUM_TILES)])
        self.router = nn.Linear(FEAT_DIM, NUM_TILES)  # simple linear router
        self.frozen_tile_ids = set()

    def forward(self, x, temperature=1.0):
        feat = self.encoder(x)
        route_weights = F.softmax(self.router(feat) / temperature, dim=1)  # (B, 20)
        tile_logits = []
        for i, tile in enumerate(self.tiles):
            tile_out = tile(feat)  # (B, 5)
            tile_logits.append(tile_out * route_weights[:, i:i+1])
        logits = torch.cat(tile_logits, dim=1)  # (B, 100)
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

    def unfreeze_router(self):
        for p in self.router.parameters(): p.requires_grad = True

    def get_active_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_tile_sim(self):
        weights = []
        for tile in self.tiles:
            w = tile.weight.detach().flatten()
            weights.append(w)
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

def get_task_data(train_set, test_set, task_range, batch_size=BATCH_SIZE):
    start, end = task_range
    train_idx = [i for i, (_, c) in enumerate(train_set) if start <= c < end]
    test_idx = [i for i, (_, c) in enumerate(test_set) if start <= c < end]
    tl = DataLoader(Subset(train_set, train_idx), batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    vl = DataLoader(Subset(test_set, test_idx), batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return tl, vl

def evaluate(model, test_set, task_ranges, device):
    model.eval()
    correct = [0] * len(task_ranges); total = [0] * len(task_ranges)
    all_correct = 0; all_total = 0
    loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _, _ = model(x)
            pred = logits.argmax(dim=1)
            for ti, (start, end) in enumerate(task_ranges):
                mask = (y >= start) & (y < end)
                if mask.any():
                    correct[ti] += (pred[mask] == y[mask]).sum().item()
                    total[ti] += mask.sum().item()
            all_correct += (pred == y).sum().item(); all_total += y.size(0)
    accs = [correct[i]/max(total[i],1)*100 for i in range(len(task_ranges))]
    return accs, all_correct / max(all_total, 1) * 100

# --- Phase 1: Pre-train all tiles + router (no AR loss) ---
def phase1_pretrain(model, train_loader, device, epochs=200):
    print(f"PHASE 1: Pre-training ({epochs} epochs, no AR loss, structural anti-clone)")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    for epoch in range(epochs):
        model.train(); total_ce = 0; n = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits, route_weights, feat = model(x)
            ce = F.cross_entropy(logits, y)
            loss = ce
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_ce += ce.item(); n += 1
        scheduler.step()
        if epoch % 20 == 0:
            sim = model.get_tile_sim()
            print(f"  P1 E{epoch:3d} ce={total_ce/n:.3f} sim={sim:+.3f} params={model.get_active_params()/1e6:.1f}M")
    sim = model.get_tile_sim()
    print(f"  Phase 1 done. Final tile sim={sim:+.3f}")
    return sim

# --- Phase 2 v4: freeze encoder + inactive tiles, router + active tiles trainable ---
def phase2_v4(model, train_set, test_set, task_ranges, device, args):
    print(f"\nPHASE 2 (v4: unfrozen router + active tiles): {len(task_ranges)} tasks")
    print(f"  Tiles per task: {[len(tiles_for_task(tr)) for tr in task_ranges]}")

    model.freeze_encoder()
    model.freeze_all_tiles()
    # Router stays trainable (not frozen)
    print(f"  Encoder frozen. Router + active tiles trainable per task.")

    results = []
    for task_id, task_range in enumerate(task_ranges):
        active_tiles = tiles_for_task(task_range)
        model.unfreeze_tiles(active_tiles)
        model.unfreeze_router()  # router always trainable
        n_params = model.get_active_params()
        print(f"\n  TASK {task_id}: classes {task_range[0]}-{task_range[1]-1}")
        print(f"    Active tiles: {active_tiles} ({len(active_tiles)} tiles, {n_params/1e6:.3f}M params)")

        train_loader, test_loader = get_task_data(train_set, test_set, task_range)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)

        for epoch in range(EPOCHS_PER_TASK):
            model.train(); total_ce = 0; n = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                logits, route_weights, feat = model(x)
                ce = F.cross_entropy(logits, y)
                optimizer.zero_grad(); ce.backward(); optimizer.step()
                total_ce += ce.item(); n += 1
            if epoch % 20 == 0:
                sim = model.get_tile_sim()
                # Show mean routing weight to active tiles
                rt_active = route_weights[:, active_tiles].sum(dim=1).mean().item()
                print(f"    T{task_id} E{epoch:3d} ce={total_ce/n:.3f} sim={sim:+.3f} active_route={rt_active:.3f}")

        accs, full = evaluate(model, test_set, task_ranges, device)
        acc_str = ' | '.join([f'task{i}={accs[i]:.1f}%' for i in range(len(task_ranges))])
        print(f"    >>> TASK {task_id} DONE: {acc_str} | full={full:.1f}%")

        # Re-freeze tiles for next task, but keep router trainable
        for tid in active_tiles:
            for p in model.tiles[tid].parameters(): p.requires_grad = False
            model.frozen_tile_ids.add(tid)
        results.append((accs, full))

    print(f"\n{'='*60}")
    print(f"FINAL after {len(task_ranges)} tasks (v4: unfrozen router):")
    for ti, (accs, full) in enumerate(results):
        acc_str = ' | '.join([f'task{i}={accs[i]:.1f}%' for i in range(len(task_ranges))])
        print(f"  After T{ti}: {acc_str} | full={full:.1f}%")
    return results[-1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_tasks', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pretrain_epochs', type=int, default=200)
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  Seed: {args.seed}  Tasks: {args.num_tasks}")
    print(f"Tiles: {NUM_TILES} x {CLASSES_PER_TILE} classes = {NUM_CLASSES} total")

    train_set, test_set = get_loaders()
    encoder = SharedEncoder()
    model = MosaicModelV4(encoder).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    print(f"  Tiles: {NUM_TILES} x Linear(256,{CLASSES_PER_TILE}) = {NUM_TILES * 256 * CLASSES_PER_TILE:,} params")
    print(f"  Router: {FEAT_DIM * NUM_TILES:,} params")

    task_ranges = get_task_ranges(args.num_tasks)

    # Phase 1
    full_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    phase1_pretrain(model, full_loader, device, epochs=args.pretrain_epochs)
    ckpt = {'model': model.state_dict(), 'seed': args.seed, 'args': vars(args)}
    torch.save(ckpt, f'/root/phase_mosaic_v4_p{args.pretrain_epochs}_s{args.seed}_prediff.pt')

    # Phase 2 v4
    accs, full = phase2_v4(model, train_set, test_set, task_ranges, device, args)

    torch.save({'model': model.state_dict(), 'accs': accs, 'full': full},
               f'/root/phase_mosaic_v4_t{args.num_tasks}_s{args.seed}_final.pt')
    print(f"\nFinal full accuracy: {full:.1f}%")

if __name__ == '__main__':
    main()
