"""Tile v3 single-task baseline: full CIFAR-100, 20 tiles, learned soft router.
Answers: "Does Tile v3 sacrifice single-task performance for CL stability?"
If Tile v3 single-task >= expert protocol (4 experts, full 100-class), then Tile v3
is not a trade-off — it's strictly better.
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
import math, time, os, random
import numpy as np

NUM_TILES = 20; TILE_SIZE = 5; NUM_CLASSES = 100; FEAT_DIM = 256
BATCH_SIZE = 128; SEED = 42; DATA_DIR = '/root/data'
CIFAR_MEAN = (0.5071, 0.4867, 0.4408); CIFAR_STD = (0.2675, 0.2565, 0.2761)

# ── Model ──
def make_classifier(capacity_idx, num_classes):
    if capacity_idx == 0: return nn.Linear(FEAT_DIM, num_classes)
    elif capacity_idx == 1: return nn.Sequential(nn.Linear(FEAT_DIM, 128), nn.ReLU(), nn.Linear(128, num_classes))
    elif capacity_idx == 2: return nn.Sequential(nn.Linear(FEAT_DIM, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, num_classes))
    elif capacity_idx == 3: return nn.Sequential(nn.Linear(FEAT_DIM, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, num_classes))

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

class SingleTaskTileMoE(nn.Module):
    """20 tiles × 5 classes each with learned soft routing on full CIFAR-100."""
    def __init__(self, capacity_idx=0):  # capacity_idx 0 = Linear for fairness
        super().__init__()
        self.encoder = SharedEncoder()
        # 20 tiles, each Linear(256, 5)
        self.tiles = nn.ModuleList([nn.Linear(FEAT_DIM, TILE_SIZE) for _ in range(NUM_TILES)])
        # Router: shared Linear(256, 20)
        self.router = nn.Linear(FEAT_DIM, NUM_TILES)
    def forward(self, x, temperature=1.0):
        feat = self.encoder(x)
        # Each tile produces [B, 5] logits
        all_logits = []
        for i, tile in enumerate(self.tiles):
            logits = tile(feat)  # [B, 5]
            all_logits.append(logits)
        # Route: softmax over 20 tiles
        route = self.router(feat) / temperature  # [B, 20]
        weights = F.softmax(route, dim=1)  # [B, 20]

        # Combine: weight each tile's 5-class logits into a 100-class vector
        combined = torch.zeros(x.size(0), NUM_CLASSES, device=x.device)
        for i in range(NUM_TILES):
            start = i * TILE_SIZE
            combined[:, start:start+TILE_SIZE] += weights[:, i:i+1] * all_logits[i]
        return combined, weights, all_logits, route, feat

def get_full_data(train=True):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    return datasets.CIFAR100(root=DATA_DIR, train=train, download=False, transform=transform)

@torch.no_grad()
def evaluate(model, device):
    model.eval()
    test_set = get_full_data(train=False)
    loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        combined, _, _, _, _ = model(xb, temperature=1.0)
        correct += (combined.argmax(1) == yb).sum().item(); total += yb.size(0)
    return correct / total * 100

@torch.no_grad()
def compute_tile_sim(model, loader, device):
    model.eval()
    all_logits = []
    for xb, yb in loader:
        _, _, logits_list, _, _ = model(xb.to(device), temperature=1.0)
        # logits_list is list of [B, 5]; stack to [B, 20, 5]
        stacked = torch.stack(logits_list, dim=1)  # [B, 20, 5]
        all_logits.append(stacked.cpu())
    logits = torch.cat(all_logits, 0)
    # Pairwise sim between tile output means
    tile_means = logits.mean(dim=0)  # [20, 5]
    sims = []
    for i in range(NUM_TILES):
        for j in range(i+1, NUM_TILES):
            s = F.cosine_similarity(tile_means[i].flatten().unsqueeze(0),
                                    tile_means[j].flatten().unsqueeze(0)).item()
            sims.append(s)
    return np.mean(sims), sims

def main():
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    dev = torch.device("cuda")

    os.makedirs('/root/exp_tile_single_task', exist_ok=True)
    log_path = '/root/exp_tile_single_task/experiment.log'

    def log(msg):
        print(msg)
        with open(log_path, 'a') as f: f.write(msg + '\n')

    log("=" * 60)
    log("TILE v3 SINGLE-TASK BASELINE: Full CIFAR-100 (no task split)")
    log(f"{NUM_TILES} tiles × {TILE_SIZE} classes, learned soft router")
    log("Goal: does Tile architecture sacrifice single-task accuracy?")
    log("=" * 60)

    model = SingleTaskTileMoE(capacity_idx=0).to(dev)
    total_p = sum(p.numel() for p in model.parameters())
    log(f"Total params: {total_p/1e6:.1f}M")
    log(f"Tile output params: {sum(p.numel() for p in model.tiles.parameters())/1e3:.1f}K")
    log(f"Router params: {sum(p.numel() for p in model.router.parameters())}")

    full_train = get_full_data(train=True)
    train_loader = DataLoader(full_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_set = get_full_data(train=False)
    sim_loader = DataLoader(Subset(test_set, np.random.choice(10000, 2000, replace=False)),
                           batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
    t0 = time.time()

    best_acc = 0
    for epoch in range(200):
        model.train()
        total_loss = 0; n_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            combined, weights, logits, route, feat = model(xb, temperature=1.0)
            loss = F.cross_entropy(combined, yb)
            loss.backward(); opt.step()
            total_loss += loss.item(); n_batches += 1
        sched.step()

        if epoch % 20 == 0:
            acc = evaluate(model, dev)
            mean_s, sims = compute_tile_sim(model, sim_loader, dev)
            log(f"  E{epoch:3d} ce={total_loss/n_batches:.3f} acc={acc:.1f}% sim={mean_s:.3f}")
            if acc > best_acc: best_acc = acc

    elapsed = time.time() - t0
    final_acc = evaluate(model, dev)
    mean_s, sims = compute_tile_sim(model, sim_loader, dev)
    log(f"\n  Final: acc={final_acc:.1f}% best={best_acc:.1f}% sim={mean_s:.3f}")
    log(f"  Time: {elapsed/60:.0f}min")
    log(f"\nSUMMARY: Tile single-task acc={final_acc:.1f}% | sim={mean_s:.3f}")
    log(f"  Expert protocol Phase 1 (4 expert, AR=0.1): full=49.9% sim=-0.136")
    log(f"  Expert protocol Phase 1 (4 expert, CE-only): full=54.5% sim=0.944")

if __name__ == '__main__':
    main()
