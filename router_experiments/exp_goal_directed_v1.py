"""Goal-Directed AI v1: Low-entropy pre-training vs uniform pre-training.
Tests core claim: goal-directed (concentrated) data deviates from Gaussian →
standard MoE collapses → anti-Gaussian operators (AR loss + freeze) rescue.

Phase 1A (Uniform): standard CIFAR-100, all 100 classes equally represented
Phase 1B (Goal-directed): concentrated distribution — only 20 classes get 80% of samples
Phase 2: 5-task CL on remaining 80 classes

Hypothesis:
- Uniform P1 → AR pre-diff works normally (baseline)
- Goal-directed P1 → CE-only model collapses HARDER (lower entropy = stronger alignment?)
- Goal-directed P1 + AR loss → breaks Gaussian attractor despite non-Gaussian data
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms, datasets
import time, os, random, numpy as np

NUM_EXPERTS = 4; NUM_CLASSES = 100; FEAT_DIM = 256
BATCH_SIZE = 128; EPOCHS_PER_TASK = 120
CIFAR_MEAN = (0.5071, 0.4867, 0.4408); CIFAR_STD = (0.2675, 0.2565, 0.2761)
SEED = 42; DATA_DIR = '/root/data'

# ── Model (same as pre-diff+freeze v3) ──
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

class MutualExpert(nn.Module):
    def __init__(self, feat_dim, num_classes, num_experts, capacity_idx):
        super().__init__()
        self.classifier = make_classifier(capacity_idx, num_classes)
        self.activation = nn.Linear(feat_dim, num_experts)
    def forward(self, x):
        return self.classifier(x), torch.sigmoid(self.activation(x))

class ContinualLifecycleMoE(nn.Module):
    def __init__(self, capacity_idx):
        super().__init__()
        self.encoder = SharedEncoder()
        self.experts = nn.ModuleList([MutualExpert(FEAT_DIM, NUM_CLASSES, NUM_EXPERTS, capacity_idx) for _ in range(NUM_EXPERTS)])
        self.register_buffer('usage_ema', torch.ones(NUM_EXPERTS) / NUM_EXPERTS)
        self.register_buffer('activation_ema', torch.ones(NUM_EXPERTS, NUM_EXPERTS) / NUM_EXPERTS)

    def freeze_for_task(self, task_id):
        active = task_id % NUM_EXPERTS
        for i, expert in enumerate(self.experts):
            frozen = (i != active)
            for param in expert.classifier.parameters(): param.requires_grad = not frozen
            for param in expert.activation.parameters(): param.requires_grad = False
        return active

    def forward(self, x, temperature=1.0):
        B = x.size(0); feat = self.encoder(x)
        all_logits, all_act = [], []
        for expert in self.experts:
            logits, act = expert(feat)
            all_logits.append(logits); all_act.append(act)
        logits = torch.stack(all_logits, dim=1)
        act_mat = torch.stack(all_act, dim=1) / temperature
        with torch.no_grad():
            usage_now = act_mat.mean(dim=1).mean(dim=0)
            self.usage_ema = 0.9 * self.usage_ema + 0.1 * usage_now
            self.activation_ema = 0.9 * self.activation_ema + 0.1 * act_mat.mean(dim=0)
            dead_mask = (self.usage_ema < 0.35).float()
            act_mat = act_mat + 0.4 * dead_mask.unsqueeze(0).unsqueeze(0)
        incoming = act_mat.sum(dim=1)
        weights = F.softmax(incoming, dim=1)
        combined = (weights.unsqueeze(-1) * logits).sum(dim=1)
        return combined, weights, logits, act_mat, feat

# ── Data utilities ──
def get_full_data(train=True):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    return datasets.CIFAR100(root=DATA_DIR, train=train, download=False, transform=transform)

def get_concentrated_sampler(dataset, primary_classes, primary_weight=0.8):
    """Create sampler where primary_classes get primary_weight of total samples."""
    targets = np.array(dataset.targets)
    n_primary = len(primary_classes)
    other_classes = [c for c in range(100) if c not in primary_classes]
    n_other = len(other_classes)

    weights = np.ones(100)
    # primary_weight of samples → primary classes
    # (1-primary_weight) of samples → other classes
    p_primary_per_class = primary_weight / n_primary
    p_other_per_class = (1 - primary_weight) / n_other

    for cls in primary_classes:
        weights[cls] = p_primary_per_class / (targets == cls).mean()
    for cls in other_classes:
        weights[cls] = p_other_per_class / (targets == cls).mean()

    sample_weights = weights[targets]
    return WeightedRandomSampler(sample_weights, len(dataset), replacement=True)

def get_task_data(task_id, train=True, num_tasks=5):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    full = datasets.CIFAR100(root=DATA_DIR, train=train, download=False, transform=transform)
    base = 100 // num_tasks
    start = task_id * base; end = start + base
    indices = [i for i, (_, y) in enumerate(full) if start <= y < end]
    return Subset(full, indices), start, end

@torch.no_grad()
def evaluate(model, device, num_tasks=5):
    model.eval()
    results = {}
    for task_id in range(num_tasks):
        test_subset, start, end = get_task_data(task_id, train=False, num_tasks=num_tasks)
        loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
        correct = total = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            combined, _, _, _, _ = model(xb, temperature=0.3)
            pred = combined.argmax(1)
            mask = (yb >= start) & (yb < end)
            if mask.sum() > 0:
                correct += (pred[mask] == yb[mask]).sum().item()
                total += mask.sum().item()
        results[f'task{task_id}'] = correct / total * 100 if total > 0 else 0
    full_test = get_full_data(train=False)
    full_loader = DataLoader(full_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    correct = total = 0
    for xb, yb in full_loader:
        xb, yb = xb.to(device), yb.to(device)
        combined, _, _, _, _ = model(xb, temperature=0.3)
        pred = combined.argmax(1)
        correct += (pred == yb).sum().item(); total += yb.size(0)
    results['full'] = correct / total * 100
    return results

@torch.no_grad()
def compute_sim(model, loader, device):
    model.eval()
    all_logits = []
    for xb, yb in loader:
        _, _, logits, _, _ = model(xb.to(device), temperature=0.3)
        all_logits.append(logits.cpu())
    logits = torch.cat(all_logits, 0)
    return np.mean([F.cosine_similarity(logits[:,i,:].mean(0).unsqueeze(0),
                    logits[:,j,:].mean(0).unsqueeze(0)).item()
                    for i in range(NUM_EXPERTS) for j in range(i+1, NUM_EXPERTS)])

def ar_loss_fn(logits, lambda_ar=0.1):
    total_ar = 0.0; count = 0
    for i in range(NUM_EXPERTS):
        for j in range(i+1, NUM_EXPERTS):
            total_ar += F.relu(F.cosine_similarity(logits[:,i,:], logits[:,j,:], dim=1)).mean(); count += 1
    return lambda_ar * total_ar / count if count > 0 else 0.0

def run_experiment(label, use_ar, use_concentrated, primary_classes, dev):
    log_path = f'/root/exp_goal_directed/{label}.log'
    def log(msg):
        print(msg)
        with open(log_path, 'a') as f: f.write(msg + '\n')

    log("=" * 60)
    log(f"GOAL-DIRECTED v1: {label}")
    log(f"  AR loss: {use_ar}, Concentrated P1: {use_concentrated}")
    log(f"  Primary classes (80% mass): {primary_classes[:5]}...")
    log("=" * 60)

    model = ContinualLifecycleMoE(3).to(dev)
    log(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # Phase 1
    dist_label = "concentrated (goal-directed, 80% mass on 20 classes)" if use_concentrated else "uniform (passive observation)"
    log(f"\n=== PHASE 1: 200 epochs, {dist_label} ===")

    full_train = get_full_data(train=True)
    if use_concentrated:
        sampler = get_concentrated_sampler(full_train, primary_classes, primary_weight=0.8)
        train_loader = DataLoader(full_train, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True)
    else:
        train_loader = DataLoader(full_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    test_set = get_full_data(train=False)
    sim_loader = DataLoader(Subset(test_set, np.random.choice(10000, 2000, replace=False)),
                           batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
    t0 = time.time()

    for epoch in range(200):
        model.train()
        total_ce = 0; total_ar = 0; n_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            combined, weights, logits, act_mat, _ = model(xb, temperature=1.0)
            ce = F.cross_entropy(combined, yb)
            if use_ar:
                ar = ar_loss_fn(logits, 0.1)
                total_ar += ar.item()
            else:
                ar = 0
            loss = ce + (ar if use_ar else 0)
            loss.backward(); opt.step()
            total_ce += ce.item(); n_batches += 1
        sched.step()
        if epoch % 20 == 0:
            mean_s = compute_sim(model, sim_loader, dev)
            ar_str = f" ar={total_ar/n_batches:.4f}" if use_ar else ""
            log(f"  P1 E{epoch:3d} ce={total_ce/n_batches:.3f}{ar_str} sim={mean_s:.3f}")

    elapsed = time.time() - t0
    mean_s = compute_sim(model, sim_loader, dev)
    log(f"  Phase 1 done: {elapsed/60:.0f}min sim={mean_s:.3f}")

    # Phase 2: CL on remaining 80 classes (or all 100 for uniform)
    log("\n=== PHASE 2: 5-Task CL (freeze encoder + experts) ===")
    for param in model.encoder.parameters():
        param.requires_grad = False
    log("  Encoder frozen from Task 0. Router frozen. Expert classifiers frozen.")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5 * EPOCHS_PER_TASK)

    for task_id in range(5):
        train_subset, task_start, task_end = get_task_data(task_id, train=True)
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

        model.freeze_for_task(task_id)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(f"\nTASK {task_id}: classes {task_start}-{task_end-1}, active expert {task_id % 4}")
        log(f"  Trainable: {trainable/1e6:.1f}M")

        for epoch in range(EPOCHS_PER_TASK):
            model.train()
            total_loss = 0; n_batches = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                opt.zero_grad()
                combined, weights, logits, act_mat, _ = model(xb, temperature=1.0)
                loss = F.cross_entropy(combined, yb)
                loss.backward(); opt.step()
                total_loss += loss.item(); n_batches += 1
            if epoch % 30 == 0:
                mean_s = compute_sim(model, sim_loader, dev)
                log(f"  T{task_id} E{epoch:3d} ce={total_loss/n_batches:.3f} sim={mean_s:.3f}")

        results = evaluate(model, dev)
        mean_s = compute_sim(model, sim_loader, dev)
        log(f"  >>> TASK {task_id} DONE: " + " | ".join(f"{k}={v:.1f}%" for k, v in results.items()))
        log(f"  sim={mean_s:.3f}")

    elapsed = time.time() - t0
    log(f"\nPhase 2 done: {elapsed/60:.0f}min")

    final = evaluate(model, dev)
    mean_s = compute_sim(model, sim_loader, dev)
    log(f"\nFINAL: " + " | ".join(f"{k}={v:.1f}%" for k, v in final.items()))
    log(f"FINAL sim={mean_s:.3f}")
    log(f"\nRESULT {label}: Full={final['full']:.1f}% | sim={mean_s:.3f} | AR={use_ar} | conc={use_concentrated}")

    torch.save({'final': final, 'sim': mean_s}, f'/root/exp_goal_directed/{label}.pt')
    return final['full'], mean_s

def main():
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    dev = torch.device("cuda")

    os.makedirs('/root/exp_goal_directed', exist_ok=True)

    # Primary classes: 20 classes that get 80% of samples in Phase 1 (goal-directed)
    primary_classes = list(range(20))  # classes 0-19

    configs = [
        ('uniform_ce_only', False, False),
        ('uniform_ar', True, False),
        ('goal_directed_ce_only', False, True),
        ('goal_directed_ar', True, True),
    ]

    results = {}
    for label, use_ar, use_conc in configs:
        full, sim = run_experiment(label, use_ar, use_conc, primary_classes, dev)
        results[label] = (full, sim)
        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("GOAL-DIRECTED v1 — RESULTS MATRIX")
    print("=" * 60)
    print(f"{'Config':<30} {'Full':>8} {'sim':>8}")
    print("-" * 48)
    for label, (full, sim) in results.items():
        print(f"{label:<30} {full:>7.1f}% {sim:>7.3f}")
    print()
    print("Prediction:")
    print("  uniform + CE only  → clone state, CL collapses")
    print("  uniform + AR       → anti-correlated, CL succeeds (64.5%)")
    print("  goal-directed + CE  → WORSE than uniform (lower entropy = stronger alignment)")
    print("  goal-directed + AR  → AR should still break Gaussian attractor")

if __name__ == '__main__':
    main()
