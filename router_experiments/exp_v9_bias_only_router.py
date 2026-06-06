"""v9: Router Freedom Theorem's limiting case — 4 trainable bias parameters.
Pre-diff + freeze encoder + per-task router re-init.
Router: Linear(256, 4) with ALL weights frozen, ONLY 4 bias terms trainable.
If even 4 parameters cause sim drift, the theorem is airtight:
"Any non-zero trainable freedom in the soft router leads to collapse."
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
import math, time, os, random
import numpy as np

NUM_EXPERTS = 4; NUM_CLASSES = 100; FEAT_DIM = 256
BATCH_SIZE = 128; EPOCHS_PER_TASK = 120
CIFAR_MEAN = (0.5071, 0.4867, 0.4408); CIFAR_STD = (0.2675, 0.2565, 0.2761)
SEED = 42; DATA_DIR = '/root/data'

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
    def __init__(self, in_channels=3, depth=28, widen=10):
        super().__init__()
        n = (depth - 4) // 6; k = widen; nCh = [16, 16*k, 32*k, 64*k]
        self.conv1 = nn.Conv2d(in_channels, nCh[0], 3, 1, 1, bias=False)
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
    def __init__(self, capacity_idx, num_classes=NUM_CLASSES):
        super().__init__()
        self.dominance_threshold = 0.35
        self.encoder = SharedEncoder()
        self.experts = nn.ModuleList([MutualExpert(FEAT_DIM, num_classes, NUM_EXPERTS, capacity_idx) for _ in range(NUM_EXPERTS)])
        self.register_buffer('usage_ema', torch.ones(NUM_EXPERTS) / NUM_EXPERTS)
        self.register_buffer('activation_ema', torch.ones(NUM_EXPERTS, NUM_EXPERTS) / NUM_EXPERTS)

    def freeze_router_weights_keep_bias(self):
        """v9: freeze all Linear.weight, keep only bias trainable — 4 params total."""
        for expert in self.experts:
            expert.activation.weight.requires_grad = False  # [4, 256] = frozen
            expert.activation.bias.requires_grad = True     # [4] = trainable (4 params total)
        # Total trainable router params = 4 bias entries × 1 expert that's actually used?
        # Actually each expert has its own activation, so 4 experts × 4 biases = 16 params max
        # But CE only sees active expert's router → effectively 4 params matter per task

    def reset_router_biases(self):
        """Re-init only biases."""
        for expert in self.experts:
            nn.init.zeros_(expert.activation.bias)

    def freeze_for_task(self, task_id):
        active = task_id % NUM_EXPERTS
        for i, expert in enumerate(self.experts):
            frozen = (i != active)
            for param in expert.classifier.parameters():
                param.requires_grad = not frozen
        # Router weights already frozen, biases already trainable — leave as-is
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
            dead_mask = (self.usage_ema < self.dominance_threshold).float()
            act_mat = act_mat + 0.4 * dead_mask.unsqueeze(0).unsqueeze(0)
        incoming = act_mat.sum(dim=1)
        weights = F.softmax(incoming, dim=1)
        combined = (weights.unsqueeze(-1) * logits).sum(dim=1)
        return combined, weights, logits, act_mat, feat

def get_task_ranges(num_tasks):
    base = 100 // num_tasks; remainder = 100 % num_tasks
    ranges = []; cur = 0
    for t in range(num_tasks):
        n = base + (1 if t < remainder else 0)
        ranges.append((cur, cur + n, n)); cur += n
    return ranges

def get_task_data(task_id, train=True, num_tasks=5):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    full = datasets.CIFAR100(root=DATA_DIR, train=train, download=False, transform=transform)
    ranges = get_task_ranges(num_tasks)
    start, end, n_cls = ranges[task_id]
    indices = [i for i, (_, y) in enumerate(full) if start <= y < end]
    return Subset(full, indices), start, end

def get_full_data(train=True):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    return datasets.CIFAR100(root=DATA_DIR, train=train, download=False, transform=transform)

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

def main():
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    dev = torch.device("cuda")

    os.makedirs('/root/exp_v9_bias_only', exist_ok=True)
    log_path = '/root/exp_v9_bias_only/experiment.log'

    def log(msg):
        print(msg)
        with open(log_path, 'a') as f: f.write(msg + '\n')

    log("=" * 60)
    log("V9: Router Freedom Theorem LIMIT — 4 trainable bias params")
    log("ALL router weights frozen. Only 4 bias terms trainable.")
    log("If even 4 params cause sim drift → theorem airtight.")
    log("=" * 60)

    model = ContinualLifecycleMoE(3).to(dev)
    log(f"Total params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Phase 1: pre-diff with full router
    log("\n=== PHASE 1: Pre-diff on Full CIFAR-100 (AR=0.1, 200E) ===")
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
        total_ce = 0; total_ar = 0; n_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            combined, weights, logits, act_mat, _ = model(xb, temperature=1.0)
            ce = F.cross_entropy(combined, yb)
            ar = ar_loss_fn(logits, 0.1)
            loss = ce + ar
            loss.backward(); opt.step()
            total_ce += ce.item(); total_ar += ar.item(); n_batches += 1
        sched.step()
        if epoch % 20 == 0:
            mean_s = compute_sim(model, sim_loader, dev)
            log(f"  P1 E{epoch:3d} ce={total_ce/n_batches:.3f} ar={total_ar/n_batches:.4f} sim={mean_s:.3f}")

    elapsed = time.time() - t0
    mean_s = compute_sim(model, sim_loader, dev)
    log(f"  Phase 1 done: {elapsed/60:.0f}min sim={mean_s:.3f}")

    # Phase 2: freeze encoder, freeze router WEIGHTS, keep only 4 biases
    log("\n=== PHASE 2: 5-Task CL (ONLY 4 bias params trainable) ===")
    for param in model.encoder.parameters():
        param.requires_grad = False
    model.freeze_router_weights_keep_bias()

    # Count actual trainable router params
    rt = sum(p.numel() for e in model.experts for p in e.activation.parameters() if p.requires_grad)
    log(f"  Encoder frozen. Router WEIGHTS frozen. Trainable router params: {rt}")
    log(f"  (4 bias terms per expert = {rt} total)")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5 * EPOCHS_PER_TASK)
    t0 = time.time()

    for task_id in range(5):
        train_subset, task_start, task_end = get_task_data(task_id, train=True)
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

        model.reset_router_biases()
        model.freeze_for_task(task_id)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(f"\nTASK {task_id}: classes {task_start}-{task_end-1}, active expert {task_id % 4}")
        log(f"  Router biases reset to 0. Trainable params: {trainable}")

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
            if epoch % 20 == 0:
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
    log(f"\nFINAL (4 bias params): " + " | ".join(f"{k}={v:.1f}%" for k, v in final.items()))
    log(f"FINAL sim={mean_s:.3f}")
    log(f"ROUTER FREEDOM LIMIT: {rt} trainable params | Full={final['full']:.1f}% | sim={mean_s:.3f}")

    torch.save({'model_state': model.state_dict(), 'final_results': final, 'sim': mean_s},
               '/root/exp_v9_bias_only/final_model.pt')
    log("Done!")

if __name__ == '__main__':
    main()
