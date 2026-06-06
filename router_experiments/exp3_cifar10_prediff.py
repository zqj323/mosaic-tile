"""Experiment 3: CIFAR-10 pre-differentiation + freeze CL.
Second benchmark — same WRN-28-10 architecture, different dataset.
CIFAR-10: 10 classes, 5 tasks x 2 classes each.
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
import math, time, os, copy, random
import numpy as np

NUM_EXPERTS = 4; NUM_CLASSES = 10; FEAT_DIM = 256
BATCH_SIZE = 128; EPOCHS_PER_TASK = 120
C10_MEAN = (0.4914, 0.4822, 0.4465); C10_STD = (0.2471, 0.2435, 0.2616)
SEED = 42
DATA_DIR = '/root/data'

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
    def __init__(self, capacity_idx, max_life=10, dominance_threshold=0.35, num_kill=1, cooldown=3,
                 use_snapshot=True, use_min_gen=True, freeze_encoder=False, trainable_router=False,
                 static_routing=False, num_classes=NUM_CLASSES, in_channels=3):
        super().__init__()
        self.capacity_idx = capacity_idx; self.max_life = max_life
        self.dominance_threshold = dominance_threshold; self.num_kill = num_kill
        self.cooldown = cooldown; self.use_snapshot = use_snapshot
        self.use_min_gen = use_min_gen; self.freeze_encoder = freeze_encoder
        self.trainable_router = trainable_router; self.static_routing = static_routing
        self.static_expert_idx = -1; self.num_classes = num_classes
        self.encoder = SharedEncoder(in_channels=in_channels)
        self.experts = nn.ModuleList([MutualExpert(FEAT_DIM, num_classes, NUM_EXPERTS, capacity_idx) for _ in range(NUM_EXPERTS)])
        self.register_buffer('usage_ema', torch.ones(NUM_EXPERTS) / NUM_EXPERTS)
        self.register_buffer('activation_ema', torch.ones(NUM_EXPERTS, NUM_EXPERTS) / NUM_EXPERTS)
        self.life_counter = torch.zeros(NUM_EXPERTS); self.generation = torch.zeros(NUM_EXPERTS)
        self.total_deaths = 0; self.cooldown_counter = 0; self.death_history = []; self.snapshots = []

    def save_snapshot(self, sim_score):
        state = {k: v.clone() for k, v in self.experts.state_dict().items()}
        self.snapshots.append((sim_score, state))
        self.snapshots.sort(key=lambda x: x[0]); self.snapshots = self.snapshots[:3]

    def _reinit_expert(self, idx, device):
        for name, param in self.experts[idx].named_parameters():
            if 'weight' in name and param.dim() >= 2: nn.init.kaiming_uniform_(param, a=math.sqrt(5))
            elif 'weight' in name: nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name: nn.init.zeros_(param)
        self.generation[idx] += 1; self.total_deaths += 1
        self.usage_ema[idx] = 1.0 / NUM_EXPERTS
        self.activation_ema[idx, :] = 1.0 / NUM_EXPERTS
        self.activation_ema[:, idx] = 1.0 / NUM_EXPERTS

    def check_lifecycle(self, epoch, sim_current, edges_current):
        N = NUM_EXPERTS; device = self.usage_ema.device
        if sim_current < 0.85 and edges_current >= 8 and epoch > 2: self.save_snapshot(sim_current)
        if self.cooldown_counter > 0: self.cooldown_counter -= 1
        collapsed = (sim_current > 0.83 and edges_current < 7)
        for i in range(N):
            if collapsed: self.life_counter[i] += 1
            else: self.life_counter[i] = max(0, self.life_counter[i] - 1.0)
        deaths = []
        if self.cooldown_counter == 0:
            candidates = sorted(
                [(i, self.life_counter[i].item()) for i in range(N) if self.life_counter[i] >= self.max_life],
                key=lambda x: x[1], reverse=True)
            if self.use_min_gen and candidates:
                candidates.sort(key=lambda x: (self.generation[x[0]].item(), -x[1]))
                kill_indices = [c[0] for c in candidates[:self.num_kill]]
            else: kill_indices = [c[0] for c in candidates[:self.num_kill]]
            for idx in kill_indices:
                self._reinit_expert(idx, device); deaths.append(idx)
            if deaths:
                self.life_counter[:] = 0; self.cooldown_counter = self.cooldown
                self.death_history.append((epoch, deaths))
        return deaths

    def freeze_for_task(self, task_id):
        active_idx = task_id % NUM_EXPERTS
        for i, expert in enumerate(self.experts):
            frozen = (i != active_idx)
            for param in expert.classifier.parameters(): param.requires_grad = not frozen
            router_frozen = frozen and (not self.trainable_router)
            for param in expert.activation.parameters(): param.requires_grad = not router_frozen
        if self.freeze_encoder and task_id > 0:
            for param in self.encoder.parameters(): param.requires_grad = False
        return active_idx

    def forward(self, x, temperature=1.0):
        B = x.size(0); feat = self.encoder(x)
        all_logits, all_act = [], []
        for expert in self.experts:
            logits, act = expert(feat)
            all_logits.append(logits); all_act.append(act)
        logits = torch.stack(all_logits, dim=1)
        act_mat = torch.stack(all_act, dim=1) / temperature
        if self.static_routing and self.static_expert_idx >= 0:
            weights = torch.zeros(B, NUM_EXPERTS, device=x.device)
            weights[:, self.static_expert_idx] = 1.0
            combined = (weights.unsqueeze(-1) * logits).sum(dim=1)
            return combined, weights, logits, act_mat, feat
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

# ── CIFAR-10 utilities ──
def get_c10_task_ranges(num_tasks=5):
    base = 10 // num_tasks; remainder = 10 % num_tasks
    ranges = []; cur = 0
    for t in range(num_tasks):
        n = base + (1 if t < remainder else 0)
        ranges.append((cur, cur + n, n)); cur += n
    return ranges

def get_c10_task_data(task_id, train=True):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(C10_MEAN, C10_STD)])
    full = datasets.CIFAR10(root=DATA_DIR, train=train, download=True, transform=transform)
    ranges = get_c10_task_ranges()
    start, end, n_cls = ranges[task_id]
    indices = [i for i, (_, y) in enumerate(full) if start <= y < end]
    return Subset(full, indices), start, end

@torch.no_grad()
def evaluate_c10(model, device, num_tasks=5):
    model.eval()
    results = {}
    for task_id in range(num_tasks):
        test_subset, start, end = get_c10_task_data(task_id, train=False)
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
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(C10_MEAN, C10_STD)])
    full_test = datasets.CIFAR10(root=DATA_DIR, train=False, download=False, transform=transform)
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
        combined, weights, logits, act_mat, _ = model(xb.to(device), temperature=0.3)
        all_logits.append(logits.cpu())
    logits = torch.cat(all_logits, 0)
    sims = []
    for i in range(NUM_EXPERTS):
        for j in range(i+1, NUM_EXPERTS):
            s = F.cosine_similarity(logits[:,i,:].mean(0).unsqueeze(0),
                                    logits[:,j,:].mean(0).unsqueeze(0)).item()
            sims.append(s)
    return np.mean(sims), sims

def ar_loss_fn(logits, lambda_ar=0.1):
    total_ar = 0.0; count = 0
    for i in range(NUM_EXPERTS):
        for j in range(i+1, NUM_EXPERTS):
            cos_sim = F.cosine_similarity(logits[:, i, :], logits[:, j, :], dim=1)
            total_ar += F.relu(cos_sim).mean(); count += 1
    return lambda_ar * total_ar / count if count > 0 else 0.0

# ── Main ──
def main():
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    dev = torch.device("cuda")

    os.makedirs('/root/exp3_c10', exist_ok=True)
    log_path = '/root/exp3_c10/experiment.log'

    def log(msg):
        print(msg)
        with open(log_path, 'a') as f: f.write(msg + '\n')

    log("=" * 60)
    log("EXPERIMENT 3: CIFAR-10 Pre-Diff + Freeze CL")
    log(f"5 tasks x 2 classes, {NUM_EXPERTS} experts, WRN-28-10")
    log(f"Phase 1: Pre-diff on full CIFAR-10, AR=0.1, 150 epochs")
    log(f"Phase 2: Freeze encoder + old experts, {EPOCHS_PER_TASK} epochs/task")
    log("=" * 60)

    model = ContinualLifecycleMoE(3, max_life=10, dominance_threshold=0.35,
                                   num_kill=1, cooldown=3,
                                   use_snapshot=True, use_min_gen=True,
                                   num_classes=10).to(dev)
    log(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # ── Phase 1: Pre-differentiation on full CIFAR-10 ──
    log("\n=== PHASE 1: Pre-differentiation on Full CIFAR-10 ===")
    log("lambda_AR=0.1, epochs=150")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(C10_MEAN, C10_STD)])
    full_train = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform)
    train_loader = DataLoader(full_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    full_test = datasets.CIFAR10(root=DATA_DIR, train=False, download=False, transform=transform)
    sim_loader = DataLoader(Subset(full_test, np.random.choice(10000, 2000, replace=False)),
                           batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=150)
    t0 = time.time()

    for epoch in range(150):
        model.train()
        total_ce = 0; total_ar = 0; n_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            combined, weights, logits, act_mat, _ = model(xb, temperature=1.0)
            ce = F.cross_entropy(combined, yb)
            ar = ar_loss_fn(logits, lambda_ar=0.1)
            loss = ce + ar
            loss.backward(); opt.step()
            total_ce += ce.item(); total_ar += ar.item(); n_batches += 1
        sched.step()

        if epoch % 10 == 0:
            mean_s, sims = compute_sim(model, sim_loader, dev)
            log(f"  P1 E{epoch:3d} ce={total_ce/n_batches:.3f} ar={total_ar/n_batches:.4f} sim={mean_s:.3f}")

    elapsed = time.time() - t0
    mean_s, sims = compute_sim(model, sim_loader, dev)
    results = evaluate_c10(model, dev)
    log(f"  Phase 1 done: {elapsed/60:.0f}min, full={results['full']:.1f}% sim={mean_s:.3f}")
    log(f"  Pairwise sims: {[f'{s:.3f}' for s in sims]}")

    prediff_ckpt = {k: v.clone() for k, v in model.state_dict().items()}
    torch.save({'model_state': prediff_ckpt, 'sim': mean_s, 'results': results},
               '/root/exp3_c10/prediff_ckpt.pt')
    log("  Pre-diff checkpoint saved.")

    # ── Phase 2: 5-task CL with freeze ──
    log("\n=== PHASE 2: 5-Task CL (Freeze Encoder + Old Experts) ===")
    model.freeze_encoder = True
    # Freeze encoder from Task 0 (10 classes total, 2 per task — avoid overfitting features to first task)
    for param in model.encoder.parameters():
        param.requires_grad = False
    log("  Encoder frozen from Task 0 (2 classes/task prevents balanced feature adaptation)")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5 * EPOCHS_PER_TASK)
    t0 = time.time()

    for task_id in range(5):
        train_subset, task_start, task_end = get_c10_task_data(task_id, train=True)
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

        model.freeze_for_task(task_id)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(f"\nTASK {task_id}: CIFAR-10 classes {task_start}-{task_end-1}, active expert {task_id % 4}")
        log(f"  Trainable: {trainable/1e6:.1f}M params")

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
            sched.step()

            if epoch % 10 == 0:
                mean_s, _ = compute_sim(model, sim_loader, dev)
                log(f"  T{task_id} E{epoch:3d} ce={total_loss/n_batches:.3f} sim={mean_s:.3f}")

        results = evaluate_c10(model, dev)
        mean_s, sims = compute_sim(model, sim_loader, dev)
        log(f"  >>> TASK {task_id} DONE: " + " | ".join(f"{k}={v:.1f}%" for k, v in results.items()))
        log(f"  sim={mean_s:.3f} ({[f'{s:.3f}' for s in sims]})")

    elapsed = time.time() - t0
    log(f"\nPhase 2 done: {elapsed/60:.0f}min")

    final = evaluate_c10(model, dev)
    mean_s, sims = compute_sim(model, sim_loader, dev)
    log(f"\nFINAL: " + " | ".join(f"{k}={v:.1f}%" for k, v in final.items()))
    log(f"FINAL sim={mean_s:.3f} ({[f'{s:.3f}' for s in sims]})")
    log(f"Deaths: {model.total_deaths}, Gen: {model.generation.tolist()}")

    torch.save({'model_state': model.state_dict(), 'final_results': final, 'sim': mean_s},
               '/root/exp3_c10/final_model.pt')
    log("Saved: /root/exp3_c10/final_model.pt")

if __name__ == '__main__':
    main()
