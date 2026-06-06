"""v8: Per-task router re-init + Reduced router capacity.
Tests Expert Weakness Principle on routers:
"Fewer router params = fewer CE alignment degrees of freedom = more stable."
v5 (4112 params): CE re-aligns in 120 epochs.
v8: test shared_router (1028→260 params) — can CE still re-align?
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

# ── SharedEncoder, WideResNetBlock, make_classifier (same) ──
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

# ── v8 Expert: classifier + optional shared router ──
class MutualExpertV8(nn.Module):
    def __init__(self, feat_dim, num_classes, num_experts, capacity_idx, shared_router=False, router_dim=4):
        super().__init__()
        self.classifier = make_classifier(capacity_idx, num_classes)
        self.shared_router = shared_router
        self.num_experts = num_experts
        if not shared_router:
            self.activation = nn.Linear(feat_dim, router_dim)
        # else: activation is shared and set externally

    def forward(self, x):
        return self.classifier(x), None  # act handled by shared_router in model

class ContinualLifecycleMoEV8(nn.Module):
    def __init__(self, capacity_idx, use_shared_router=False, router_dim=4, num_classes=NUM_CLASSES):
        super().__init__()
        self.use_shared_router = use_shared_router
        self.router_dim = router_dim
        self.dominance_threshold = 0.35
        self.encoder = SharedEncoder()
        self.experts = nn.ModuleList([MutualExpertV8(FEAT_DIM, num_classes, NUM_EXPERTS, capacity_idx,
                                                       shared_router=use_shared_router, router_dim=router_dim)
                                       for _ in range(NUM_EXPERTS)])
        if use_shared_router:
            # Single shared activation producing [B, NUM_EXPERTS * router_dim] → reshape to [B, N, router_dim]
            self.shared_activation = nn.Linear(FEAT_DIM, NUM_EXPERTS * router_dim)
        else:
            self.shared_activation = None
            # Per-expert activations
            for e in self.experts:
                e.activation = nn.Linear(FEAT_DIM, router_dim)

        self.register_buffer('usage_ema', torch.ones(NUM_EXPERTS) / NUM_EXPERTS)
        self.register_buffer('activation_ema', torch.ones(NUM_EXPERTS, NUM_EXPERTS) / NUM_EXPERTS)

    def router_params(self):
        if self.use_shared_router:
            return sum(p.numel() for p in self.shared_activation.parameters())
        else:
            return sum(p.numel() for e in self.experts for p in e.activation.parameters())

    def reset_router(self):
        if self.use_shared_router:
            for name, param in self.shared_activation.named_parameters():
                if 'weight' in name: nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                elif 'bias' in name: nn.init.zeros_(param)
        else:
            for e in self.experts:
                for name, param in e.activation.named_parameters():
                    if 'weight' in name: nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                    elif 'bias' in name: nn.init.zeros_(param)
        self.usage_ema.fill_(1.0 / NUM_EXPERTS)
        self.activation_ema.fill_(1.0 / NUM_EXPERTS)

    def freeze_for_task(self, task_id):
        active = task_id % NUM_EXPERTS
        for i, expert in enumerate(self.experts):
            frozen = (i != active)
            for param in expert.classifier.parameters():
                param.requires_grad = not frozen
            if not self.use_shared_router and hasattr(expert, 'activation'):
                for param in expert.activation.parameters():
                    param.requires_grad = True
        if self.use_shared_router:
            for param in self.shared_activation.parameters():
                param.requires_grad = True
        return active

    def forward(self, x, temperature=1.0):
        B = x.size(0); feat = self.encoder(x)
        all_logits = []
        for expert in self.experts:
            logits, _ = expert(feat)
            all_logits.append(logits)
        logits = torch.stack(all_logits, dim=1)  # B × N × C

        if self.use_shared_router:
            shared_act = self.shared_activation(feat)  # B × (N * router_dim)
            # Reshape to B × N × router_dim, then pool to B × N
            act = shared_act.view(B, NUM_EXPERTS, self.router_dim).mean(dim=2)  # B × N
            act_mat = act.unsqueeze(-1).expand(-1, -1, NUM_EXPERTS) * 0.1  # weak routing
        else:
            all_act = []
            for expert in self.experts:
                act = torch.sigmoid(expert.activation(feat))  # B × router_dim
                all_act.append(act)
            act_tensor = torch.stack(all_act, dim=1)  # B × N × router_dim
            # Pool across router_dim to get act_mat
            act_mat = act_tensor.mean(dim=2).unsqueeze(-1).expand(-1, -1, NUM_EXPERTS) * 0.1

        act_mat = act_mat / temperature

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

# ── Utilities ──
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
    args_router = int(os.environ.get('ROUTER_DIM', '4'))
    args_shared = os.environ.get('SHARED', '0') == '1'

    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    dev = torch.device("cuda")

    config = f"shared{args_shared}_dim{args_router}"
    os.makedirs('/root/exp_v8_weak_router', exist_ok=True)
    log_path = f'/root/exp_v8_weak_router/{config}.log'
    PREDIFF_ENCODER = '/root/exp_v8_weak_router/prediff_encoder.pt'
    PREDIFF_CLASSIFIERS = '/root/exp_v8_weak_router/prediff_classifiers.pt'

    def log(msg):
        print(msg)
        with open(log_path, 'a') as f: f.write(msg + '\n')

    log("=" * 60)
    log(f"V8: Pre-diff + Per-Task Router Re-Init + WEAK ROUTER")
    log(f"Expert Weakness Principle: fewer params = less CE alignment capacity")
    log(f"Config: shared_router={args_shared}, router_dim={args_router}")
    log("=" * 60)

    model = ContinualLifecycleMoEV8(3, use_shared_router=args_shared, router_dim=args_router).to(dev)
    router_p = model.router_params()
    total_p = sum(p.numel() for p in model.parameters())
    log(f"Total params: {total_p/1e6:.1f}M | Router params: {router_p}")
    log(f"Router capacity: {router_p/4112*100:.0f}% of original (4112)")

    # ── Phase 1: Pre-diff (with checkpoint save/reuse) ──
    test_set = get_full_data(train=False)
    sim_loader = DataLoader(Subset(test_set, np.random.choice(10000, 2000, replace=False)),
                           batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    if os.path.exists(PREDIFF_ENCODER) and os.path.exists(PREDIFF_CLASSIFIERS):
        log("\n=== PHASE 1: LOADING existing pre-diff checkpoint ===")
        model.encoder.load_state_dict(torch.load(PREDIFF_ENCODER, map_location=dev))
        classifier_sd = torch.load(PREDIFF_CLASSIFIERS, map_location=dev)
        for i, expert in enumerate(model.experts):
            expert.classifier.load_state_dict(classifier_sd[f'expert_{i}'])
        mean_s = compute_sim(model, sim_loader, dev)
        log(f"  Loaded encoder + 4 classifiers. sim={mean_s:.3f} (skipping 200E)")
    else:
        log("\n=== PHASE 1: Pre-diff on Full CIFAR-100 (AR=0.1, 200E) ===")
        full_train = get_full_data(train=True)
        train_loader = DataLoader(full_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

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

        # Save encoder + classifiers for future reuse (router NOT saved — gets re-init anyway)
        torch.save(model.encoder.state_dict(), PREDIFF_ENCODER)
        ckpt = {f'expert_{i}': model.experts[i].classifier.state_dict() for i in range(4)}
        torch.save(ckpt, PREDIFF_CLASSIFIERS)
        log(f"  Saved: {PREDIFF_ENCODER} + {PREDIFF_CLASSIFIERS}")

    # router carries Phase 1 class distribution bias — always re-init for Phase 2
    model.reset_router()
    mean_s = compute_sim(model, sim_loader, dev)
    log(f"  Router re-initialized (fresh). Sim (pre-Phase2) = {mean_s:.3f}")

    # Phase 2: CL with per-task re-init + weak router
    log(f"\n=== PHASE 2: 5-Task CL (Re-Init Weak Router, {router_p} params) ===")
    for param in model.encoder.parameters():
        param.requires_grad = False
    log(f"  Encoder frozen. Router params={router_p}")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5 * EPOCHS_PER_TASK)
    t0 = time.time()

    for task_id in range(5):
        train_subset, task_start, task_end = get_task_data(task_id, train=True)
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

        model.reset_router()
        model.freeze_for_task(task_id)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(f"\nTASK {task_id}: classes {task_start}-{task_end-1}, active expert {task_id % 4}")
        log(f"  Router re-initialized. Trainable: {trainable/1e6:.1f}M")

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
    log(f"\nFINAL ({config}): " + " | ".join(f"{k}={v:.1f}%" for k, v in final.items()))
    log(f"FINAL sim={mean_s:.3f} | Router params={router_p} ({router_p/4112*100:.0f}% of original)")
    log(f"SUMMARY: {config} | Full={final['full']:.1f}% | sim={mean_s:.3f} | router_params={router_p}")

    torch.save({'model_state': model.state_dict(), 'final_results': final, 'sim': mean_s,
                'router_params': router_p, 'config': config},
               f'/root/exp_v8_weak_router/final_{config}.pt')
    log("Done!")

if __name__ == '__main__':
    main()
