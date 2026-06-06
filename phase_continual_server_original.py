"""Lifecycle MoE for Continual Learning on CIFAR-100.
5 sequential tasks (20 classes each). Lifecycle mechanism naturally handles
catastrophic forgetting: old experts protect old knowledge, death replaces stale experts,
newborns specialize in new tasks.
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
import math, time, os, json, copy, random, argparse
import numpy as np

NUM_EXPERTS = 4; NUM_CLASSES = 100; FEAT_DIM = 256
BATCH_SIZE = 128; EPOCHS_PER_TASK = 120; NUM_TASKS = 5  # 5 tasks x 20 classes = 100
T_END = 0.3; T_WARMUP = 15
SEED = 42

CIFAR_MEAN = (0.5071, 0.4867, 0.4408); CIFAR_STD = (0.2675, 0.2565, 0.2761)

# ── Model (shared with existing experiments) ──
def make_classifier(capacity_idx):
    if capacity_idx == 0: return nn.Linear(FEAT_DIM, NUM_CLASSES)
    elif capacity_idx == 1: return nn.Sequential(nn.Linear(FEAT_DIM, 128), nn.ReLU(), nn.Linear(128, NUM_CLASSES))
    elif capacity_idx == 2: return nn.Sequential(nn.Linear(FEAT_DIM, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, NUM_CLASSES))
    elif capacity_idx == 3: return nn.Sequential(nn.Linear(FEAT_DIM, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, NUM_CLASSES))

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

class MutualExpert(nn.Module):
    def __init__(self, feat_dim, num_classes, num_experts, capacity_idx):
        super().__init__()
        self.classifier = make_classifier(capacity_idx)
        self.activation = nn.Linear(feat_dim, num_experts)
    def forward(self, x):
        return self.classifier(x), torch.sigmoid(self.activation(x))

class ContinualLifecycleMoE(nn.Module):
    def __init__(self, capacity_idx, max_life=10, dominance_threshold=0.35, num_kill=1, cooldown=3,
                 use_snapshot=True, use_min_gen=True, freeze_encoder=False, trainable_router=False,
                 static_routing=False):
        super().__init__()
        self.capacity_idx = capacity_idx
        self.max_life = max_life
        self.dominance_threshold = dominance_threshold
        self.num_kill = num_kill
        self.cooldown = cooldown
        self.use_snapshot = use_snapshot
        self.use_min_gen = use_min_gen
        self.freeze_encoder = freeze_encoder
        self.trainable_router = trainable_router
        self.static_routing = static_routing
        self.static_expert_idx = -1  # -1 = use learned router
        self.encoder = SharedEncoder()
        self.experts = nn.ModuleList([MutualExpert(FEAT_DIM, NUM_CLASSES, NUM_EXPERTS, capacity_idx) for _ in range(NUM_EXPERTS)])
        self.register_buffer('usage_ema', torch.ones(NUM_EXPERTS) / NUM_EXPERTS)
        self.register_buffer('activation_ema', torch.ones(NUM_EXPERTS, NUM_EXPERTS) / NUM_EXPERTS)
        self.life_counter = torch.zeros(NUM_EXPERTS)
        self.generation = torch.zeros(NUM_EXPERTS)
        self.total_deaths = 0
        self.cooldown_counter = 0
        self.death_history = []
        self.snapshots = []

    def save_snapshot(self, sim_score):
        state = {k: v.clone() for k, v in self.experts.state_dict().items()}
        self.snapshots.append((sim_score, state))
        self.snapshots.sort(key=lambda x: x[0])
        self.snapshots = self.snapshots[:3]

    def _reinit_expert(self, idx, device):
        for name, param in self.experts[idx].named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.kaiming_uniform_(param, a=math.sqrt(5))
            elif 'weight' in name:
                nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.zeros_(param)
        self.generation[idx] += 1
        self.total_deaths += 1
        self.usage_ema[idx] = 1.0 / NUM_EXPERTS
        self.activation_ema[idx, :] = 1.0 / NUM_EXPERTS
        self.activation_ema[:, idx] = 1.0 / NUM_EXPERTS

    def check_lifecycle(self, epoch, sim_current, edges_current):
        N = NUM_EXPERTS; device = self.usage_ema.device
        if sim_current < 0.85 and edges_current >= 8 and epoch > 2:
            self.save_snapshot(sim_current)
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
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
            else:
                kill_indices = [c[0] for c in candidates[:self.num_kill]]

            for idx in kill_indices:
                self._reinit_expert(idx, device)
                deaths.append(idx)

            if deaths:
                self.life_counter[:] = 0
                self.cooldown_counter = self.cooldown
                self.death_history.append((epoch, deaths))
        return deaths

    def freeze_for_task(self, task_id):
        """Freeze all experts except the one designated for this task.
        If freeze_encoder=True, also freeze encoder after Task 0 to
        prevent Gaussian attractor from bypassing frozen experts via encoder drift.
        If trainable_router=True, keep activation (router) layers trainable for
        ALL experts to preserve routing plasticity while experts are frozen."""
        active_idx = task_id % NUM_EXPERTS
        for i, expert in enumerate(self.experts):
            frozen = (i != active_idx)
            for param in expert.classifier.parameters():
                param.requires_grad = not frozen
            # Router: either frozen (default) or trainable for all
            router_frozen = frozen and (not self.trainable_router)
            for param in expert.activation.parameters():
                param.requires_grad = not router_frozen
        if self.freeze_encoder and task_id > 0:
            for param in self.encoder.parameters():
                param.requires_grad = False
        return active_idx

    def set_static_expert(self, idx):
        self.static_expert_idx = idx

    def forward(self, x, temperature=1.0):
        B = x.size(0); feat = self.encoder(x)
        all_logits, all_act = [], []
        for expert in self.experts:
            logits, act = expert(feat)
            all_logits.append(logits); all_act.append(act)
        logits = torch.stack(all_logits, dim=1)
        act_mat = torch.stack(all_act, dim=1) / temperature

        if self.static_routing and self.static_expert_idx >= 0:
            # bypass learned router: one-hot routing to static expert
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

# ── Task setup ──
def get_task_data(task_id, train=True):
    """Get CIFAR-100 subset for a specific task (20 classes)."""
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ]) if train else transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    full = datasets.CIFAR100(root='/root/data', train=train, download=False, transform=transform)
    start_cls = task_id * 20
    end_cls = start_cls + 20
    indices = [i for i, (_, y) in enumerate(full) if start_cls <= y < end_cls]
    return Subset(full, indices), start_cls

@torch.no_grad()
def evaluate_all_tasks(model, device):
    """Evaluate accuracy on all 5 tasks + combined."""
    model.eval()
    results = {}
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    full = datasets.CIFAR100(root='/root/data', train=False, download=False, transform=transform)

    for task_id in range(NUM_TASKS):
        start_cls = task_id * 20
        end_cls = start_cls + 20
        loader = DataLoader(
            Subset(full, [i for i, (_, y) in enumerate(full) if start_cls <= y < end_cls]),
            batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

        if model.static_routing:
            model.set_static_expert(task_id % NUM_EXPERTS)

        correct = total = 0
        for xb, yb in loader:
            xb = xb.to(device)
            combined, _, _, _, _ = model(xb, temperature=0.3)
            # Only evaluate on this task's classes
            task_logits = combined[:, start_cls:end_cls]
            task_labels = yb.to(device) - start_cls
            correct += (task_logits.argmax(1) == task_labels).sum().item()
            total += yb.size(0)
        results[f'task{task_id}'] = 100 * correct / total

    # Full 100-class accuracy — use learned routing if available
    if model.static_routing:
        model.set_static_expert(-1)  # revert to learned router for full eval
    loader = DataLoader(full, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
    correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        combined, _, _, _, _ = model(xb, temperature=0.3)
        correct += (combined.argmax(1) == yb).sum().item()
        total += yb.size(0)
    results['full'] = 100 * correct / total

    model.train()
    return results

@torch.no_grad()
def analyze_current(model, loader, device):
    model.eval()
    all_weights = []; all_logits_for_sim = []
    for xb, _ in loader:
        xb = xb.to(device)
        combined, weights, logits, act_mat, _ = model(xb, temperature=0.3)
        all_weights.append(weights.cpu())
        all_logits_for_sim.append(logits.cpu())

    weights = torch.cat(all_weights, dim=0)
    ent = -(weights * torch.log(weights + 1e-8)).sum(dim=1).mean().item()
    usage = weights.mean(dim=0)
    gini = 1 - (usage**2).sum().item()

    logit = torch.cat(all_logits_for_sim, dim=0)
    sims = []
    for i in range(NUM_EXPERTS):
        for j in range(i+1, NUM_EXPERTS):
            s = F.cosine_similarity(logit[:,i,:].mean(0).unsqueeze(0), logit[:,j,:].mean(0).unsqueeze(0)).item()
            sims.append(s)
    sim_mean = sum(sims) / len(sims) if sims else 0.0
    edges = (model.activation_ema > 0.15).sum().item()

    model.train()
    return {'sim_mean': sim_mean, 'edges': edges, 'routing_entropy': ent, 'routing_gini': gini}

# ── Main ──
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=SEED)
parser.add_argument('--resume', type=str, default='')
parser.add_argument('--freeze_encoder', action='store_true', default=False)
parser.add_argument('--trainable_router', action='store_true', default=False)
parser.add_argument('--static_routing', action='store_true', default=False)
parser.add_argument('--pretrain_epochs', type=int, default=0,
                    help='Pre-differentiate on full CIFAR-100 for N epochs before CL')
parser.add_argument('--pretrain_lambda_ar', type=float, default=0.0,
                    help='AR loss weight during pre-training')
parser.add_argument('--pretrain_kill', type=int, default=1,
                    help='num_kill during pre-training')
args = parser.parse_args()
SEED = args.seed

dev = torch.device("cuda")
exp_name = f"continual_lifecycle_s{SEED}"
if args.pretrain_epochs > 0:
    exp_name = f"continual_prediff_p{args.pretrain_epochs}_k{args.pretrain_kill}_ar{args.pretrain_lambda_ar}_s{SEED}"
ckpt_path = f'/root/phase_{exp_name}_ckpt.pt'

random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

model = ContinualLifecycleMoE(3, max_life=10, dominance_threshold=0.35,
                               num_kill=args.pretrain_kill if args.pretrain_epochs > 0 else 1,
                               cooldown=3,
                               use_snapshot=True, use_min_gen=True,
                               freeze_encoder=args.freeze_encoder,
                               trainable_router=args.trainable_router,
                               static_routing=args.static_routing).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_TASKS * EPOCHS_PER_TASK)

start_task = 0; start_epoch_in_task = 0; history = []; per_task_history = []; t0 = time.time()

if args.resume:
    ckpt = torch.load(args.resume, map_location=dev)
    model.load_state_dict(ckpt['model_state'])
    opt.load_state_dict(ckpt['opt_state'])
    sched.load_state_dict(ckpt['sched_state'])
    start_task = ckpt['task_id']
    start_epoch_in_task = ckpt['epoch_in_task'] + 1
    history = ckpt.get('history', [])
    per_task_history = ckpt.get('per_task_history', [])
    model.life_counter = torch.tensor(ckpt['life_counter']).to(dev)
    model.generation = torch.tensor(ckpt['generation']).to(dev)
    model.total_deaths = ckpt['total_deaths']
    model.cooldown_counter = ckpt['cooldown_counter']
    model.death_history = ckpt.get('death_history', [])
    model.usage_ema = torch.tensor(ckpt['usage_ema']).to(dev)
    model.activation_ema = torch.tensor(ckpt['activation_ema']).to(dev)
    t0 = time.time() - ckpt.get('elapsed', 0)
    print(f"Resumed from task {start_task} epoch {start_epoch_in_task}")

p = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Params: {p:.1f}M")

# ═══════════════════════════════════════════════════════════════
# Phase 1: Pre-differentiation on full CIFAR-100
# ═══════════════════════════════════════════════════════════════
pretrain_done = (start_task > 0) or (start_epoch_in_task > 0)  # skip if resuming into CL

if args.pretrain_epochs > 0 and not pretrain_done:
    print(f"\n{'═'*60}")
    print(f"PHASE 1: Pre-differentiation on full CIFAR-100")
    print(f"  epochs={args.pretrain_epochs}  kill={args.pretrain_kill}  lambda_ar={args.pretrain_lambda_ar}")
    print(f"  Goal: break clone state (sim<0.85) before continual learning")
    print(f"{'═'*60}")

    # Full CIFAR-100 (all 100 classes)
    pretrain_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    pretrain_test_transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    pretrain_full = datasets.CIFAR100(root='/root/data', train=True, download=False,
                                       transform=pretrain_transform)
    pretrain_full_test = datasets.CIFAR100(root='/root/data', train=False, download=False,
                                            transform=pretrain_test_transform)
    pretrain_loader = DataLoader(pretrain_full, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=2, pin_memory=True)

    pretrain_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.pretrain_epochs)
    pretrain_history = []

    for epoch in range(args.pretrain_epochs):
        progress = min(1.0, epoch / T_WARMUP)
        temp = T_END + (3.0 - T_END) * (1 + math.cos(math.pi * progress)) / 2

        model.train(); total_ce = 0.0; total_ar = 0.0; n_batch = 0
        for xb, yb in pretrain_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            combined, weights, logits, act_mat, feat = model(xb, temperature=temp)
            ce_loss = F.cross_entropy(combined, yb)
            weight_entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=1).mean()
            act_target = act_mat.mean(dim=1).mean(dim=0)
            act_balance = ((act_target - 1.0/NUM_EXPERTS)**2).sum()

            # AR loss: penalize positive cosine similarity only (ignore anti-correlated experts)
            ar_loss = torch.tensor(0.0, device=dev)
            if args.pretrain_lambda_ar > 0:
                for i in range(NUM_EXPERTS):
                    for j in range(i+1, NUM_EXPERTS):
                        cs = F.cosine_similarity(logits[:, i, :], logits[:, j, :], dim=1)
                        ar_loss += F.relu(cs).mean()  # relu: only penalize cos>0 (clone behavior)
                ar_loss = ar_loss / (NUM_EXPERTS * (NUM_EXPERTS-1) / 2)

            loss = ce_loss + weight_entropy * 0.01 + act_balance * 0.05 + args.pretrain_lambda_ar * ar_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_ce += ce_loss.item(); total_ar += ar_loss.item(); n_batch += 1
        pretrain_sched.step()

        # Sim analysis on test set (no augmentation)
        test_loader_small = DataLoader(Subset(pretrain_full_test, range(500)),
                                        batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
        a = analyze_current(model, test_loader_small, dev)
        deaths = model.check_lifecycle(epoch, a['sim_mean'], a['edges'])

        # NaN detection
        if math.isnan(total_ce) or math.isnan(a['sim_mean']):
            print(f"  PRE E{epoch:3d} *** NaN DETECTED — saving checkpoint and aborting ***")
            torch.save({'model_state': model.state_dict(), 'epoch': epoch},
                       f'/root/phase_{exp_name}_nan_ckpt.pt')
            break

        elapsed = time.time() - t0
        life = [str(int(model.life_counter[i].item())) for i in range(4)]
        gen = [str(int(model.generation[i].item())) for i in range(4)]
        death_str = f" DEATH: {deaths} gen={gen} total={model.total_deaths}" if deaths else ""

        if epoch % 10 == 0 or deaths:
            print(f"  PRE E{epoch:3d} ce={total_ce/n_batch:.3f} ar={total_ar/n_batch:.4f} "
                  f"sim={a['sim_mean']:.3f} edges={a['edges']}/16 "
                  f"life=[{','.join(life)}] gen=[{','.join(gen)}]{death_str}")

        pretrain_history.append({'phase': 'pretrain', 'epoch': epoch,
                                  **a, 'ce': round(total_ce/n_batch, 3),
                                  'life_counters': model.life_counter.tolist(),
                                  'generations': model.generation.tolist(),
                                  'total_deaths': model.total_deaths})

    # Save pre-diff checkpoint
    pre_diff_path = f'/root/phase_{exp_name}_prediff.pt'
    torch.save({
        'model_state': model.state_dict(),
        'opt_state': opt.state_dict(),
        'sched_state': sched.state_dict(),
        'pretrain_history': pretrain_history,
        'life_counter': model.life_counter.tolist(),
        'generation': model.generation.tolist(),
        'total_deaths': model.total_deaths,
        'cooldown_counter': model.cooldown_counter,
        'death_history': model.death_history,
        'usage_ema': model.usage_ema.tolist(),
        'activation_ema': model.activation_ema.tolist(),
        'elapsed': time.time() - t0,
        'pretrain_epochs': args.pretrain_epochs,
        'pretrain_kill': args.pretrain_kill,
        'pretrain_lambda_ar': args.pretrain_lambda_ar,
    }, pre_diff_path)
    print(f"\n  Pre-diff checkpoint saved: {pre_diff_path}")
    print(f"  Final sim={pretrain_history[-1]['sim_mean']:.3f}  "
          f"gen={model.generation.tolist()}  deaths={model.total_deaths}")
    print(f"{'═'*60}\n")

    # Reset scheduler for CL phase
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_TASKS * EPOCHS_PER_TASK)
    # Re-apply state from checkpoint
    ckpt = torch.load(pre_diff_path, map_location=dev)
    opt.load_state_dict(ckpt['opt_state'])

test_subset, _ = get_task_data(0, train=False)  # for sim analysis

for task_id in range(start_task, NUM_TASKS):
    train_data, task_start_cls = get_task_data(task_id, train=True)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    print(f"\n{'─'*50}")
    print(f"TASK {task_id}: classes {task_start_cls}-{task_start_cls+19}")
    print(f"{'─'*50}")

    # Task boundary: freeze old experts, activate one for new task
    if task_id > 0:
        active = model.freeze_for_task(task_id)
        if args.static_routing:
            model.set_static_expert(active)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  FREEZE: expert {active} active, {trainable_params/1e6:.1f}M trainable params")

    start_ep = start_epoch_in_task if task_id == start_task else 0
    for epoch_in_task in range(start_ep, EPOCHS_PER_TASK):
        global_epoch = task_id * EPOCHS_PER_TASK + epoch_in_task
        # After pre-training, use fully annealed temperature (skip warmup)
        if args.pretrain_epochs > 0:
            temp = T_END
        else:
            progress = min(1.0, global_epoch / T_WARMUP)
            temp = T_END + (3.0 - T_END) * (1 + math.cos(math.pi * progress)) / 2

        model.train(); total_ce = 0.0; n_batch = 0
        for xb, yb in train_loader:
            xb = xb.to(dev)
            # Remap labels to global class space
            yb_global = yb.to(dev)  # labels are already in [task_start_cls, task_start_cls+19]

            opt.zero_grad()
            combined, weights, logits, act_mat, feat = model(xb, temperature=temp)
            ce_loss = F.cross_entropy(combined, yb_global)
            weight_entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=1).mean()
            act_target = act_mat.mean(dim=1).mean(dim=0)
            act_balance = ((act_target - 1.0/NUM_EXPERTS)**2).sum()
            loss = ce_loss + weight_entropy * 0.01 + act_balance * 0.05
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_ce += ce_loss.item(); n_batch += 1
        sched.step()

        # Sim analysis on a small test subset
        test_loader_small = DataLoader(Subset(test_subset, range(500)), batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
        a = analyze_current(model, test_loader_small, dev)
        deaths = model.check_lifecycle(global_epoch, a['sim_mean'], a['edges'])

        elapsed = time.time() - t0
        life = [str(int(model.life_counter[i].item())) for i in range(4)]
        gen = [str(int(model.generation[i].item())) for i in range(4)]
        death_str = ""
        if deaths:
            death_str = f" DEATH: {deaths} gen={gen} total={model.total_deaths}"

        if epoch_in_task % 10 == 0 or deaths:
            print(f"  T{task_id} E{epoch_in_task:3d} (global {global_epoch:3d}) "
                  f"ce={total_ce/n_batch:.3f} sim={a['sim_mean']:.3f} edges={a['edges']}/16 "
                  f"life=[{','.join(life)}] gen=[{','.join(gen)}]{death_str}")

        history.append({'task': task_id, 'epoch_in_task': epoch_in_task, 'global_epoch': global_epoch,
                        **a, 'ce': round(total_ce/n_batch, 3),
                        'life_counters': model.life_counter.tolist(),
                        'generations': model.generation.tolist(),
                        'total_deaths': model.total_deaths,
                        'deaths_this_epoch': deaths})

    # End of task: evaluate all tasks
    accs = evaluate_all_tasks(model, dev)
    per_task_history.append({'task': task_id, 'accs': accs, 'gen': model.generation.tolist(),
                             'deaths': model.total_deaths})
    print(f"  >>> TASK {task_id} DONE: " + " | ".join(f"{k}={v:.1f}%" for k, v in accs.items()))

    # Save checkpoint
    torch.save({
        'task_id': task_id, 'epoch_in_task': EPOCHS_PER_TASK - 1,
        'model_state': model.state_dict(), 'opt_state': opt.state_dict(),
        'sched_state': sched.state_dict(), 'history': history,
        'per_task_history': per_task_history,
        'life_counter': model.life_counter.tolist(),
        'generation': model.generation.tolist(), 'total_deaths': model.total_deaths,
        'cooldown_counter': model.cooldown_counter, 'death_history': model.death_history,
        'usage_ema': model.usage_ema.tolist(), 'activation_ema': model.activation_ema.tolist(),
        'elapsed': time.time() - t0,
    }, ckpt_path)

elapsed_h = (time.time() - t0) / 3600
print(f"\n{'='*60}")
print(f"FINAL per-task accuracy after {NUM_TASKS} tasks:")
for entry in per_task_history:
    print(f"  After T{entry['task']}: " + " | ".join(f"{k}={v:.1f}%" for k, v in entry['accs'].items()))
print(f"Total: {elapsed_h:.1f}h | Deaths: {model.total_deaths} | Gen: {model.generation.tolist()}")

with open(f'/root/phase_{exp_name}.json', 'w') as f:
    json.dump({'config': 'continual_lifecycle', 'seed': SEED, 'per_task_history': per_task_history,
               'history': history}, f)
