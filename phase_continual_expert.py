"""Self-contained resume for overlap/granularity Phase 2. No imports from original script."""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
import math, time, random, argparse
import numpy as np

NUM_EXPERTS = 4; NUM_CLASSES = 100; FEAT_DIM = 256
BATCH_SIZE = 128; EPOCHS_PER_TASK = 120
T_END = 0.3; SEED = 42

def get_task_ranges(num_tasks, class_offset=0, overlap=0):
    if overlap > 0:
        step = 20 - overlap
        ranges = []
        for t in range(num_tasks):
            start = t * step; end = start + 20
            ranges.append((start, min(end, NUM_CLASSES), min(20, NUM_CLASSES-start)))
        return ranges
    total = NUM_CLASSES - class_offset
    base = total // num_tasks; remainder = total % num_tasks
    ranges = []; cur = class_offset
    for t in range(num_tasks):
        n = base + (1 if t < remainder else 0)
        ranges.append((cur, cur+n, n)); cur += n
    return ranges

CIFAR_MEAN = (0.5071, 0.4867, 0.4408); CIFAR_STD = (0.2675, 0.2565, 0.2761)

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
        self.classifier = make_classifier(capacity_idx, num_classes)
        self.activation = nn.Linear(feat_dim, num_experts)
    def forward(self, x):
        return self.classifier(x), torch.sigmoid(self.activation(x))

class ContinualLifecycleMoE(nn.Module):
    def __init__(self, capacity_idx, max_life=10, dominance_threshold=0.35, num_kill=1, cooldown=3,
                 use_snapshot=True, use_min_gen=True, freeze_encoder=False, trainable_router=False,
                 static_routing=False, num_classes=NUM_CLASSES):
        super().__init__()
        self.capacity_idx = capacity_idx; self.max_life = max_life
        self.dominance_threshold = dominance_threshold; self.num_kill = num_kill; self.cooldown = cooldown
        self.use_snapshot = use_snapshot; self.use_min_gen = use_min_gen
        self.freeze_encoder = freeze_encoder; self.trainable_router = trainable_router
        self.static_routing = static_routing; self.static_expert_idx = -1; self.num_classes = num_classes
        self.encoder = SharedEncoder()
        self.experts = nn.ModuleList([MutualExpert(FEAT_DIM, num_classes, NUM_EXPERTS, capacity_idx) for _ in range(NUM_EXPERTS)])
        self.register_buffer('usage_ema', torch.ones(NUM_EXPERTS) / NUM_EXPERTS)
        self.register_buffer('activation_ema', torch.ones(NUM_EXPERTS, NUM_EXPERTS) / NUM_EXPERTS)
        self.life_counter = torch.zeros(NUM_EXPERTS); self.generation = torch.zeros(NUM_EXPERTS)
        self.total_deaths = 0; self.cooldown_counter = 0; self.death_history = []; self.snapshots = []

    def save_snapshot(self, sim_score):
        state = {k: v.clone() for k, v in self.experts.state_dict().items()}
        self.snapshots.append((sim_score, state)); self.snapshots.sort(key=lambda x: x[0]); self.snapshots = self.snapshots[:3]

    def _reinit_expert(self, idx, device):
        for name, param in self.experts[idx].named_parameters():
            if 'weight' in name and param.dim() >= 2: nn.init.kaiming_uniform_(param, a=math.sqrt(5))
            elif 'weight' in name: nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name: nn.init.zeros_(param)
        self.generation[idx] += 1; self.total_deaths += 1
        self.usage_ema[idx] = 1.0 / NUM_EXPERTS
        self.activation_ema[idx, :] = 1.0 / NUM_EXPERTS; self.activation_ema[:, idx] = 1.0 / NUM_EXPERTS

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
            candidates = sorted([(i, self.life_counter[i].item()) for i in range(N) if self.life_counter[i] >= self.max_life], key=lambda x: x[1], reverse=True)
            if self.use_min_gen and candidates:
                candidates.sort(key=lambda x: (self.generation[x[0]].item(), -x[1]))
                kill_indices = [c[0] for c in candidates[:self.num_kill]]
            else: kill_indices = [c[0] for c in candidates[:self.num_kill]]
            for idx in kill_indices:
                self._reinit_expert(idx, device); deaths.append(idx)
            if deaths: self.life_counter[:] = 0; self.cooldown_counter = self.cooldown; self.death_history.append((epoch, deaths))
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

    def set_static_expert(self, idx): self.static_expert_idx = idx

    def forward(self, x, temperature=1.0):
        B = x.size(0); feat = self.encoder(x)
        all_logits, all_act = [], []
        for expert in self.experts:
            logits, act = expert(feat); all_logits.append(logits); all_act.append(act)
        logits = torch.stack(all_logits, dim=1); act_mat = torch.stack(all_act, dim=1) / temperature
        if self.static_routing and self.static_expert_idx >= 0:
            weights = torch.zeros(B, NUM_EXPERTS, device=x.device); weights[:, self.static_expert_idx] = 1.0
            combined = (weights.unsqueeze(-1) * logits).sum(dim=1)
            return combined, weights, logits, act_mat, feat
        with torch.no_grad():
            usage_now = act_mat.mean(dim=1).mean(dim=0)
            self.usage_ema = 0.9 * self.usage_ema + 0.1 * usage_now
            self.activation_ema = 0.9 * self.activation_ema + 0.1 * act_mat.mean(dim=0)
            dead_mask = (self.usage_ema < self.dominance_threshold).float()
            act_mat = act_mat + 0.4 * dead_mask.unsqueeze(0).unsqueeze(0)
        incoming = act_mat.sum(dim=1); weights = F.softmax(incoming, dim=1)
        combined = (weights.unsqueeze(-1) * logits).sum(dim=1)
        return combined, weights, logits, act_mat, feat

def get_task_data(task_id, train=True, num_tasks=5, class_offset=0, overlap=0):
    transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)]) if train else transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    full = datasets.CIFAR100(root='/root/data', train=train, download=False, transform=transform)
    ranges = get_task_ranges(num_tasks, class_offset, overlap)
    start_cls, end_cls, _ = ranges[task_id]
    indices = [i for i, (_, y) in enumerate(full) if start_cls <= y < end_cls]
    return Subset(full, indices), start_cls, end_cls

@torch.no_grad()
def evaluate_all_tasks(model, device, num_tasks=5, class_offset=0, overlap=0):
    model.eval(); results = {}
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    full = datasets.CIFAR100(root='/root/data', train=False, download=False, transform=transform)
    ranges = get_task_ranges(num_tasks, class_offset, overlap)
    for task_id in range(num_tasks):
        start_cls, end_cls, _ = ranges[task_id]
        loader = DataLoader(Subset(full, [i for i, (_, y) in enumerate(full) if start_cls <= y < end_cls]), batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
        if model.static_routing: model.set_static_expert(task_id % NUM_EXPERTS)
        correct = total = 0
        for xb, yb in loader:
            xb = xb.to(device)
            combined, _, _, _, _ = model(xb, temperature=0.3)
            task_logits = combined[:, start_cls:end_cls]; task_labels = yb.to(device) - start_cls
            correct += (task_logits.argmax(1) == task_labels).sum().item(); total += yb.size(0)
        results[f'task{task_id}'] = 100 * correct / total
    if model.static_routing: model.set_static_expert(-1)
    loader = DataLoader(full, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
    correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        combined, _, _, _, _ = model(xb, temperature=0.3)
        correct += (combined.argmax(1) == yb).sum().item(); total += yb.size(0)
    results['full'] = 100 * correct / total
    model.train(); return results

@torch.no_grad()
def analyze_current(model, loader, device):
    model.eval(); all_weights = []; all_logits_for_sim = []
    for xb, _ in loader:
        xb = xb.to(device)
        combined, weights, logits, act_mat, _ = model(xb, temperature=0.3)
        all_weights.append(weights.cpu()); all_logits_for_sim.append(logits.cpu())
    weights = torch.cat(all_weights, dim=0)
    ent = -(weights * torch.log(weights + 1e-8)).sum(dim=1).mean().item()
    usage = weights.mean(dim=0); gini = 1 - (usage**2).sum().item()
    logit = torch.cat(all_logits_for_sim, dim=0)
    sims = []
    for i in range(NUM_EXPERTS):
        for j in range(i+1, NUM_EXPERTS):
            sims.append(F.cosine_similarity(logit[:,i,:].mean(0).unsqueeze(0), logit[:,j,:].mean(0).unsqueeze(0)).item())
    sim_mean = sum(sims) / len(sims) if sims else 0.0
    edges = (model.activation_ema > 0.15).sum().item()
    model.train(); return {'sim_mean': sim_mean, 'edges': edges, 'routing_entropy': ent, 'routing_gini': gini}

# === Main ===
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, required=True)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--num_tasks', type=int, default=5)
parser.add_argument('--overlap', type=int, default=0)
args = parser.parse_args()

NUM_TASKS = args.num_tasks
CLASS_OFFSET = 0

random.seed(args.seed); np.random.seed(args.seed)
torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

dev = torch.device("cuda")

print(f"Loading: {args.ckpt}")
ckpt = torch.load(args.ckpt, map_location=dev)
print(f"  Pre-diff sim={ckpt.get('pretrain_history', [{}])[-1].get('sim_mean', '?') if ckpt.get('pretrain_history') else '?'}  deaths={ckpt.get('total_deaths', '?')}")

model = ContinualLifecycleMoE(3, max_life=10, dominance_threshold=0.35,
                               num_kill=1, cooldown=3,
                               use_snapshot=True, use_min_gen=True,
                               freeze_encoder=True, trainable_router=False,
                               static_routing=False, num_classes=100).to(dev)
model.load_state_dict(ckpt['model_state'])
model.life_counter = torch.tensor(ckpt['life_counter']).to(dev)
model.generation = torch.tensor(ckpt['generation']).to(dev)
model.total_deaths = ckpt['total_deaths']
model.cooldown_counter = ckpt['cooldown_counter']
model.death_history = ckpt.get('death_history', [])
model.usage_ema = torch.tensor(ckpt['usage_ema']).to(dev)
model.activation_ema = torch.tensor(ckpt['activation_ema']).to(dev)

opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_TASKS * EPOCHS_PER_TASK)

print(f"Model ready. Tasks={NUM_TASKS}  Overlap={args.overlap}")
t0 = time.time()

test_subset, _, _ = get_task_data(0, train=False, num_tasks=NUM_TASKS, class_offset=CLASS_OFFSET, overlap=args.overlap)
per_task_history = []

for task_id in range(NUM_TASKS):
    train_data, task_start_cls, _ = get_task_data(task_id, train=True, num_tasks=NUM_TASKS, class_offset=CLASS_OFFSET, overlap=args.overlap)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    ranges = get_task_ranges(NUM_TASKS, CLASS_OFFSET, args.overlap)
    print(f"\nTASK {task_id}: classes {ranges[task_id][0]}-{ranges[task_id][1]-1}")

    if task_id > 0:
        active = model.freeze_for_task(task_id)
        tp = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  FREEZE: expert {active} active, {tp/1e6:.1f}M trainable")

    for epoch_in_task in range(EPOCHS_PER_TASK):
        global_epoch = task_id * EPOCHS_PER_TASK + epoch_in_task
        model.train(); total_ce = 0.0; n_batch = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            combined, weights, logits, act_mat, feat = model(xb, temperature=T_END)
            ce_loss = F.cross_entropy(combined, yb)
            loss = ce_loss + 0.01 * (-(weights * torch.log(weights + 1e-8)).sum(dim=1).mean()) + 0.05 * ((act_mat.mean(dim=1).mean(dim=0) - 1.0/NUM_EXPERTS)**2).sum()
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total_ce += ce_loss.item(); n_batch += 1
        sched.step()

        test_loader_small = DataLoader(Subset(test_subset, range(500)), batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
        a = analyze_current(model, test_loader_small, dev)
        deaths = model.check_lifecycle(global_epoch, a['sim_mean'], a['edges'])

        if epoch_in_task % 10 == 0 or deaths:
            life = [str(int(model.life_counter[i].item())) for i in range(4)]
            gen = [str(int(model.generation[i].item())) for i in range(4)]
            ds = f" DEATH: {deaths} gen={gen} total={model.total_deaths}" if deaths else ""
            print(f"  T{task_id} E{epoch_in_task:3d} ce={total_ce/n_batch:.3f} sim={a['sim_mean']:.3f} edges={a['edges']}/16 life=[{','.join(life)}] gen=[{','.join(gen)}]{ds}")

    accs = evaluate_all_tasks(model, dev, num_tasks=NUM_TASKS, class_offset=CLASS_OFFSET, overlap=args.overlap)
    per_task_history.append({'task': task_id, 'accs': accs})
    print(f"  >>> TASK {task_id} DONE: " + " | ".join(f"{k}={v:.1f}%" for k, v in accs.items()))

print(f"\n{'='*60}")
print(f"FINAL after {NUM_TASKS} tasks ({args.overlap} overlap):")
for h in per_task_history:
    print(f"  After T{h['task']}: " + " | ".join(f"{k}={v:.1f}%" for k, v in h['accs'].items()))
elapsed = time.time() - t0
print(f"\nTotal: {elapsed/3600:.1f}h | Deaths: {model.total_deaths} | Gen: {model.generation.tolist()}")
print(f"Final full accuracy: {per_task_history[-1]['accs']['full']:.1f}%")
