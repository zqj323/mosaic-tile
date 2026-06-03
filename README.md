# Mosaic Tile: Hard-Routed Output-Space Isolation for Collapse-Free MoE Continual Learning

Replication package for *"The Gaussian Attractor in Mixture-of-Experts: Why Soft Routers Collapse and How Mosaic Tiles Solve It"* (Zhang, 2026).

## Overview

This repository contains the core training scripts for two Mixture-of-Experts continual learning protocols:

1. **Expert Protocol** — soft router + AR loss pre-differentiation + expert freezing
2. **Mosaic Tile Architecture** — 20 small tiles (Linear(256,5)) with hard routing by class range

All experiments use WRN-28-10 encoder + CIFAR-100 on NVIDIA RTX 4090 (24 GB).

## Scripts

| Script | Description | Key Arguments |
|--------|-------------|---------------|
| `phase_continual_expert.py` | Expert protocol: pre-diff + freeze + multi-task CL | `--seed`, `--pretrain_epochs`, `--pretrain_lambda_ar`, `--num_tasks`, `--freeze_encoder` |
| `phase_mosaic_tiles.py` | Tile architecture: v1 (naive), v2 (soft frozen), v3 (hard routing), v4 (soft unfrozen) | `--seed`, `--num_tasks`, `--pretrain_epochs` |
| `tile_v3_extended.py` | Tile v3 hard routing with overlap and fine-grain support | `--seed`, `--num_tasks`, `--overlap`, `--pretrain_epochs` |
| `tile_v4_ablation.py` | Tile v4 ablation (unfrozen soft router, 5-task + 10-task) | `--seed`, `--num_tasks`, `--pretrain_epochs` |
| `tile_shuffle.py` | Shuffle experiment: random class-to-tile mapping | `--seed`, `--num_tasks`, `--pretrain_epochs` |

## Quick Start

```bash
# Expert protocol: 5-task continual learning
python phase_continual_expert.py --seed 42 --pretrain_epochs 200 --pretrain_lambda_ar 0.1 --freeze_encoder --num_tasks 5

# Tile v3: 5-task hard routing
python tile_v3_extended.py --seed 42 --num_tasks 5 --pretrain_epochs 200

# Tile v3: overlapping-class tasks (6 tasks, 5-class overlap)
python tile_v3_extended.py --seed 42 --num_tasks 6 --overlap 5 --pretrain_epochs 200

# Tile v3: fine-grain (20 tasks, 5 classes/task)
python tile_v3_extended.py --seed 42 --num_tasks 20 --pretrain_epochs 200
```

## Requirements

- PyTorch 2.x
- torchvision
- numpy
- CUDA 12.x (RTX 4090 or equivalent with ≥24 GB VRAM)

## Expected Results

### Expert Protocol (soft router + AR loss)

| λ_AR | s=42 Full | s=123 Full |
|:----:|:---------:|:----------:|
| 0.05 | 62.3% | 63.3% |
| 0.1  | 64.5% | 66.2% |
| 0.2  | 65.7% | 63.9% |
| 0.5  | 64.5% | 65.6% |

### Tile v3 (hard routing, no AR loss)

| Configuration | s=42 | s=123 | s=789 | Mean |
|:---|:---:|:---:|:---:|:---:|
| 5-task | 85.8% | 82.1% | 83.6% | 83.8% |
| 10-task | 90.5% | 86.5% | 88.1% | 88.4% |
| 20-task | 90.4% | 93.0% | — | 91.7% |
| 33-task | 85.8% | — | — | 85.8% |
| Overlap=5 | 82.8% | 83.2% | 83.5% | 83.2% |

## Citation

```bibtex
@article{zhang2026gaussian,
  title={The Gaussian Attractor in Mixture-of-Experts: Why Soft Routers Collapse and How Mosaic Tiles Solve It},
  author={Zhang, Qingjun},
  year={2026},
  note={Preprint, Research Square}
}
```

## License

CC-BY 4.0
