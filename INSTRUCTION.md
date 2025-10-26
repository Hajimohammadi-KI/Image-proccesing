# INSTRUCTION.md — Code Implementation Spec (xAI‑Proj‑B)
_For: coding agent “GPT‑5 Codex High”_

## 1) Objective & Scope
Build a **reproducible** PyTorch codebase to train and evaluate image classifiers on:
- **CIFAR‑10** and an **ImageNet‑1k subset** (provided in course VC).
- A **new, student‑collected “hard” test set** of 10 indoor object classes placed in unusual/outdoor contexts, to measure **robustness**.

The code must:
- Support multiple model architectures, training methods, optimizers/schedulers.
- Run controlled **data augmentation experiments**.
- Report clear **evaluation metrics** (overall + per‑class; confusion matrix).
- Evaluate each experiment over **≥3 random seeds** and summarize results.
- Ship with a self‑explanatory **README** and a small **tutorial**.
- Be easy to run locally and on **Google Colab**.

## 2) Tech Stack & Conventions
- **Language**: Python (>=3.10).
- **DL Framework**: PyTorch (>=2.2), torchvision, timm.
- **Experiment Tracking** (optional but supported): Weights & Biases.
- **Config**: YAML‑first; single source of truth for all hyperparameters.
- **CLIs**: One entry CLI dispatching to `train`, `evaluate`, `robustness`, `sweep`.
- **Reproducibility**: Deterministic flags when feasible; seed everything.
- **Packaging**: `pyproject.toml` + `src/` layout; ready to `pip install -e .`.
- **Envs**: `environment.yml` (Conda) + `requirements.txt` (pip) for Colab.
- **Style**: Black, isort, Ruff; pre‑commit hooks.
- **Tests**: Pytest smoke/unit tests for datasets, transforms and determinism.

## 3) Repository Layout (create exactly this)
```
.
├── README.md
├── INSTRUCTION.md
├── environment.yml
├── requirements.txt
├── pyproject.toml
├── setup.cfg
├── .pre-commit-config.yaml
├── .gitignore
├── configs/
│   ├── cifar10_baseline.yaml
│   ├── imagenet_subset_baseline.yaml
│   ├── aug/
│   │   ├── weak.yaml
│   │   └── strong.yaml
│   └── sweeps/
│       └── hparams.yaml
├── data/                # (ignored) user-provided datasets & cache
│   └── own_dataset/     # placeholder with README_DATA.md
├── notebooks/           # exploratory analysis, optional
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── sweep.py
│   ├── prepare_imagenet_subset.py
│   └── validate_own_dataset.py
├── src/
│   ├── xai_proj_b/
│   │   ├── __init__.py
│   │   ├── cli.py
│   │   ├── utils/
│   │   │   ├── config.py
│   │   │   ├── seed.py
│   │   │   ├── logging.py
│   │   │   └── checkpoint.py
│   │   ├── data/
│   │   │   ├── datasets.py
│   │   │   ├── transforms.py
│   │   │   └── loaders.py
│   │   ├── models/
│   │   │   ├── factory.py
│   │   │   └── heads.py
│   │   ├── train/
│   │   │   ├── loop.py
│   │   │   ├── metrics.py
│   │   │   └── callbacks.py
│   │   └── eval/
│   │       ├── evaluate.py
│   │       └── robustness.py
└── tests/
    ├── test_datasets.py
    ├── test_transforms.py
    ├── test_reproducibility.py
    └── test_cli.py
```

## 4) Datasets & Data Rules
### 4.1 Provided datasets
- **CIFAR‑10**: use torchvision’s official split & normalization defaults.
- **ImageNet‑1k subset**: a course‑provided subset; add a script `prepare_imagenet_subset.py` that builds the expected directory structure:
  ```
  data/imagenet_subset/
    train/<class_name>/*.jpg
    val/<class_name>/*.jpg
  ```

### 4.2 Own “hard” test set (student‑collected)
- **Classes** (10): `coffee_mug, wooden_spoon, notebook, teapot, soup_bowl, remote_control, computer_keyboard, mouse, toilet_tissue, binder`.
- **Per‑student quota**: 30 JPEGs per class (bonus: 60). Accept only `.jpg` (case‑insensitive).
- **Diversity**: encourage distinct physical objects and varied orientations/backgrounds.
- **Filename schema** (parser must enforce):
  - Pattern: ``studentID_phoneID_classname_imageID.jpg``
  - Use **hyphen** (`-`) inside multi‑word `classname` when writing filenames (e.g., `coffee-mug`); accept both hyphen/underscore **on read**, but standardize to hyphen internally.
- **Repo location**: group’s public GitHub repo (images live there; code repo must be able to read from a local checkout or a path).
- **Validator**: provide `scripts/validate_own_dataset.py` to check counts, extensions, naming, and class coverage; output a CSV summary.

### 4.3 Data module behavior
- Central factory `src/xai_proj_b/data/datasets.py` with registry keys: `cifar10`, `imagenet_subset`, `own_dataset`.
- Each dataset must expose: `num_classes`, `class_names`, canonical `transforms` for train/val/test.
- Support **class‑balanced** sampling option and standard train/val/test splits.
- Save a `dataset_manifest.json` for each run with resolved file paths and hashes.

## 5) Models & Training
### 5.1 Model zoo
- Implement a `timm`‑backed `models/factory.py` to build by name, e.g.:
  - `resnet18`, `vgg16_bn`, `efficientnet_b0`, `vit_tiny_patch16_224`, etc.
- Support `pretrained: true/false`, custom classifier head, dropout, and input size adaptation.

### 5.2 Optimizers & schedulers
- Implement choices: `sgd`, `adam`, `adamw` and schedulers: `cosine`, `step`, `plateau`.
- All choices are config‑driven; include sensible defaults in baseline configs.

### 5.3 Training loop
- Mixed precision (AMP) toggle, gradient clipping, early stopping, checkpointing (best + last).
- Log **epoch** metrics: loss, top‑1 accuracy (and top‑k), per‑class accuracy.
- Save `metrics.csv` and `metrics.jsonl` per run; include the full **config** in `run.yaml`.
- Optional Weights & Biases logging (`WANDB_PROJECT` env‑based).

### 5.4 Evaluation & robustness
- `evaluate.py` must compute: overall accuracy, per‑class accuracy, confusion matrix, and a table of **failure modes** (top‑k misclassifications per class).
- `robustness.py` must evaluate the trained model on the **own_dataset** (hard test set) with the same normalization and a **no‑augmentation** policy.

## 6) Data Augmentation Experiments
- Provide two presets in `configs/aug/`: `weak.yaml` and `strong.yaml` (e.g., random crop/flip vs. RandAugment/ColorJitter/Cutout/MixUp/CutMix).
- Training script accepts `--aug=weak|strong|none` or a custom YAML path.
- Record augmentations applied in `run.yaml`; for evaluation, **disable** training‑time stochastic aug except test‑time augmentation if explicitly enabled.

## 7) Seeds & Reliability
- Each experiment must support `--seeds 0 1 2` and report the **mean ± std** of metrics across seeds.
- Determinism helper (`utils/seed.py`) must set Python/NumPy/PyTorch seeds and relevant PyTorch deterministic flags (with a `--deterministic` switch).

## 8) Command‑Line Interface (single entry)
Expose via `python -m xai_proj_b.cli` (and short wrappers in `scripts/`).

**Train**
```bash
python -m xai_proj_b.cli train --config configs/cifar10_baseline.yaml --seed 0
python -m xai_proj_b.cli train --config configs/imagenet_subset_baseline.yaml --aug strong --seeds 0 1 2
```

**Evaluate**
```bash
python -m xai_proj_b.cli evaluate --checkpoint runs/cifar10/best.pt --dataset cifar10
python -m xai_proj_b.cli evaluate --checkpoint runs/imagenet_subset/best.pt --dataset own_dataset --data-root data/own_dataset
```

**Robustness**
```bash
python -m xai_proj_b.cli robustness --checkpoint runs/cifar10/best.pt --data-root data/own_dataset
```

**Hyperparameter sweeps**
```bash
python -m xai_proj_b.cli sweep --config configs/sweeps/hparams.yaml --seeds 0 1 2
```

## 9) Config Schema (YAML)
```yaml
experiment: "cifar10_baseline"
output_dir: "runs/${experiment}"

dataset:
  name: cifar10            # cifar10 | imagenet_subset | own_dataset
  root: data/
  batch_size: 128
  num_workers: 8
  img_size: 224            # auto for cifar10 unless overridden
  aug: configs/aug/weak.yaml

model:
  name: resnet18           # any timm model id
  pretrained: true
  dropout: 0.1

optim:
  name: adamw
  lr: 3.0e-4
  weight_decay: 0.05
  momentum: null           # used if sgd

sched:
  name: cosine             # cosine | step | plateau
  epochs: 100
  warmup_epochs: 5
  step_size: 30
  gamma: 0.1

train:
  amp: true
  grad_clip_norm: 1.0
  early_stop_patience: 20
  label_smoothing: 0.0

eval:
  topk: [1, 5]
  tta: false

logging:
  wandb: false
  project: xai-proj-b
  entity: null

seeds: [0, 1, 2]
```

## 10) Results & Artifacts
- Save **checkpoints** (`best.pt`, `last.pt`), **configs**, **metrics**, and a **confusion matrix PNG** per run in `runs/<experiment>/<timestamp>/`.
- Write a `SUMMARY.md` per experiment with:
  - dataset, model, augmentations
  - seed‑wise metrics and mean±std
  - best validation epoch and test scores
  - link to W&B run (if used)

## 11) README Content Checklist (ship with the repo)
- Project motivation & goals.
- Quickstart (create env; download/prepare datasets; run baseline).
- How to reproduce the paper‑grade results (exact configs + seeds).
- Explanation of code layout and extension points.
- Tutorial: training CIFAR‑10 baseline and evaluating on own_dataset.
- Contact information and (optional) license for images (e.g., MIT for reuse).

## 12) Makefile (optional but recommended)
Targets: `setup`, `format`, `test`, `train`, `eval`, `robustness`, `clean`.

## 13) Unit Tests (minimum set)
- `test_datasets.py`: parsing of own_dataset names; class coverage; split sizes.
- `test_transforms.py`: deterministic behavior given fixed seeds.
- `test_reproducibility.py`: ensure metric parity within tolerance across seeds.
- `test_cli.py`: argument parsing and config merging.

## 14) Colab Support
- Provide a `notebooks/colab_quickstart.ipynb` that:
  - Installs `requirements.txt`,
  - Downloads CIFAR‑10 automatically,
  - Mounts/loads `imagenet_subset` and `own_dataset` from Drive/Git,
  - Trains the CIFAR‑10 baseline, evaluates, and exports artifacts.

## 15) Delivery Acceptance Criteria
- Repo clones and **runs end‑to‑end** with the provided configs.
- `train → evaluate → robustness` works for both CIFAR‑10 and ImageNet‑subset.
- `validate_own_dataset.py` passes on the group’s image repo.
- `README` covers motivation, how‑it‑works, tutorial, and contact details.
- Metrics are present for **≥3 seeds** with mean±std summaries.
- Code quality: formatting, linting, tests passing.

---

### Implementation Notes for the Coding Agent
- Prefer composition over inheritance; keep modules small.
- Avoid global state; pass config explicitly.
- Log everything needed for full reproducibility.
- Keep imports minimal; guard Weights & Biases import behind a flag.
- Write first; then optimize. Baseline first, then aug/robustness studies.

