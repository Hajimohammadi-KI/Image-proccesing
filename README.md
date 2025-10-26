# xAI Project B - Vision Robustness Toolkit

This repo implements the full training/evaluation stack requested in `INSTRUCTION.md`: a reproducible PyTorch pipeline for CIFAR-10, the course-provided ImageNet-1k subset, and a new hard indoor-object dataset. All commands were smoke-tested on Windows 11 + Python 3.13 with an RTX 5080 GPU (CUDA 12.6) and degrade gracefully to CPU.

## Why this exists
- Compare multiple architectures/optimizers under controlled augmentation policies.
- Quantify robustness via mixup/cutmix, corruptions, and a custom 10-class dataset.
- Enforce filename/data-quality rules for the student-collected images.
- Track accuracy/precision/recall, per-class metrics, and confusion matrices across **>=3 seeds**.

## Quickstart
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
Prepare data assets:
```powershell
python scripts/prepare_imagenet_subset.py --input-dir ImageNetSubset/raw --output-dir ImageNetSubset
python scripts/validate_own_dataset.py --data-root data/own_dataset
python scripts/build_robust_dataset.py --output-dir data/robustness_dataset --samples-per-class 400
```

## Training & evaluation
All workflows go through the central CLI (also available via `python -m xai_proj_b.cli`).
```powershell
python scripts/train.py --config configs/cifar10_baseline.yaml
python scripts/train.py --config configs/imagenet_subset_baseline.yaml --aug strong --dataset-root ImageNetSubset
python scripts/evaluate.py --config configs/cifar10_baseline.yaml --checkpoint runs/cifar10_baseline/*/seed_0/best.pt --dataset cifar10
python -m xai_proj_b.cli robustness --config configs/cifar10_baseline.yaml --checkpoint runs/cifar10_baseline/*/seed_0/best.pt --data-root data/own_dataset
python scripts/sweep.py --config configs/sweeps/hparams.yaml
```
Each run saves results in `runs/<experiment>/<timestamp>/seed_<id>/`:
- `best.pt`, `last.pt`
- `metrics.csv`
- `confusion.png` + per-class precision/recall JSON
- `SUMMARY.json` and `SUMMARY.md` with seed-wise stats and mean+/-std aggregates

## Config-driven experiments
YAML is the single source of truth. Example (`configs/cifar10_baseline.yaml`):
```yaml
dataset:
  name: cifar10
  root: data/cifar10
  aug: configs/aug/weak.yaml
model:
  name: resnet18
  pretrained: true
optim:
  name: sgd
  lr: 0.05
sched:
  name: cosine
  epochs: 60
train:
  grad_clip_norm: 1.0
seeds: [0, 1, 2]
```
Override on the fly with CLI flags (`--aug strong`, `--dataset-root ImageNetSubset`, `--seeds 4 5 6`, `--output-dir custom_runs`).

## Tutorial snippet
1. **Train CIFAR-10 baseline**: `python scripts/train.py --config configs/cifar10_baseline.yaml`
2. **Evaluate on own dataset**: place your images under `data/own_dataset/{train,val}/<class>/`, then run `python -m xai_proj_b.cli robustness --config configs/cifar10_baseline.yaml --checkpoint <best.pt> --data-root data/own_dataset/val`.
3. **Interpret metrics**: inspect `SUMMARY.json` for mean+/-std scores, and open `confusion.png` for failure analysis.

## Google Colab usage
1. **Create the notebook**  
   - Open [colab.research.google.com](https://colab.research.google.com), click “New Notebook”, then choose *Runtime → Change runtime type → GPU*.

2. **Clone the repo**
   ```python
   !git clone https://github.com/Hajimohammadi-KI/Image-proccesing.git
   %cd /content/Image-proccesing
   ```

3. **Install requirements** (CPU wheels work fine on Colab GPUs)
   ```python
   %pip install -r requirements.txt
   ```

4. **Install the repo itself (editable mode so `xai_proj_b` resolves)**
   ```python
   %pip install -e .
   ```

5. **Mount Google Drive (optional)**  
   Needed if the ImageNet subset or custom dataset lives in Drive.
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   !ln -s /content/drive/MyDrive/ImageNetSubset ImageNetSubset
   !ln -s /content/drive/MyDrive/data/own_dataset data/own_dataset
   ```

6. **Train**
   ```python
   !python -m xai_proj_b.cli train --config configs/cifar10_baseline.yaml --output-dir /content/runs
   ```
   Adjust `--config`, `--dataset-root`, `--aug`, or `--seeds` as needed.

7. **Evaluate**
   ```python
   !python -m xai_proj_b.cli evaluate --config configs/cifar10_baseline.yaml \
       --checkpoint /content/runs/cifar10_baseline/*/seed_0/best.pt --dataset cifar10
   ```
   For robustness on your own dataset:
   ```python
   !python -m xai_proj_b.cli robustness --config configs/cifar10_baseline.yaml \
       --checkpoint /content/runs/cifar10_baseline/*/seed_0/best.pt --data-root data/own_dataset
   ```

8. **Save artifacts**  
   Copy results back to Drive or download:
   ```python
   !cp -r /content/runs /content/drive/MyDrive/runs_backup
   ```
   or use the Colab file browser to download specific folders.

> Tip: instead of typing these commands manually, upload `notebooks/colab_quickstart.ipynb` from the repo and run it top-to-bottom; it already contains the cells above.

## Progress dashboard (local browser)
1. Start a training run (CPU or GPU). Each new run folder writes `progress.json` after every epoch.
2. Serve the dashboard:
   ```powershell
   python scripts/serve_progress.py --run-dir runs/cpu_cifar10/<timestamp>
   ```
   Replace `<timestamp>` with the folder created for your run. The script copies `progress_dashboard.html` into that directory and opens it in your browser.
3. The HTML/JS page polls `progress.json` every few seconds and shows percent complete, train/val loss, accuracy, and elapsed time. Leave the server running until training finishes.

## Code layout
```
configs/                 # YAML configs + augmentation presets + sweep definitions
scripts/                 # Thin wrappers around the CLI + dataset utilities
src/xai_proj_b/          # Installable package (pip install -e .)
|-- cli.py               # train/evaluate/robustness/sweep dispatcher
|-- utils/               # config loader, seeding, logging, checkpoints
|-- data/                # dataset builders, transforms, loaders
|-- models/              # timm-backed factory + classifier heads
|-- train/               # training loop, callbacks, metrics, summaries
|-- eval/                # evaluation + robustness helpers and dataset builder
web/                     # Static assets (progress dashboard template)
runs/, data/, wandb/     # git-ignored experiment outputs
```
Tests live under `tests/` (pytest), and a lightweight Colab notebook sits at `notebooks/colab_quickstart.ipynb` for remote execution.

## Syncing with GitHub
1. Log in to GitHub and create an empty repository (no README/license).
2. From this folder run:
   ```powershell
   git init
   git add .
   git commit -m "Initial project import"
   git branch -M main
   git remote add origin https://github.com/<user>/<repo>.git
   git push -u origin main
   ```
3. For subsequent updates: `git add -u`, `git commit -m "..."; git push`.

## Contact & licensing
- Code: MIT (specify in your submission if different). User-provided images remain under their original licenses.
