.PHONY: setup format test train eval robustness clean

setup:
	python -m venv .venv && .\.venv\Scripts\pip install -r requirements.txt

format:
	black src tests && isort src tests && ruff check src tests --fix

test:
	pytest -q

train:
	python scripts/train.py --config configs/cifar10_baseline.yaml

eval:
	python scripts/evaluate.py --config configs/cifar10_baseline.yaml --checkpoint runs/cifar10_baseline/*/seed_0/best.pt --dataset cifar10

robustness:
	python -m xai_proj_b.cli robustness --config configs/cifar10_baseline.yaml --checkpoint runs/cifar10_baseline/*/seed_0/best.pt --data-root data/own_dataset

clean:
	python - <<'PY'
import shutil
from pathlib import Path
for folder in [Path('runs'), Path('wandb')]:
    if folder.exists():
        shutil.rmtree(folder)
PY
