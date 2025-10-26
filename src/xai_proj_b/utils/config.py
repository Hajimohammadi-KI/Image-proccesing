from __future__ import annotations

import copy
import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _expand_path(value: str, variables: Dict[str, Any]) -> str:
    if not isinstance(value, str):
        return value
    expanded = value
    for key, val in variables.items():
        expanded = expanded.replace(f"${{{key}}}", str(val))
    return os.path.expanduser(expanded)


@dataclass
class DatasetConfig:
    name: str
    root: str
    batch_size: int
    num_workers: int
    img_size: Optional[int] = None
    aug: Optional[str] = None
    val_batch_size: Optional[int] = None
    download: bool = True
    pin_memory: bool = True
    persistent_workers: bool = False
    drop_last: bool = False
    shuffle: bool = True
    train_subdir: str = "train"
    val_subdir: str = "val"


@dataclass
class ModelConfig:
    name: str
    pretrained: bool = True
    dropout: float = 0.0


@dataclass
class OptimConfig:
    name: str
    lr: float
    weight_decay: float = 0.0
    momentum: Optional[float] = None
    betas: Optional[List[float]] = None


@dataclass
class SchedulerConfig:
    name: str
    epochs: int
    warmup_epochs: int = 0
    step_size: int = 30
    gamma: float = 0.1
    min_lr: float = 1e-6


@dataclass
class TrainConfig:
    amp: bool = True
    grad_clip_norm: Optional[float] = None
    early_stop_patience: Optional[int] = None
    label_smoothing: float = 0.0
    deterministic: bool = False


@dataclass
class EvalConfig:
    topk: List[int] = field(default_factory=lambda: [1, 5])
    tta: bool = False


@dataclass
class LoggingConfig:
    wandb: bool = False
    project: Optional[str] = None
    entity: Optional[str] = None


@dataclass
class ExperimentConfig:
    experiment: str
    output_dir: str
    dataset: DatasetConfig
    model: ModelConfig
    optim: OptimConfig
    sched: SchedulerConfig
    train: TrainConfig
    eval: EvalConfig
    logging: LoggingConfig
    seeds: List[int] = field(default_factory=lambda: [0])

    def expanded_output_dir(self) -> Path:
        variables = {"experiment": self.experiment}
        return Path(_expand_path(self.output_dir, variables)).resolve()


def _dict_to_dataclass(cls, data: Dict[str, Any]):
    field_names = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in data.items() if k in field_names}
    return cls(**filtered)


def load_yaml(path: str | os.PathLike[str]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


def load_config(path: str, overrides: Optional[Dict[str, Any]] = None) -> ExperimentConfig:
    overrides = overrides or {}
    raw = load_yaml(path)
    raw = _merge_dicts(raw, overrides)
    dataset_cfg = _dict_to_dataclass(DatasetConfig, raw["dataset"])
    model_cfg = _dict_to_dataclass(ModelConfig, raw["model"])
    optim_cfg = _dict_to_dataclass(OptimConfig, raw["optim"])
    sched_cfg = _dict_to_dataclass(SchedulerConfig, raw["sched"])
    train_cfg = _dict_to_dataclass(TrainConfig, raw.get("train", {}))
    eval_cfg = _dict_to_dataclass(EvalConfig, raw.get("eval", {}))
    logging_cfg = _dict_to_dataclass(LoggingConfig, raw.get("logging", {}))

    return ExperimentConfig(
        experiment=raw["experiment"],
        output_dir=raw.get("output_dir", f"runs/${{experiment}}"),
        dataset=dataset_cfg,
        model=model_cfg,
        optim=optim_cfg,
        sched=sched_cfg,
        train=train_cfg,
        eval=eval_cfg,
        logging=logging_cfg,
        seeds=raw.get("seeds", [0]),
    )


def _merge_dicts(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if key not in merged or not isinstance(value, dict):
            merged[key] = value
            continue
        merged[key] = _merge_dicts(merged.get(key, {}), value)
    return merged


def dataclass_to_dict(obj: Any) -> Dict[str, Any]:
    return dataclasses.asdict(obj)


def save_config(cfg: ExperimentConfig, path: str | os.PathLike[str]) -> None:
    with open(path, "w", encoding="utf-8") as fp:
        yaml.safe_dump(dataclass_to_dict(cfg), fp, sort_keys=False)
