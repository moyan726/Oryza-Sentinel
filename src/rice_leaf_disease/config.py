from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CLASS_NAMES = [
    "Bacterialblight",
    "Blast",
    "Brownspot",
    "Tungro",
]


@dataclass
class ExperimentConfig:
    experiment_name: str
    dataset_root: str = "Rice Leaf Disease Images"
    manifest_root: str = "outputs/manifests"
    output_root: str = "outputs"
    dataset_view: str = "official"
    model_name: str = "efficientnet_b0"
    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 0
    optimizer: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.3
    label_smoothing: float = 0.0
    freeze_epochs: int = 3
    finetune_epochs: int = 10
    early_stopping_patience: int = 4
    scheduler_patience: int = 2
    min_delta: float = 0.0
    seed: int = 42
    device: str = "auto"
    amp: str = "auto"
    use_pretrained: bool = True
    allow_pretrained_fallback: bool = True
    train_split: str = "train"
    val_split: str = "validation"
    test_split: str = "test"
    eval_after_train: bool = True
    class_names: list[str] = field(default_factory=lambda: DEFAULT_CLASS_NAMES.copy())
    max_train_batches_per_epoch: int | None = None
    max_eval_batches_per_epoch: int | None = None
    tuning_trials: int = 20
    tuning_timeout_minutes: int | None = None

    def resolve_path(self, value: str, repo_root: Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return (repo_root / path).resolve()

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(config_path: str | Path, overrides: dict[str, Any] | None = None) -> tuple[ExperimentConfig, Path]:
    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if overrides:
        raw.update(overrides)
    config = ExperimentConfig(**raw)
    return config, config_path


def dump_config(config: ExperimentConfig, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, allow_unicode=True, sort_keys=False)
