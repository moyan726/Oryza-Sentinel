from __future__ import annotations

import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def timestamp_string() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        composed = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_dict(value, prefix=composed))
        else:
            flat[composed] = value
    return flat


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_manifest_path(config, repo_root: Path | None = None) -> Path:
    repo_root = repo_root or get_repo_root()
    manifest_root = config.resolve_path(config.manifest_root, repo_root)
    return manifest_root / f"{config.dataset_view}_manifest.csv"


def resolve_output_root(config, repo_root: Path | None = None) -> Path:
    repo_root = repo_root or get_repo_root()
    return config.resolve_path(config.output_root, repo_root)


def resolve_torch_device(requested: str):
    import torch

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)
