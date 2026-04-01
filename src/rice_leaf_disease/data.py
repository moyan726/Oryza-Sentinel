from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

try:
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
except ModuleNotFoundError:
    DataLoader = None

    class Dataset:  # type: ignore[override]
        pass

    transforms = None


SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
AUGMENT_KEYWORDS = ("rotated", "flip", "aug", "copy")


def hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def scan_official_dataset(dataset_root: str | Path, class_names: Iterable[str]) -> pd.DataFrame:
    dataset_root = Path(dataset_root).resolve()
    rows: list[dict[str, object]] = []
    class_to_label = {name: index for index, name in enumerate(class_names)}
    for split_dir in sorted(dataset_root.iterdir()):
        if not split_dir.is_dir():
            continue
        split = split_dir.name
        if split not in {"train", "validation", "test"}:
            continue
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            if class_name not in class_to_label:
                continue
            for image_path in sorted(class_dir.iterdir()):
                if not image_path.is_file():
                    continue
                if image_path.suffix.lower() not in SUPPORTED_SUFFIXES:
                    continue
                rows.append(
                    {
                        "dataset_view": "official",
                        "split": split,
                        "class_name": class_name,
                        "label": class_to_label[class_name],
                        "file_name": image_path.name,
                        "relative_path": image_path.relative_to(dataset_root).as_posix(),
                        "abs_path": str(image_path.resolve()),
                        "image_hash": hash_file(image_path),
                        "is_augmented_name": int(any(keyword in image_path.stem.lower() for keyword in AUGMENT_KEYWORDS)),
                    }
                )
    if not rows:
        raise FileNotFoundError(f"No supported images found under {dataset_root}")
    manifest = pd.DataFrame(rows)
    manifest["duplicate_count_global"] = manifest.groupby("image_hash")["image_hash"].transform("size")
    manifest["duplicate_count_split"] = manifest.groupby(["split", "image_hash"])["image_hash"].transform("size")
    manifest["cross_split_duplicate"] = manifest.groupby("image_hash")["split"].transform("nunique").gt(1).astype(int)
    return manifest.sort_values(["split", "label", "relative_path"]).reset_index(drop=True)


def _stratified_three_way_split(labels: list[int], seed: int, ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)) -> list[str]:
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("Split ratios must sum to 1.0")
    rng = np.random.default_rng(seed)
    labels_np = np.array(labels)
    splits = np.empty(len(labels_np), dtype=object)
    for label in sorted(set(labels)):
        indices = np.flatnonzero(labels_np == label)
        rng.shuffle(indices)
        total = len(indices)
        train_count = max(1, int(round(total * ratios[0])))
        val_count = max(1, int(round(total * ratios[1])))
        if train_count + val_count >= total:
            train_count = max(1, total - 2)
            val_count = 1
        test_count = total - train_count - val_count
        if test_count <= 0:
            test_count = 1
            if train_count > val_count:
                train_count -= 1
            else:
                val_count -= 1
        splits[indices[:train_count]] = "train"
        splits[indices[train_count : train_count + val_count]] = "validation"
        splits[indices[train_count + val_count :]] = "test"
    return splits.tolist()


def build_clean_manifest(official_manifest: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_rows: list[dict[str, object]] = []
    duplicate_rows: list[dict[str, object]] = []
    grouped = official_manifest.groupby("image_hash", sort=False)
    for image_hash, group in grouped:
        class_names = sorted(group["class_name"].unique().tolist())
        class_name = class_names[0]
        representative = group.iloc[0]
        group_rows.append(
            {
                "dataset_view": "clean",
                "image_hash": image_hash,
                "class_name": class_name,
                "label": int(representative["label"]),
                "file_name": representative["file_name"],
                "relative_path": representative["relative_path"],
                "abs_path": representative["abs_path"],
                "source_count": int(len(group)),
                "source_splits": "|".join(sorted(group["split"].unique().tolist())),
                "label_conflict": int(len(class_names) > 1),
                "contains_augmented_name": int(group["is_augmented_name"].max()),
            }
        )
        if len(group) > 1:
            duplicate_rows.append(
                {
                    "image_hash": image_hash,
                    "class_name": class_name,
                    "source_count": int(len(group)),
                    "source_splits": "|".join(sorted(group["split"].unique().tolist())),
                    "paths": " || ".join(group["relative_path"].tolist()),
                }
            )
    clean_manifest = pd.DataFrame(group_rows)
    clean_manifest["split"] = _stratified_three_way_split(clean_manifest["label"].tolist(), seed=seed)
    clean_manifest = clean_manifest.sort_values(["split", "label", "relative_path"]).reset_index(drop=True)
    duplicate_report = pd.DataFrame(duplicate_rows).sort_values(["source_count", "class_name"], ascending=[False, True])
    return clean_manifest, duplicate_report


def save_manifests(
    official_manifest: pd.DataFrame,
    clean_manifest: pd.DataFrame,
    duplicate_report: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "official": output_dir / "official_manifest.csv",
        "clean": output_dir / "clean_manifest.csv",
        "duplicates": output_dir / "duplicate_groups.csv",
    }
    official_manifest.to_csv(paths["official"], index=False, encoding="utf-8-sig")
    clean_manifest.to_csv(paths["clean"], index=False, encoding="utf-8-sig")
    duplicate_report.to_csv(paths["duplicates"], index=False, encoding="utf-8-sig")
    return paths


def summarize_manifest(manifest: pd.DataFrame) -> dict[str, object]:
    summary: dict[str, object] = {
        "total_images": int(len(manifest)),
        "unique_hashes": int(manifest["image_hash"].nunique()),
        "split_counts": manifest["split"].value_counts().sort_index().to_dict(),
        "class_counts": manifest["class_name"].value_counts().sort_index().to_dict(),
        "split_class_counts": (
            manifest.groupby(["split", "class_name"]).size().rename("count").reset_index().to_dict(orient="records")
        ),
    }
    if "cross_split_duplicate" in manifest.columns:
        summary["cross_split_duplicate_images"] = int(manifest["cross_split_duplicate"].sum())
    if "duplicate_count_global" in manifest.columns:
        summary["duplicate_hash_groups"] = int((manifest["duplicate_count_global"] > 1).sum())
    return summary


def plot_manifest_distribution(manifest: pd.DataFrame, output_path: str | Path, title: str) -> None:
    pivot = manifest.groupby(["split", "class_name"]).size().unstack(fill_value=0)
    ax = pivot.plot(kind="bar", figsize=(10, 6))
    ax.set_title(title)
    ax.set_xlabel("Split")
    ax.set_ylabel("Image Count")
    ax.legend(title="Class")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_duplicate_overview(official_manifest: pd.DataFrame, output_path: str | Path) -> None:
    counters = Counter(
        "|".join(sorted(group["split"].unique().tolist()))
        for _, group in official_manifest.groupby("image_hash", sort=False)
        if len(group) > 1
    )
    if not counters:
        counters = Counter({"no_duplicates": 1})
    names = list(counters.keys())
    values = list(counters.values())
    plt.figure(figsize=(8, 5))
    plt.bar(names, values, color="#d97706")
    plt.title("Duplicate Hash Groups by Split Membership")
    plt.xlabel("Split Combination")
    plt.ylabel("Group Count")
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def load_manifest(manifest_path: str | Path) -> pd.DataFrame:
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    return pd.read_csv(manifest_path)


def infer_class_names(manifest: pd.DataFrame) -> list[str]:
    class_pairs = manifest[["label", "class_name"]].drop_duplicates().sort_values("label")
    return class_pairs["class_name"].tolist()


def build_transforms(img_size: int, is_training: bool):
    if transforms is None:
        raise ModuleNotFoundError("torchvision is required for training/evaluation dataloaders.")
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    if is_training:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
                transforms.ToTensor(),
                transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
            ]
        )
    resize_size = int(round(img_size * 1.14))
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )


def build_split_dataframe(manifest: pd.DataFrame, split: str) -> pd.DataFrame:
    split_df = manifest[manifest["split"] == split].copy()
    if split_df.empty:
        raise ValueError(f"Split '{split}' is empty in the provided manifest.")
    return split_df.reset_index(drop=True)


def build_dataloader(
    manifest: pd.DataFrame,
    split: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    is_training: bool,
    shuffle: bool | None = None,
) -> tuple[DataLoader, pd.DataFrame]:
    if DataLoader is None:
        raise ModuleNotFoundError("PyTorch is required for training/evaluation dataloaders.")
    split_df = build_split_dataframe(manifest, split)
    dataset = ManifestDataset(split_df, transform=build_transforms(img_size, is_training))
    if shuffle is None:
        shuffle = is_training
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=bool(num_workers > 0),
    )
    return dataloader, split_df


class ManifestDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        image = Image.open(row["abs_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {
            "image": image,
            "label": int(row["label"]),
            "path": row["abs_path"],
            "class_name": row["class_name"],
        }
