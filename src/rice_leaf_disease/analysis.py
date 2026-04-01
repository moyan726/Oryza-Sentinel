from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm

from .config import ExperimentConfig
from .data import build_dataloader, infer_class_names, load_manifest
from .gradcam import GradCAM, preprocess_single_image, save_gradcam_overlay
from .models import get_target_layer, load_model_from_checkpoint
from .utils import ensure_dir, resolve_manifest_path, resolve_output_root, resolve_torch_device, save_json


def collect_predictions(model, dataloader, device: torch.device, max_batches: int | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    progress = tqdm(dataloader, desc="predict", leave=False)
    softmax = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        for step, batch in enumerate(progress, start=1):
            if max_batches is not None and step > max_batches:
                break
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            outputs = model(images)
            probabilities = softmax(outputs)
            confidences, predictions = probabilities.max(dim=1)
            for index in range(images.size(0)):
                rows.append(
                    {
                        "path": batch["path"][index],
                        "true_label": int(labels[index].item()),
                        "pred_label": int(predictions[index].item()),
                        "confidence": float(confidences[index].item()),
                        "correct": int(predictions[index].item() == labels[index].item()),
                    }
                )
    return pd.DataFrame(rows)


def compute_metrics(predictions: pd.DataFrame, class_names: list[str]) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    y_true = predictions["true_label"].to_numpy()
    y_pred = predictions["pred_label"].to_numpy()
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        average=None,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.index.name = "label_name"
    report_df = report_df.reset_index()
    confusion = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    metrics = {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "per_class": [
            {
                "class_name": class_names[index],
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in range(len(class_names))
        ],
    }
    return metrics, report_df, confusion


def plot_confusion_matrix(confusion: np.ndarray, class_names: list[str], output_path: str | Path, title: str) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="YlGnBu", xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_class_metrics(report_df: pd.DataFrame, output_path: str | Path) -> None:
    per_class = report_df[
        report_df["label_name"].isin(["Bacterialblight", "Blast", "Brownspot", "Tungro"])
    ].copy()
    melted = per_class.melt(
        id_vars="label_name",
        value_vars=["precision", "recall", "f1-score"],
        var_name="metric",
        value_name="value",
    )
    plt.figure(figsize=(10, 5))
    sns.barplot(data=melted, x="label_name", y="value", hue="metric")
    plt.title("Per-Class Precision / Recall / F1")
    plt.xlabel("Class")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_misclassified_examples(predictions: pd.DataFrame, class_names: list[str], output_path: str | Path, max_items: int = 9) -> None:
    wrong = predictions[predictions["correct"] == 0].head(max_items)
    if wrong.empty:
        return
    cols = 3
    rows = int(np.ceil(len(wrong) / cols))
    figure, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = np.array(axes).reshape(-1)
    for axis in axes:
        axis.axis("off")
    for axis, (_, row) in zip(axes, wrong.iterrows()):
        image = plt.imread(row["path"])
        axis.imshow(image)
        axis.set_title(
            f"T:{class_names[int(row['true_label'])]} | P:{class_names[int(row['pred_label'])]}\nConf:{row['confidence']:.3f}"
        )
        axis.axis("off")
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(figure)


def generate_gradcam_gallery(
    model,
    checkpoint: dict[str, Any],
    predictions: pd.DataFrame,
    img_size: int,
    output_dir: str | Path,
    max_items: int = 4,
) -> None:
    output_dir = ensure_dir(output_dir)
    target_layer = get_target_layer(model, checkpoint["config"]["model_name"])
    cam = GradCAM(model, target_layer)
    selected = predictions.head(max_items)
    gallery_paths: list[Path] = []
    try:
        for index, row in enumerate(selected.itertuples(index=False), start=1):
            inputs = preprocess_single_image(row.path, img_size).to(next(model.parameters()).device)
            heatmap = cam.generate(inputs, target_index=row.pred_label)
            overlay_path = output_dir / f"gradcam_{index:02d}.png"
            save_gradcam_overlay(row.path, heatmap, overlay_path)
            gallery_paths.append(overlay_path)
    finally:
        cam.remove()
    if not gallery_paths:
        return
    figure, axes = plt.subplots(1, len(gallery_paths), figsize=(4 * len(gallery_paths), 4))
    axes = np.array(axes).reshape(-1)
    for axis, path in zip(axes, gallery_paths):
        axis.imshow(plt.imread(path))
        axis.axis("off")
        axis.set_title(path.stem)
    plt.tight_layout()
    plt.savefig(output_dir / "gradcam_gallery.png", dpi=220)
    plt.close(figure)


def update_leaderboard(summary_row: dict[str, Any], leaderboard_path: str | Path) -> pd.DataFrame:
    leaderboard_path = Path(leaderboard_path)
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    if leaderboard_path.exists():
        leaderboard = pd.read_csv(leaderboard_path)
    else:
        leaderboard = pd.DataFrame()
    leaderboard = pd.concat([leaderboard, pd.DataFrame([summary_row])], ignore_index=True)
    dedupe_cols = ["experiment_name", "dataset_view", "model_name", "split", "checkpoint_path"]
    leaderboard = leaderboard.drop_duplicates(subset=dedupe_cols, keep="last")
    leaderboard.to_csv(leaderboard_path, index=False, encoding="utf-8-sig")
    return leaderboard


def generate_summary_plots(leaderboard: pd.DataFrame, output_dir: str | Path) -> None:
    output_dir = ensure_dir(output_dir)
    test_only = leaderboard[leaderboard["split"] == "test"].copy()
    if test_only.empty:
        return
    test_only["model_view"] = test_only["dataset_view"] + "\n" + test_only["model_name"]

    plt.figure(figsize=(10, 5))
    sns.barplot(data=test_only, x="model_view", y="accuracy", hue="dataset_view")
    plt.title("Model Accuracy Comparison")
    plt.xlabel("Dataset View and Model")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison_accuracy.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.barplot(data=test_only, x="model_view", y="macro_f1", hue="dataset_view")
    plt.title("Model Macro-F1 Comparison")
    plt.xlabel("Dataset View and Model")
    plt.ylabel("Macro F1")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison_macro_f1.png", dpi=220)
    plt.close()

    if test_only["dataset_view"].nunique() > 1:
        pivot = test_only.pivot_table(index="model_name", columns="dataset_view", values="accuracy", aggfunc="max")
        pivot.plot(kind="bar", figsize=(8, 5))
        plt.title("Official vs Clean Accuracy Comparison")
        plt.xlabel("Model")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(output_dir / "official_vs_clean_accuracy.png", dpi=220)
        plt.close()


def evaluate_checkpoint(
    config: ExperimentConfig,
    config_path: str | Path,
    checkpoint_path: str | Path,
    split: str | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    repo_root = config_path.parent.parent
    manifest_path = resolve_manifest_path(config, repo_root)
    manifest = load_manifest(manifest_path)
    class_names = config.class_names or infer_class_names(manifest)
    split = split or config.test_split
    device = resolve_torch_device(config.device)
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, device=device)

    dataloader, _ = build_dataloader(
        manifest=manifest,
        split=split,
        img_size=config.img_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        is_training=False,
        shuffle=False,
    )
    predictions = collect_predictions(model, dataloader, device, max_batches=config.max_eval_batches_per_epoch)
    predictions["true_class"] = predictions["true_label"].map(lambda index: class_names[index])
    predictions["pred_class"] = predictions["pred_label"].map(lambda index: class_names[index])

    metrics, report_df, confusion = compute_metrics(predictions, class_names)
    output_root = resolve_output_root(config, repo_root)
    evaluation_dir = ensure_dir(output_root / "evaluations" / config.experiment_name / split)
    figure_dir = ensure_dir(evaluation_dir / "figures")
    predictions.to_csv(evaluation_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    report_df.to_csv(evaluation_dir / "classification_report.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(confusion, index=class_names, columns=class_names).to_csv(
        evaluation_dir / "confusion_matrix.csv",
        encoding="utf-8-sig",
    )
    save_json(metrics, evaluation_dir / "metrics.json")
    plot_confusion_matrix(confusion, class_names, figure_dir / "confusion_matrix.png", title=f"{split} Confusion Matrix")
    plot_class_metrics(report_df, figure_dir / "class_metrics.png")
    plot_misclassified_examples(predictions, class_names, figure_dir / "misclassified_examples.png")
    generate_gradcam_gallery(model, checkpoint, predictions, config.img_size, figure_dir / "gradcam")

    leaderboard = update_leaderboard(
        {
            "experiment_name": config.experiment_name,
            "dataset_view": config.dataset_view,
            "model_name": config.model_name,
            "split": split,
            "accuracy": metrics["accuracy"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
            "checkpoint_path": str(Path(checkpoint_path).resolve()),
        },
        output_root / "summary" / "leaderboard.csv",
    )
    generate_summary_plots(leaderboard, output_root / "summary" / "figures")
    return {
        "evaluation_dir": evaluation_dir,
        "metrics": metrics,
        "report_df": report_df,
        "predictions": predictions,
    }
