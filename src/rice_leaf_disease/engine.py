from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from .config import ExperimentConfig, dump_config
from .data import build_dataloader, infer_class_names, load_manifest
from .models import TRANSFER_MODELS, build_model, set_backbone_trainable
from .utils import ensure_dir, resolve_manifest_path, resolve_output_root, resolve_torch_device, save_json, set_seed


@dataclass
class TrainResult:
    run_dir: Path
    checkpoint_path: Path
    history_path: Path
    history: pd.DataFrame
    device: str
    best_val_accuracy: float
    manifest_path: Path
    pretrained_loaded: bool


def build_optimizer(model: nn.Module, optimizer_name: str, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    name = optimizer_name.lower()
    if name == "adamw":
        return AdamW(parameters, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return SGD(parameters, lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def create_scheduler(optimizer: torch.optim.Optimizer, patience: int) -> ReduceLROnPlateau:
    return ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=patience)


def _amp_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def run_epoch(
    model: nn.Module,
    dataloader,
    criterion,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    max_batches: int | None = None,
    description: str = "train",
) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(mode=is_training)
    running_loss = 0.0
    running_correct = 0
    total_samples = 0
    progress = tqdm(dataloader, desc=description, leave=False)
    amp_enabled = bool(scaler is not None and scaler.is_enabled())
    for step, batch in enumerate(progress, start=1):
        if max_batches is not None and step > max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        if is_training:
            optimizer.zero_grad(set_to_none=True)
        with _amp_context(device=device, enabled=amp_enabled):
            outputs = model(images)
            loss = criterion(outputs, labels)
        if is_training:
            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        predictions = outputs.argmax(dim=1)
        batch_size = labels.size(0)
        running_loss += float(loss.item()) * batch_size
        running_correct += int((predictions == labels).sum().item())
        total_samples += batch_size
    epoch_loss = running_loss / max(total_samples, 1)
    epoch_accuracy = running_correct / max(total_samples, 1)
    return {"loss": epoch_loss, "accuracy": epoch_accuracy}


def plot_training_history(history_df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="train_loss", color="#2563eb")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="val_loss", color="#dc2626")
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(history_df["epoch"], history_df["train_accuracy"], label="train_accuracy", color="#059669")
    axes[1].plot(history_df["epoch"], history_df["val_accuracy"], label="val_accuracy", color="#d97706")
    axes[1].set_title("Accuracy Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(figure)


def train_experiment(config: ExperimentConfig, config_path: str | Path, trial=None) -> TrainResult:
    config_path = Path(config_path).resolve()
    repo_root = config_path.parent.parent
    set_seed(config.seed)
    manifest_path = resolve_manifest_path(config, repo_root)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. Please run prepare_dataset.py before training."
        )
    output_root = resolve_output_root(config, repo_root)
    run_dir = ensure_dir(output_root / "runs" / config.experiment_name)
    figure_dir = ensure_dir(run_dir / "figures")

    manifest = load_manifest(manifest_path)
    class_names = config.class_names or infer_class_names(manifest)
    device = resolve_torch_device(config.device)
    train_loader, _train_df = build_dataloader(
        manifest=manifest,
        split=config.train_split,
        img_size=config.img_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        is_training=True,
        shuffle=True,
    )
    val_loader, _val_df = build_dataloader(
        manifest=manifest,
        split=config.val_split,
        img_size=config.img_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        is_training=False,
        shuffle=False,
    )

    built = build_model(
        model_name=config.model_name,
        num_classes=len(class_names),
        dropout=config.dropout,
        use_pretrained=config.use_pretrained,
        allow_pretrained_fallback=config.allow_pretrained_fallback,
    )
    model = built.model.to(device)

    if config.model_name.lower() in TRANSFER_MODELS and config.freeze_epochs > 0:
        set_backbone_trainable(model, config.model_name, trainable=False)
    else:
        set_backbone_trainable(model, config.model_name, trainable=True)

    optimizer = build_optimizer(model, config.optimizer, config.lr, config.weight_decay)
    scheduler = create_scheduler(optimizer, config.scheduler_patience)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and config.amp != "off")

    history_rows: list[dict[str, Any]] = []
    best_val_accuracy = -1.0
    best_val_loss = float("inf")
    stale_epochs = 0
    total_epochs = max(config.freeze_epochs, 0) + max(config.finetune_epochs, 0)
    if total_epochs <= 0:
        raise ValueError("freeze_epochs + finetune_epochs must be positive.")

    for epoch in range(1, total_epochs + 1):
        entering_finetune = (
            config.model_name.lower() in TRANSFER_MODELS
            and config.freeze_epochs > 0
            and epoch == config.freeze_epochs + 1
        )
        if entering_finetune:
            set_backbone_trainable(model, config.model_name, trainable=True)
            optimizer = build_optimizer(model, config.optimizer, config.lr * 0.1, config.weight_decay)
            scheduler = create_scheduler(optimizer, config.scheduler_patience)

        phase = (
            "freeze"
            if config.model_name.lower() in TRANSFER_MODELS and epoch <= config.freeze_epochs
            else "finetune"
        )
        train_stats = run_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            max_batches=config.max_train_batches_per_epoch,
            description=f"train:{epoch}",
        )
        val_stats = run_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            scaler=None,
            max_batches=config.max_eval_batches_per_epoch,
            description=f"val:{epoch}",
        )
        scheduler.step(val_stats["loss"])
        current_lr = float(optimizer.param_groups[0]["lr"])
        history_rows.append(
            {
                "epoch": epoch,
                "phase": phase,
                "train_loss": train_stats["loss"],
                "train_accuracy": train_stats["accuracy"],
                "val_loss": val_stats["loss"],
                "val_accuracy": val_stats["accuracy"],
                "learning_rate": current_lr,
            }
        )

        improved = val_stats["accuracy"] > best_val_accuracy + config.min_delta
        if improved:
            best_val_accuracy = val_stats["accuracy"]
            best_val_loss = val_stats["loss"]
            stale_epochs = 0
            checkpoint = {
                "model_state": model.state_dict(),
                "config": config.to_dict(),
                "class_names": class_names,
                "best_val_accuracy": best_val_accuracy,
                "best_val_loss": best_val_loss,
                "manifest_path": str(manifest_path),
                "pretrained_loaded": built.pretrained_loaded,
            }
            torch.save(checkpoint, run_dir / "best_model.pt")
        else:
            stale_epochs += 1

        if trial is not None:
            trial.report(val_stats["accuracy"], step=epoch)
            if trial.should_prune():
                raise RuntimeError("Trial pruned by Optuna.")

        if stale_epochs >= config.early_stopping_patience:
            break

    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config.to_dict(),
            "class_names": class_names,
            "best_val_accuracy": best_val_accuracy,
            "best_val_loss": best_val_loss,
            "manifest_path": str(manifest_path),
            "pretrained_loaded": built.pretrained_loaded,
        },
        run_dir / "last_model.pt",
    )

    history_df = pd.DataFrame(history_rows)
    history_path = run_dir / "history.csv"
    history_df.to_csv(history_path, index=False, encoding="utf-8-sig")
    plot_training_history(history_df, figure_dir / "training_curves.png")
    dump_config(config, run_dir / "resolved_config.yaml")
    save_json(
        {
            "experiment_name": config.experiment_name,
            "dataset_view": config.dataset_view,
            "model_name": config.model_name,
            "device": str(device),
            "manifest_path": str(manifest_path),
            "best_val_accuracy": best_val_accuracy,
            "best_val_loss": best_val_loss,
            "pretrained_loaded": built.pretrained_loaded,
            "epochs_completed": int(history_df["epoch"].max()),
        },
        run_dir / "run_summary.json",
    )
    return TrainResult(
        run_dir=run_dir,
        checkpoint_path=run_dir / "best_model.pt",
        history_path=history_path,
        history=history_df,
        device=str(device),
        best_val_accuracy=best_val_accuracy,
        manifest_path=manifest_path,
        pretrained_loaded=built.pretrained_loaded,
    )
