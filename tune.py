from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import optuna
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rice_leaf_disease.analysis import evaluate_checkpoint
from rice_leaf_disease.config import ExperimentConfig, load_config
from rice_leaf_disease.engine import train_experiment
from rice_leaf_disease.utils import ensure_dir, resolve_torch_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune EfficientNet-B0 hyperparameters with Optuna.")
    parser.add_argument("--config", type=str, required=True, help="Base config path.")
    parser.add_argument("--trials", type=int, default=None, help="Override trial count.")
    parser.add_argument("--retrain-best", action="store_true", help="Retrain the best trial after search.")
    return parser.parse_args()


def plot_tuning_history(trials_df: pd.DataFrame, output_path: str | Path) -> None:
    complete = trials_df[trials_df["state"] == "COMPLETE"].copy()
    if complete.empty:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(complete["number"], complete["value"], marker="o", color="#2563eb")
    plt.title("Optuna Validation Accuracy History")
    plt.xlabel("Trial")
    plt.ylabel("Validation Accuracy")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_best_params(best_params: dict[str, object], output_path: str | Path) -> None:
    numeric_items = {key: value for key, value in best_params.items() if isinstance(value, (int, float))}
    text_items = {key: value for key, value in best_params.items() if key not in numeric_items}
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    if numeric_items:
        axes[0].barh(list(numeric_items.keys()), list(numeric_items.values()), color="#059669")
        axes[0].set_title("Best Trial Numeric Params")
    else:
        axes[0].axis("off")
    axes[1].axis("off")
    axes[1].set_title("Best Trial Param Notes")
    text_lines = [f"{key}: {value}" for key, value in text_items.items()] or ["No categorical params."]
    axes[1].text(0.0, 1.0, "\n".join(text_lines), va="top", fontsize=11)
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    base_config, config_path = load_config(args.config)
    if base_config.model_name.lower() != "efficientnet_b0":
        raise ValueError("tune.py only supports EfficientNet-B0 based configs.")

    device = resolve_torch_device(base_config.device)
    configured_trials = args.trials or base_config.tuning_trials
    effective_trials = configured_trials if device.type == "cuda" else min(configured_trials, 8)
    tuning_dir = ensure_dir(PROJECT_ROOT / "outputs" / "tuning" / base_config.experiment_name)

    def objective(trial: optuna.Trial) -> float:
        batch_choices = [16, 32, 48] if device.type == "cuda" else [8, 16, 32]
        if base_config.max_train_batches_per_epoch is not None:
            freeze_low, freeze_high = 1, max(1, min(2, base_config.freeze_epochs or 1))
            finetune_low, finetune_high = 1, max(1, min(2, base_config.finetune_epochs or 1))
        else:
            freeze_low, freeze_high = 1, 4
            finetune_low, finetune_high = 4, 10
        overrides = {
            "optimizer": trial.suggest_categorical("optimizer", ["adamw", "sgd"]),
            "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 5e-4, log=True),
            "dropout": trial.suggest_float("dropout", 0.2, 0.5),
            "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.15),
            "freeze_epochs": trial.suggest_int("freeze_epochs", freeze_low, freeze_high),
            "finetune_epochs": trial.suggest_int("finetune_epochs", finetune_low, finetune_high),
            "batch_size": trial.suggest_categorical("batch_size", batch_choices),
            "eval_after_train": False,
            "experiment_name": f"{base_config.experiment_name}_trial_{trial.number:03d}",
        }
        trial_config = ExperimentConfig(**{**base_config.to_dict(), **overrides})
        try:
            result = train_experiment(trial_config, config_path, trial=trial)
            return result.best_val_accuracy
        except RuntimeError as error:
            if "pruned" in str(error).lower():
                raise optuna.TrialPruned(str(error)) from error
            raise

    study = optuna.create_study(direction="maximize", study_name=base_config.experiment_name)
    study.optimize(objective, n_trials=effective_trials)

    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    trials_df.to_csv(tuning_dir / "trials.csv", index=False, encoding="utf-8-sig")
    with (tuning_dir / "best_params.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(study.best_trial.params, handle, allow_unicode=True, sort_keys=False)
    plot_tuning_history(trials_df, tuning_dir / "tuning_history.png")
    plot_best_params(study.best_trial.params, tuning_dir / "best_params.png")

    print(f"Finished Optuna search with {effective_trials} trial(s).")
    print("Best validation accuracy:", study.best_value)
    print("Best params:", study.best_trial.params)

    if args.retrain_best:
        best_overrides = {**study.best_trial.params, "experiment_name": f"{base_config.experiment_name}_tuned"}
        tuned_config = ExperimentConfig(**{**base_config.to_dict(), **best_overrides})
        train_result = train_experiment(tuned_config, config_path)
        if tuned_config.eval_after_train:
            evaluate_checkpoint(tuned_config, config_path, train_result.checkpoint_path, split=tuned_config.test_split)


if __name__ == "__main__":
    main()
