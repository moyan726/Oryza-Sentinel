from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
FINAL_ROOT = PROJECT_ROOT / "final_assets"


KEY_EXPERIMENTS = [
    "official_customcnn",
    "official_resnet18",
    "official_efficientnet_tuned",
    "clean_customcnn",
    "clean_resnet18",
    "clean_resnet18_seed123",
    "clean_resnet18_seed2026",
    "clean_efficientnet_tuned",
]


def copy_if_exists(source: Path, target: Path) -> None:
    if source.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def export_resnet18_seed_consistency(test_rows: pd.DataFrame, final_root: Path) -> None:
    seed_experiments = ["clean_resnet18", "clean_resnet18_seed123", "clean_resnet18_seed2026"]
    seed_rows = test_rows[test_rows["experiment_name"].isin(seed_experiments)].copy()
    if seed_rows.empty:
        return

    seed_mapping = {
        "clean_resnet18": 42,
        "clean_resnet18_seed123": 123,
        "clean_resnet18_seed2026": 2026,
    }
    seed_rows["seed"] = seed_rows["experiment_name"].map(seed_mapping)
    seed_rows = seed_rows.sort_values("seed")
    export_columns = [
        "experiment_name",
        "dataset_view",
        "model_name",
        "split",
        "seed",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "checkpoint_path",
    ]
    seed_rows.loc[:, export_columns].to_csv(
        final_root / "resnet18_seed_consistency.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metric_columns = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    summary = {
        "seed": "summary",
        "experiment_name": "mean",
        "dataset_view": "clean",
        "model_name": "resnet18",
        "split": "test",
        "checkpoint_path": "",
    }
    for metric in metric_columns:
        summary[metric] = float(seed_rows[metric].mean())
        summary[f"{metric}_std"] = float(seed_rows[metric].std(ddof=0))
        summary[f"{metric}_min"] = float(seed_rows[metric].min())
        summary[f"{metric}_max"] = float(seed_rows[metric].max())
    pd.DataFrame([summary]).to_csv(final_root / "resnet18_seed_consistency_summary.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(8, 5))
    plt.bar(seed_rows["seed"].astype(str), seed_rows["accuracy"], color="#2563eb")
    plt.axhline(seed_rows["accuracy"].mean(), color="#dc2626", linestyle="--", label="mean accuracy")
    plt.ylim(0.95, 1.01)
    plt.xlabel("Seed")
    plt.ylabel("Accuracy")
    plt.title("Clean ResNet18 Seed Consistency")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(final_root / "figures" / "resnet18_seed_consistency.png", dpi=220)
    plt.close()


def generate_filtered_comparison_plots(summary_table: pd.DataFrame, final_root: Path) -> None:
    figure_root = final_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)

    plot_df = summary_table.copy()
    plot_df["label"] = plot_df["dataset_view"] + "\n" + plot_df["model_name"]

    plt.figure(figsize=(10, 5))
    plt.bar(plot_df["label"], plot_df["accuracy"], color=["#2563eb", "#2563eb", "#2563eb", "#059669", "#059669", "#059669"])
    plt.title("Model Accuracy Comparison")
    plt.xlabel("Dataset View and Model")
    plt.ylabel("Accuracy")
    plt.ylim(0.95, 1.01)
    plt.xticks(rotation=0)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(figure_root / "model_comparison_accuracy.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(plot_df["label"], plot_df["macro_f1"], color=["#1d4ed8", "#1d4ed8", "#1d4ed8", "#047857", "#047857", "#047857"])
    plt.title("Model Macro-F1 Comparison")
    plt.xlabel("Dataset View and Model")
    plt.ylabel("Macro F1")
    plt.ylim(0.95, 1.01)
    plt.xticks(rotation=0)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(figure_root / "model_comparison_macro_f1.png", dpi=220)
    plt.close()

    pivot = summary_table.pivot(index="model_name", columns="dataset_view", values="accuracy")
    pivot.plot(kind="bar", figsize=(8, 5), color=["#2563eb", "#059669"])
    plt.title("Official vs Clean Accuracy Comparison")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.ylim(0.95, 1.01)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(figure_root / "official_vs_clean_accuracy.png", dpi=220)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export lightweight final assets from training outputs.")
    parser.add_argument("--output-root", type=str, default=str(OUTPUT_ROOT))
    parser.add_argument("--final-root", type=str, default=str(FINAL_ROOT))
    args = parser.parse_args()

    output_root = Path(args.output_root)
    final_root = Path(args.final_root)
    final_root.mkdir(parents=True, exist_ok=True)

    leaderboard_path = output_root / "summary" / "leaderboard.csv"
    leaderboard = pd.read_csv(leaderboard_path)
    leaderboard = leaderboard[leaderboard["experiment_name"].isin(KEY_EXPERIMENTS)].copy()
    leaderboard.to_csv(final_root / "leaderboard_filtered.csv", index=False, encoding="utf-8-sig")

    test_rows = leaderboard[leaderboard["split"] == "test"].copy()
    test_rows["metric_rank"] = test_rows.groupby(["dataset_view", "model_name"])["accuracy"].rank(method="first", ascending=False)
    test_best = test_rows[test_rows["metric_rank"] == 1].drop(columns=["metric_rank"]).copy()
    test_best.to_csv(final_root / "final_test_metrics.csv", index=False, encoding="utf-8-sig")
    export_resnet18_seed_consistency(test_rows, final_root)

    summary_table = test_best[
        ["dataset_view", "model_name", "accuracy", "macro_precision", "macro_recall", "macro_f1", "checkpoint_path"]
    ].sort_values(["dataset_view", "model_name"])
    summary_table.to_csv(final_root / "report_table_main.csv", index=False, encoding="utf-8-sig")
    generate_filtered_comparison_plots(summary_table, final_root)

    best_params_path = output_root / "tuning" / "official_efficientnet_search" / "best_params.yaml"
    if best_params_path.exists():
        shutil.copy2(best_params_path, final_root / "official_efficientnet_best_params.yaml")
        with best_params_path.open("r", encoding="utf-8") as handle:
            params = yaml.safe_load(handle) or {}
        pd.DataFrame([params]).to_csv(final_root / "official_efficientnet_best_params.csv", index=False, encoding="utf-8-sig")

    trial_csv = output_root / "tuning" / "official_efficientnet_search" / "trials.csv"
    if trial_csv.exists():
        shutil.copy2(trial_csv, final_root / "official_efficientnet_trials.csv")

    copy_if_exists(output_root / "dataset_audit" / "official_summary.json", final_root / "dataset_audit" / "official_summary.json")
    copy_if_exists(output_root / "dataset_audit" / "clean_summary.json", final_root / "dataset_audit" / "clean_summary.json")
    copy_if_exists(output_root / "dataset_audit" / "figures" / "official_class_distribution.png", final_root / "figures" / "official_class_distribution.png")
    copy_if_exists(output_root / "dataset_audit" / "figures" / "clean_class_distribution.png", final_root / "figures" / "clean_class_distribution.png")
    copy_if_exists(output_root / "dataset_audit" / "figures" / "duplicate_overview.png", final_root / "figures" / "duplicate_overview.png")
    copy_if_exists(output_root / "tuning" / "official_efficientnet_search" / "tuning_history.png", final_root / "figures" / "tuning_history.png")
    copy_if_exists(output_root / "tuning" / "official_efficientnet_search" / "best_params.png", final_root / "figures" / "best_params.png")

    for experiment_name in KEY_EXPERIMENTS:
        copy_if_exists(
            output_root / "runs" / experiment_name / "figures" / "training_curves.png",
            final_root / "figures" / f"{experiment_name}_training_curves.png",
        )

    key_eval_source = output_root / "evaluations" / "official_efficientnet_tuned" / "test" / "figures"
    copy_if_exists(key_eval_source / "confusion_matrix.png", final_root / "figures" / "official_best_confusion_matrix.png")
    copy_if_exists(key_eval_source / "class_metrics.png", final_root / "figures" / "official_best_class_metrics.png")
    copy_if_exists(key_eval_source / "misclassified_examples.png", final_root / "figures" / "official_best_misclassified_examples.png")
    copy_if_exists(key_eval_source / "gradcam" / "gradcam_gallery.png", final_root / "figures" / "official_best_gradcam_gallery.png")

    clean_resnet_eval_source = output_root / "evaluations" / "clean_resnet18" / "test" / "figures"
    copy_if_exists(clean_resnet_eval_source / "confusion_matrix.png", final_root / "figures" / "clean_resnet18_confusion_matrix.png")
    copy_if_exists(clean_resnet_eval_source / "class_metrics.png", final_root / "figures" / "clean_resnet18_class_metrics.png")
    copy_if_exists(clean_resnet_eval_source / "misclassified_examples.png", final_root / "figures" / "clean_resnet18_misclassified_examples.png")
    copy_if_exists(clean_resnet_eval_source / "gradcam" / "gradcam_gallery.png", final_root / "figures" / "clean_resnet18_gradcam_gallery.png")

    print(f"Exported lightweight final assets to: {final_root}")


if __name__ == "__main__":
    main()
