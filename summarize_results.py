from __future__ import annotations

import argparse
import shutil
from pathlib import Path

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
    "clean_efficientnet_tuned",
]


def copy_if_exists(source: Path, target: Path) -> None:
    if source.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


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

    summary_table = test_best[
        ["dataset_view", "model_name", "accuracy", "macro_precision", "macro_recall", "macro_f1", "checkpoint_path"]
    ].sort_values(["dataset_view", "model_name"])
    summary_table.to_csv(final_root / "report_table_main.csv", index=False, encoding="utf-8-sig")

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
    copy_if_exists(output_root / "summary" / "figures" / "model_comparison_accuracy.png", final_root / "figures" / "model_comparison_accuracy.png")
    copy_if_exists(output_root / "summary" / "figures" / "model_comparison_macro_f1.png", final_root / "figures" / "model_comparison_macro_f1.png")
    copy_if_exists(output_root / "summary" / "figures" / "official_vs_clean_accuracy.png", final_root / "figures" / "official_vs_clean_accuracy.png")
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

    print(f"Exported lightweight final assets to: {final_root}")


if __name__ == "__main__":
    main()
