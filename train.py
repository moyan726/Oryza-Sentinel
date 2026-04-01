from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rice_leaf_disease.analysis import evaluate_checkpoint
from rice_leaf_disease.config import load_config
from rice_leaf_disease.engine import train_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a rice leaf disease classification experiment.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip post-training evaluation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, config_path = load_config(args.config)
    result = train_experiment(config, config_path)
    print(f"Training finished. Best checkpoint: {result.checkpoint_path}")
    print(f"Best validation accuracy: {result.best_val_accuracy:.4f}")
    if config.eval_after_train and not args.skip_eval:
        evaluation = evaluate_checkpoint(config, config_path, result.checkpoint_path, split=config.test_split)
        print("Evaluation metrics:", evaluation["metrics"])


if __name__ == "__main__":
    main()
