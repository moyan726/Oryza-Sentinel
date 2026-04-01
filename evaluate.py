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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file.")
    parser.add_argument("--split", type=str, default=None, help="Optional split override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, config_path = load_config(args.config)
    result = evaluate_checkpoint(config, config_path, args.checkpoint, split=args.split)
    print("Saved evaluation artifacts to:", result["evaluation_dir"])
    print("Metrics:", result["metrics"])


if __name__ == "__main__":
    main()
