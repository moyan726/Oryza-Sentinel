from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rice_leaf_disease.config import DEFAULT_CLASS_NAMES
from rice_leaf_disease.data import (
    build_clean_manifest,
    plot_duplicate_overview,
    plot_manifest_distribution,
    save_manifests,
    scan_official_dataset,
    summarize_manifest,
)
from rice_leaf_disease.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build official and clean manifests for the rice leaf dataset.")
    parser.add_argument("--dataset-root", type=str, default="Rice Leaf Disease Images")
    parser.add_argument("--output-root", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = (PROJECT_ROOT / args.dataset_root).resolve()
    output_root = ensure_dir(PROJECT_ROOT / args.output_root)
    manifest_dir = ensure_dir(output_root / "manifests")
    figure_dir = ensure_dir(output_root / "dataset_audit" / "figures")
    summary_dir = ensure_dir(output_root / "dataset_audit")

    official_manifest = scan_official_dataset(dataset_root=dataset_root, class_names=DEFAULT_CLASS_NAMES)
    clean_manifest, duplicate_report = build_clean_manifest(official_manifest, seed=args.seed)
    manifest_paths = save_manifests(official_manifest, clean_manifest, duplicate_report, manifest_dir)

    official_summary = summarize_manifest(official_manifest)
    clean_summary = summarize_manifest(clean_manifest)
    with (summary_dir / "official_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(official_summary, handle, indent=2, ensure_ascii=False)
    with (summary_dir / "clean_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(clean_summary, handle, indent=2, ensure_ascii=False)

    plot_manifest_distribution(official_manifest, figure_dir / "official_class_distribution.png", "Official Split Distribution")
    plot_manifest_distribution(clean_manifest, figure_dir / "clean_class_distribution.png", "Clean Split Distribution")
    plot_duplicate_overview(official_manifest, figure_dir / "duplicate_overview.png")

    print("Saved manifests:")
    for name, path in manifest_paths.items():
        print(f"  {name}: {path}")
    print("Saved dataset audit summaries to:", summary_dir)


if __name__ == "__main__":
    main()
