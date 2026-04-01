from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rice_leaf_disease.config import load_config
from rice_leaf_disease.gradcam import preprocess_single_image
from rice_leaf_disease.models import load_model_from_checkpoint
from rice_leaf_disease.utils import ensure_dir, resolve_torch_device, timestamp_string


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a single image with a trained checkpoint.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, _config_path = load_config(args.config)
    device = resolve_torch_device(config.device)
    model, checkpoint = load_model_from_checkpoint(args.checkpoint, device)
    inputs = preprocess_single_image(args.image, config.img_size).to(device)
    with torch.no_grad():
        logits = model(inputs)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    class_names = checkpoint["class_names"]
    predicted_index = int(probabilities.argmax())
    result = {
        "image": str(Path(args.image).resolve()),
        "predicted_class": class_names[predicted_index],
        "confidence": float(probabilities[predicted_index]),
        "probabilities": {class_names[index]: float(probabilities[index]) for index in range(len(class_names))},
    }

    output_dir = ensure_dir(PROJECT_ROOT / "outputs" / "predictions" / f"{Path(args.image).stem}_{timestamp_string()}")
    with (output_dir / "prediction.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)

    plt.figure(figsize=(8, 4))
    plt.bar(class_names, probabilities, color="#2563eb")
    plt.title("Single Image Prediction Probabilities")
    plt.xlabel("Class")
    plt.ylabel("Probability")
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(output_dir / "probabilities.png", dpi=220)
    plt.close()

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
