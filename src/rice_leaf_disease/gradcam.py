from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.forward_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, _module, _inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()

    def generate(self, inputs: torch.Tensor, target_index: int | None = None) -> np.ndarray:
        outputs = self.model(inputs)
        if target_index is None:
            target_index = int(outputs.argmax(dim=1).item())
        self.model.zero_grad(set_to_none=True)
        outputs[:, target_index].sum().backward()
        gradients = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (gradients * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam


def _resize_heatmap(heatmap: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray((heatmap * 255).astype(np.uint8))
    image = image.resize(size, resample=Image.BILINEAR)
    return np.asarray(image).astype(np.float32) / 255.0


def save_gradcam_overlay(
    image_path: str | Path,
    heatmap: np.ndarray,
    output_path: str | Path,
    alpha: float = 0.4,
) -> None:
    image_path = Path(image_path)
    output_path = Path(output_path)
    original = Image.open(image_path).convert("RGB")
    image_np = np.asarray(original).astype(np.float32) / 255.0
    heatmap_resized = _resize_heatmap(heatmap, original.size)

    plt.figure(figsize=(5, 5))
    plt.imshow(image_np)
    plt.imshow(heatmap_resized, cmap="jet", alpha=alpha)
    plt.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close()


def preprocess_single_image(image_path: str | Path, img_size: int) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize((int(round(img_size * 1.14)), int(round(img_size * 1.14)))),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)
