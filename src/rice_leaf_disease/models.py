from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision import models


TRANSFER_MODELS = {"resnet18", "efficientnet_b0"}


class CustomCNN(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        return self.classifier(features)


@dataclass
class BuiltModel:
    model: nn.Module
    pretrained_loaded: bool


def _load_with_fallback(factory, use_pretrained: bool, allow_fallback: bool) -> tuple[nn.Module, bool]:
    if not use_pretrained:
        return factory(None), False
    try:
        return factory("default"), True
    except Exception:
        if not allow_fallback:
            raise
        return factory(None), False


def build_model(
    model_name: str,
    num_classes: int,
    dropout: float,
    use_pretrained: bool = True,
    allow_pretrained_fallback: bool = True,
) -> BuiltModel:
    model_name = model_name.lower()
    if model_name == "custom_cnn":
        return BuiltModel(CustomCNN(num_classes=num_classes, dropout=dropout), pretrained_loaded=False)

    if model_name == "resnet18":
        def factory(mode: str | None):
            weights = models.ResNet18_Weights.DEFAULT if mode == "default" else None
            model = models.resnet18(weights=weights)
            in_features = model.fc.in_features
            model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
            return model

        model, pretrained_loaded = _load_with_fallback(factory, use_pretrained, allow_pretrained_fallback)
        return BuiltModel(model=model, pretrained_loaded=pretrained_loaded)

    if model_name == "efficientnet_b0":
        def factory(mode: str | None):
            weights = models.EfficientNet_B0_Weights.DEFAULT if mode == "default" else None
            model = models.efficientnet_b0(weights=weights)
            in_features = model.classifier[-1].in_features
            model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
            return model

        model, pretrained_loaded = _load_with_fallback(factory, use_pretrained, allow_pretrained_fallback)
        return BuiltModel(model=model, pretrained_loaded=pretrained_loaded)

    raise ValueError(f"Unsupported model_name: {model_name}")


def set_backbone_trainable(model: nn.Module, model_name: str, trainable: bool) -> None:
    model_name = model_name.lower()
    if model_name == "custom_cnn":
        for parameter in model.parameters():
            parameter.requires_grad = True
        return
    if model_name == "resnet18":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
        if trainable:
            for parameter in model.parameters():
                parameter.requires_grad = True
        return
    if model_name == "efficientnet_b0":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
        if trainable:
            for parameter in model.parameters():
                parameter.requires_grad = True
        return
    raise ValueError(f"Unsupported model_name: {model_name}")


def get_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "custom_cnn":
        return model.features[-2]
    if model_name == "resnet18":
        return model.layer4[-1]
    if model_name == "efficientnet_b0":
        return model.features[-1]
    raise ValueError(f"Unsupported model_name: {model_name}")


def load_model_from_checkpoint(checkpoint_path: str | Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    config = checkpoint["config"]
    built = build_model(
        model_name=config["model_name"],
        num_classes=len(checkpoint["class_names"]),
        dropout=float(config["dropout"]),
        use_pretrained=False,
        allow_pretrained_fallback=True,
    )
    model = built.model
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint
