"""
Pretrained vision backbones for feature extraction (VGG16, ResNet18, ViT-B/16).
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
from torchvision import models
from torchvision.transforms import v2

# ---------------------------------------------------------------------
# Supported backbones (pretrained ImageNet weights)
# ---------------------------------------------------------------------
ALL_BACKBONES: List[str] = [
    "vgg16",
    "resnet18",
    "vit_b_16",
]


def _weights_for_backbone(backbone: str):
    """Return torchvision pretrained weights enum for a supported backbone."""
    b = backbone.lower()

    if b == "vgg16":
        return models.VGG16_Weights.IMAGENET1K_V1
    if b == "resnet18":
        return models.ResNet18_Weights.IMAGENET1K_V1
    if b == "vit_b_16":
        return models.ViT_B_16_Weights.IMAGENET1K_V1

    raise ValueError(f"Unknown backbone '{backbone}'. Supported: {', '.join(ALL_BACKBONES)}")


def weights_mean_std(backbone: str) -> Tuple[List[float], List[float]]:
    """Return pretrained normalization stats (mean, std) for a backbone."""
    weights = _weights_for_backbone(backbone)

    # Prefer stats from the exact pretrained preprocessing pipeline.
    preset = weights.transforms()
    mean = getattr(preset, "mean", None)
    std = getattr(preset, "std", None)

    # Backward-compatible fallback for older torchvision presets.
    if mean is None or std is None:
        mean = weights.meta.get("mean")
        std = weights.meta.get("std")

    if mean is None or std is None:
        raise RuntimeError(f"Cannot resolve mean/std for backbone '{backbone}'.")

    return list(mean), list(std)


def eval_transform_for_backbone(backbone: str, image_size: Tuple[int, int]):
    """Resize + ImageNet-style normalize matching the backbone's pretrained weights."""
    mean, std = weights_mean_std(backbone)
    return v2.Compose(
        [
            v2.Resize(image_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ]
    )


def build_backbone(name: str) -> nn.Module:
    """
    Frozen pretrained feature extractor (no classification head).
    - vgg16: 4096-d before final FC (we take penultimate FC block output)
    - resnet18: 512-d after global pool
    - vit_b_16: 768-d (classification head replaced with Identity)
    """
    name = name.lower()
    weights = _weights_for_backbone(name)

    if name == "vgg16":
        net = models.vgg16(weights=weights)
        extractor = nn.Sequential(
            net.features,
            net.avgpool,
            nn.Flatten(1),
            *list(net.classifier.children())[:-1],
        )

    elif name == "resnet18":
        net = models.resnet18(weights=weights)
        extractor = nn.Sequential(*list(net.children())[:-1], nn.Flatten(1))

    elif name == "vit_b_16":
        net = models.vit_b_16(weights=weights)
        net.heads = nn.Identity()
        extractor = net

    else:
        raise ValueError(f"Unknown backbone '{name}'. Supported: {', '.join(ALL_BACKBONES)}")

    for p in extractor.parameters():
        p.requires_grad = False
    extractor.eval()
    return extractor
