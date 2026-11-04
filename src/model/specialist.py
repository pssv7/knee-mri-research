"""Backbone and input settings used by the Experiment 11 specialists."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.model.dataset import sequence_flag


RADIMAGENET_RESNET50_SHA256 = (
    "08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734")
RADIMAGENET_PARAMETER_COUNT = 23_508_032


class RadImageNetResNet50(nn.Module):
    """The exact encoder layout used by the pinned public competition asset."""

    num_features = 2048

    def __init__(self):
        super().__init__()
        # Lazy import keeps target-building and CPU tests independent of the
        # optional torchvision binary.
        from torchvision.models import resnet50
        self.backbone = nn.Sequential(*list(resnet50(weights=None).children())[:-2])

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone(image).mean(dim=(2, 3))


class ProjectedRadImageNetResNet50(nn.Module):
    """RadImageNet trunk with a narrower study-model interface.

    The native 2,048-wide output would make the series transformer needlessly
    large.  The learned projection keeps the exact pretrained trunk and gives
    the existing aggregation stack its established 768-wide representation.
    """

    num_features = 768

    def __init__(self, encoder=None):
        super().__init__()
        self.encoder = encoder if encoder is not None else RadImageNetResNet50()
        self.project = nn.Linear(self.encoder.num_features, self.num_features)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.project(self.encoder(image))


def load_radimagenet_resnet50(path, *, expected_sha256=RADIMAGENET_RESNET50_SHA256,
                              frozen=False) -> RadImageNetResNet50:
    """Load and verify the public RadImageNet encoder without key conversion."""
    path = Path(path)
    with path.open("rb") as stream:
        observed = hashlib.file_digest(stream, "sha256").hexdigest()
    if observed != expected_sha256:
        raise ValueError(f"RadImageNet encoder hash mismatch: {observed}")
    model = RadImageNetResNet50()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True),
                          strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != RADIMAGENET_PARAMETER_COUNT:
        raise ValueError("RadImageNet encoder parameter-count drift")
    for parameter in model.parameters():
        parameter.requires_grad_(not frozen)
    return model


def load_projected_radimagenet_resnet50(path, *, frozen_encoder=False):
    encoder = load_radimagenet_resnet50(path, frozen=frozen_encoder)
    return ProjectedRadImageNetResNet50(encoder)


def radimagenet_input_defaults() -> dict:
    """Map the project's [0,1] images to the encoder's exact [-1,1] input."""
    return {
        "input_mean": (0.5, 0.5, 0.5),
        "input_std": (0.5, 0.5, 0.5),
        "backbone_lr_scale": 0.1,
    }


def select_specialist_series(rows: pd.DataFrame, maximum: int, arm: str) -> np.ndarray:
    """Choose deterministic series positions for a specialist.

    This stays outside the frozen general dataset module.  The caller freezes
    the selected manifest before constructing ``StudyIndex``, so historical
    training bundles retain their exact source hashes.
    """
    if arm not in ("synovitis", "pfoa"):
        raise ValueError(f"unknown specialist arm: {arm}")
    if maximum < 1:
        raise ValueError("maximum must be positive")
    if len(rows) <= maximum:
        return np.arange(len(rows))
    records = []
    for i, (_, row) in enumerate(rows.iterrows()):
        plane = str(row.get("Anatomical_Plane", ""))
        fluid = sequence_flag(row.get("Fluid_Sensitive")) > 0.5
        fat = sequence_flag(row.get("Fat_Suppression")) > 0.5
        if arm == "synovitis":
            rank = {
                ("Axial", True): 0, ("Sagittal", True): 1,
                ("Coronal", True): 2, ("Axial", False): 3,
                ("Sagittal", False): 4,
            }.get((plane, fluid), 5)
        else:
            rank = {"Axial": 0, "Sagittal": 1,
                    "Coronal": 2}.get(plane, 3)
        records.append((rank, not fat, i, plane))
    ordered = sorted(records)
    useful_planes = (("Axial", "Sagittal", "Coronal")
                     if arm == "synovitis" else ("Axial", "Sagittal"))
    chosen = []
    for plane in useful_planes:
        candidate = next((i for _, _, i, p in ordered
                          if p == plane and i not in chosen), None)
        if candidate is not None:
            chosen.append(candidate)
        if len(chosen) == maximum:
            return np.array(sorted(chosen))
    chosen.extend(i for _, _, i, _ in ordered if i not in chosen)
    return np.array(sorted(chosen[:maximum]))
