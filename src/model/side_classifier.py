"""Predict whether a cached series images a left or right knee.

The DICOM side tag is blank on ~41% of series and the descriptions are
anonymised, so metadata cannot resolve them. Four of the twelve findings are
compartment-specific, and placing them requires knowing the side, so the
remainder is predicted from the imaging itself.

This trains on the 14,403 series whose side is known and applies to the rest.
Both halves come from the same cache, so it behaves identically at submission
time -- unlike report text, which exists only for training studies.

**Why the whole ordered stack rather than one representative slice.** For
coronal and axial acquisitions the left-right axis lies in-plane, and a single
image settles the side: the fibula sits laterally. For sagittal acquisitions
the left-right axis IS the slice axis, so individual slices look much the same
either way -- what differs is the direction of travel through the volume, which
runs medial-to-lateral for one side and lateral-to-medial for the other. A
model shown one slice cannot see that. Slices are therefore fed as ordered
channels, which preserves it.
"""

from typing import Optional, Sequence

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:  # the module is importable without torch for testing
    torch = None
    nn = object

from src import config

# Downsampled in-plane resolution. The side of a knee is a coarse, global cue,
# so half resolution is ample and quadruples throughput.
INPUT_SIZE = 112

SIDES = ["L", "R"]


def prepare_batch(volumes: np.ndarray, size: int = INPUT_SIZE) -> np.ndarray:
    """Cache volumes to model input: (N, slices, size, size) float32 in [0, 1].

    Slices become channels, keeping their order, so the network can use the
    medial-to-lateral direction that distinguishes sides on sagittal series.
    """
    import cv2

    if volumes.ndim == 3:
        volumes = volumes[None]

    count, depth, _, _ = volumes.shape
    output = np.empty((count, depth, size, size), dtype=np.float32)

    for i in range(count):
        for j in range(depth):
            output[i, j] = cv2.resize(volumes[i, j], (size, size),
                                      interpolation=cv2.INTER_AREA)

    return output / 255.0


def flip_input(batch: np.ndarray, plane_axis: Sequence[int]) -> np.ndarray:
    """Mirror a prepared batch along each item's medial-lateral axis.

    Used for augmentation: mirroring an input flips its label, which doubles
    the training set and forces the model to key on anatomy rather than on any
    incidental left/right imbalance in the source institutions.

    `plane_axis` gives the cache axis (0 for slices, 2 for columns) per item;
    in the prepared batch slices are dimension 1 and columns dimension 3.
    """
    output = np.empty_like(batch)
    for i, axis in enumerate(plane_axis):
        output[i] = np.flip(batch[i], axis=0 if axis == 0 else 2)
    return output


if torch is None:
    # Importable without torch so the array handling above can be tested on a
    # machine that never trains. The network itself needs the real thing.
    SideNet = None
else:
  class SideNet(nn.Module):
      """A small convolutional classifier over slice-channel input.

      Trained from scratch deliberately. It is a coarse two-class problem with
      14,000 labelled examples, so a pretrained backbone buys little -- and the
      competition's stance on pretrained weights is still unconfirmed, which this
      piece of the pipeline should not depend on.
      """

      def __init__(self, in_channels: int = None, width: int = 32):
          super().__init__()
          in_channels = in_channels or config.CACHE_SLICES

          def block(inp, out, stride=2):
              return nn.Sequential(
                  nn.Conv2d(inp, out, 3, stride=stride, padding=1, bias=False),
                  nn.BatchNorm2d(out),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(out, out, 3, padding=1, bias=False),
                  nn.BatchNorm2d(out),
                  nn.ReLU(inplace=True),
              )

          self.features = nn.Sequential(
              block(in_channels, width),        # 112 -> 56
              block(width, width * 2),          # 56 -> 28
              block(width * 2, width * 4),      # 28 -> 14
              block(width * 4, width * 4),      # 14 -> 7
          )
          self.head = nn.Sequential(
              nn.AdaptiveAvgPool2d(1),
              nn.Flatten(),
              nn.Dropout(0.2),
              nn.Linear(width * 4, 1),
          )

      def forward(self, x):
          return self.head(self.features(x)).squeeze(-1)


def study_level_vote(predictions, study_uids, confidences=None) -> dict:
    """Collapse per-series predictions to one side per study.

    A study is one knee, so its series must agree. Voting across them is
    strictly more reliable than any single series -- and the observed
    conflicting-tag rate is under 1%, so the assumption holds in practice.

    Returns {study_uid: "L" | "R"}.
    """
    if len(predictions) != len(study_uids):
        raise ValueError(
            f"predictions and study_uids differ in length: "
            f"{len(predictions)} vs {len(study_uids)}")
    if confidences is None:
        confidences = np.ones(len(predictions))
    elif len(confidences) != len(predictions):
        # zip would truncate silently, shifting weights onto the wrong series.
        raise ValueError(
            f"confidences length {len(confidences)} does not match "
            f"{len(predictions)} predictions")

    scores = {}
    for prediction, study, weight in zip(predictions, study_uids, confidences):
        scores.setdefault(study, []).append((float(prediction), float(weight)))

    result = {}
    for study, entries in scores.items():
        total = sum(weight for _, weight in entries)
        mean = (sum(p * w for p, w in entries) / total) if total else 0.5
        result[study] = "R" if mean >= 0.5 else "L"
    return result


def accuracy(predictions: np.ndarray, truth: np.ndarray) -> float:
    """Fraction correct, with predictions as probabilities of the right knee."""
    return float(((predictions >= 0.5).astype(int) == truth).mean())
