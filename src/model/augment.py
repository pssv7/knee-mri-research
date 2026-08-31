"""Training augmentation for study volumes.

The pipeline shipped with one augmentation -- a single intensity scale in
[0.9, 1.1] applied to a whole study -- while the plan (§3.4) calls for what the
RSNA winners used: geometric transforms, photometric jitter, noise, region
dropout, and slice-order permutation, all kept consistent across the slices of
a series.

**Series-consistent by construction.** Every transform draws its parameters
once per series and applies them to all of that series' slices. The alternative
-- fresh parameters per slice -- turns a coherent volume into a stack of
unrelated images, which is exactly the through-plane structure `stack_2p5d`
exists to give the backbone.

**What is deliberately absent, and why:**

- **Rotation, perspective, affine.** They need `scipy.ndimage` or `cv2` and run
  per-slice interpolation over ~80 slices per study; at 4,349 studies an epoch
  that is a dataloader bottleneck, not a free regulariser. Worth adding behind
  a measurement of loader throughput, not before.
- **Arbitrary slice permutation.** The plan lists it, but it would break the
  2.5D stacking: `stack_2p5d` gives each slice its two neighbours as channels,
  and a shuffled stack makes those neighbours meaningless. `slice_reverse`
  is the order change that keeps neighbour structure intact -- it reverses the
  through-plane direction rather than destroying it.
- **MixUp.** Not implementable at `BATCH_STUDIES = 1`: a within-batch
  permutation is the identity. Recorded in `docs/plan.md`.

**Laterality flip is a study-level transform, not a series-level one.** All of
a study's series image the same knee, so they must flip together, and the flip
swaps medial with lateral -- which means `config.LATERAL_LABEL_PAIRS` has to be
swapped in the targets at the same time. `config.py` has carried that constant
and that warning since before there was anything to use it. Getting this half
right (flipping pixels, not labels) corrupts four of the twelve findings
silently, which is why the dataset refused to flip at all until now.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from src import config
from src.model.dataset import (DEFAULT_MIRROR_AXIS, META_COLUMNS,
                               PLANE_MIRROR_AXIS)

# Index of `is_sagittal` in the per-series metadata row. The flip axis depends
# on the acquisition plane, and this is where that is already encoded -- no
# need to reach back into the index for `Anatomical_Plane`.
SAGITTAL_COLUMN = META_COLUMNS.index("is_sagittal")


def flip_volume(volume: np.ndarray, plane=None,
                sagittal: Optional[bool] = None) -> np.ndarray:
    """Mirror one series across the medial-lateral axis.

    The axis comes from `dataset.PLANE_MIRROR_AXIS`, the same table
    `mirror_volume` uses to canonicalise laterality -- deliberately, because a
    training flip and a test-time flip that disagreed about which axis is
    medial-lateral would be a mismatch that runs clean and scores worse. On a
    sagittal series medial-lateral is the slice axis; on coronal and axial it
    lies in-plane.

    Give it either `plane` (the string from the series metadata) or `sagittal`
    (the bit already encoded in a metadata row).
    """
    if sagittal is not None:
        axis = PLANE_MIRROR_AXIS["Sagittal"] if sagittal else DEFAULT_MIRROR_AXIS
    else:
        axis = PLANE_MIRROR_AXIS.get(str(plane), DEFAULT_MIRROR_AXIS)
    return np.flip(volume, axis=axis).copy()


def lateral_swap_order():
    """Index permutation that exchanges the medial/lateral finding pairs.

    A mirrored knee puts the medial compartment where the lateral one was, so
    any prediction or target read off a flipped image has to be permuted by
    this before it means anything. Used by the training augmentation and by
    the submission's flip TTA, from one definition, so the two cannot drift.
    """
    order = list(range(len(config.LABELS)))
    index_of = {name: i for i, name in enumerate(config.LABELS)}
    for medial, lateral in config.LATERAL_LABEL_PAIRS:
        a, b = index_of[medial], index_of[lateral]
        order[a], order[b] = order[b], order[a]
    return order


@dataclass(frozen=True)
class AugmentConfig:
    """Which augmentations run, and how hard.

    The defaults reproduce the incumbent behaviour exactly -- one intensity
    scale, nothing else -- so `AugmentConfig()` is the reference configuration
    and every measured result stays comparable. Turn things on one at a time.
    """

    intensity_scale: Optional[Tuple[float, float]] = (0.9, 1.1)
    intensity_shift: float = 0.0        # +/- this many 8-bit levels
    gamma: Optional[Tuple[float, float]] = None
    noise_std: float = 0.0              # gaussian sigma, in 8-bit levels
    cutout_boxes: int = 0
    cutout_fraction: float = 0.25       # box side, as a fraction of the image
    slice_reverse: float = 0.0          # probability, per series
    laterality_flip: float = 0.0        # probability, per study

    def __post_init__(self):
        if not 0.0 <= self.slice_reverse <= 1.0:
            raise ValueError("slice_reverse must be a probability")
        if not 0.0 <= self.laterality_flip <= 1.0:
            raise ValueError("laterality_flip must be a probability")
        if not 0.0 < self.cutout_fraction <= 1.0:
            raise ValueError("cutout_fraction must be in (0, 1]")


REFERENCE = AugmentConfig()


def augment_series(volume: np.ndarray, rng: np.random.Generator,
                   cfg: AugmentConfig = REFERENCE) -> np.ndarray:
    """(D, H, W) uint8 -> (D, H, W) uint8, one parameter draw for the series.

    Photometric work happens in float32 and is clipped back to uint8 once at
    the end, so a scale followed by a shift does not round twice.
    """
    working = volume.astype(np.float32)

    if cfg.intensity_scale is not None:
        low, high = cfg.intensity_scale
        working *= rng.uniform(low, high)

    if cfg.intensity_shift:
        working += rng.uniform(-cfg.intensity_shift, cfg.intensity_shift)

    if cfg.gamma is not None:
        low, high = cfg.gamma
        # Gamma is defined on [0, 1]; the volume is 8-bit, so scale in and out.
        working = np.clip(working, 0, 255)
        working = 255.0 * np.power(working / 255.0, rng.uniform(low, high))

    if cfg.noise_std > 0:
        # One noise field for the whole series rather than per slice: the
        # sensor noise being imitated is spatially structured, and per-slice
        # noise is trivially averaged away by the slice pooling.
        working = working + rng.normal(0.0, cfg.noise_std,
                                       size=working.shape[1:])[None]

    if cfg.cutout_boxes > 0:
        depth, height, width = working.shape
        box_h = max(1, int(round(height * cfg.cutout_fraction)))
        box_w = max(1, int(round(width * cfg.cutout_fraction)))
        for _ in range(cfg.cutout_boxes):
            top = int(rng.integers(0, max(1, height - box_h + 1)))
            left = int(rng.integers(0, max(1, width - box_w + 1)))
            # The SAME box on every slice: a box that moves through the stack
            # is a different occlusion per slice, which the slice pooling can
            # simply average out instead of learning around.
            working[:, top:top + box_h, left:left + box_w] = 0.0

    if cfg.slice_reverse and rng.random() < cfg.slice_reverse:
        working = working[::-1]

    return np.clip(working, 0, 255).astype(np.uint8)


def flip_laterality(item: dict, rng: np.random.Generator) -> dict:
    """Mirror every series of a study and swap the medial/lateral labels.

    Volumes reach here already canonicalised to the right knee, so this maps
    the study onto the left and back-labels it accordingly -- doubling the
    laterality distribution the model sees without inventing anatomy.

    The flip axis depends on the acquisition plane, exactly as
    `dataset.mirror_volume` decides it: on a sagittal series left-right runs
    *through* the plane, so mirroring reverses the slice order; on coronal and
    axial it lies in-plane, so it flips the columns. `is_sagittal` in the
    metadata row is what distinguishes them.

    Targets and weights are permuted by `config.LATERAL_LABEL_PAIRS`. Both, not
    just the targets: a weight says how much a finding's label is believed, and
    it belongs to the finding, so it has to travel with it.
    """
    volumes, meta = item["volumes"], item["meta"]
    flipped = np.empty_like(volumes)
    for i in range(volumes.shape[0]):
        flipped[i] = flip_volume(
            volumes[i], sagittal=bool(meta[i, SAGITTAL_COLUMN] > 0.5))
    item["volumes"] = flipped

    order = lateral_swap_order()
    for key in ("targets", "weights"):
        if item.get(key) is not None:
            item[key] = np.asarray(item[key])[order]

    # `is_right` in the metadata now describes the wrong side. Left as it is on
    # purpose: it is a *prediction* from the side classifier describing the
    # image as acquired, and the model reads it as conditioning rather than as
    # a label. Rewriting it here would tell the model the flip happened, which
    # is precisely the invariance the augmentation exists to teach.
    return item


def augment_study(item: dict, rng: np.random.Generator,
                  cfg: AugmentConfig = REFERENCE) -> dict:
    """Apply the study-level flip, then per-series photometric work."""
    if cfg.laterality_flip and rng.random() < cfg.laterality_flip:
        item = flip_laterality(item, rng)

    volumes = item["volumes"]
    if volumes.shape[0]:
        item["volumes"] = np.stack(
            [augment_series(volumes[i], rng, cfg)
             for i in range(volumes.shape[0])])
    return item
