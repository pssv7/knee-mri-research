"""Frozen DINOv2 slice embeddings.

Runs the backbone once over every cached slice and stores the result, so the
aggregation stack trains on cached features in minutes rather than hours. The
same code runs at submission time, which is the point -- a divergence between
how features are computed for training and for inference is invisible locally
and fatal on the leaderboard.

**Why frozen, and why first.** The competition scores a second prize track on
`AUC + RuntimeSeconds / 32400`, so inference cost is a scored quantity rather
than a ceiling to stay under. A frozen trunk is cheap at submission and lets
the aggregation design -- which is where the modelling actually happens -- be
iterated dozens of times a day instead of three.

**What is stored per slice.** The CLS token concatenated with the patch tokens
reduced by `POOL_GRID`:

    grid 1     CLS + mean of all patches      768 dims    0.9 GB
    grid 2     CLS + 2x2 pooled grid        1,920 dims    2.3 GB
    grid 4     CLS + 4x4 pooled grid        6,528 dims    7.6 GB

Storing the full 16x16 grid would cost ~115 GB, which is why it is not offered.

The first model used grid 1, and its weakness appeared exactly where the mean
predicts one: regional findings (effusion, OA, Baker's) averaged 0.865 on gold
while focal ones (meniscus, ligaments, fracture) averaged 0.695. A meniscal
tear occupies two or three patches out of 256, so a mean over all of them
buries it in 250-odd patches of normal tissue.

Wide embeddings must be projected before the aggregation stack -- see
`KneeNet(input_dim=...)`. Running the stack at 6,528 wide puts ~769M parameters
in it against 8.2M with a projection.
"""

from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

from src import config

# The registers variant. Registers exist to absorb the high-norm artefact
# tokens that otherwise contaminate patch features -- precisely what gets
# pooled here.
MODEL_NAME = "vit_small_patch14_reg4_dinov2.lvd142m"

# timm's DINOv2 defaults to 518. The cache is 224, which divides cleanly by the
# patch size of 14 into a 16x16 grid, so the size is passed explicitly and the
# position embeddings are interpolated rather than silently mismatched.
INPUT_SIZE = config.CACHE_SIZE

# DINOv2-S token width.
TOKEN_DIM = 384

# How the 16x16 patch grid is reduced before storage.
#
#   1  mean over all patches       768 dims   0.9 GB over the dataset
#   4  4x4 average-pooled grid   6,528 dims   7.6 GB
#
# The mean is what the first model used, and its weakness showed up exactly
# where predicted: regional findings averaged 0.865 on gold while focal ones
# averaged 0.695. A meniscal tear occupies two or three patches out of 256, so
# averaging buries it. A 4x4 grid keeps enough spatial structure for the
# attention pooling to find it.
POOL_GRID = 1


def embed_dim(grid: int = None) -> int:
    """Stored width per slice: CLS plus the pooled patch tokens."""
    grid = POOL_GRID if grid is None else grid
    tokens = 1 + (grid * grid if grid > 1 else 1)
    return TOKEN_DIM * tokens


EMBED_DIM = embed_dim()

# ImageNet statistics, which DINOv2 was trained under.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def build_encoder(model_name: str = MODEL_NAME, pretrained: bool = True,
                  checkpoint_path: Optional[str] = None):
    """A frozen DINOv2 feature extractor in eval mode.

    `checkpoint_path` loads weights from disk, which the submission notebook
    needs: internet is disabled there, so timm cannot fetch anything.
    """
    import timm

    model = timm.create_model(
        model_name,
        pretrained=pretrained and checkpoint_path is None,
        img_size=INPUT_SIZE,
        num_classes=0,
    )

    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location="cpu")
        state = state.get("state_dict", state)
        model.load_state_dict(state, strict=False)

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def normalize(slices):
    """uint8 (N, 3, H, W) -> normalised float tensor."""
    x = slices.float().div_(255.0) if slices.dtype == torch.uint8 else slices.float()
    mean = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def pool_patches(patches: "torch.Tensor", grid: int) -> "torch.Tensor":
    """(N, P, C) patch tokens -> (N, grid*grid*C), spatially pooled.

    The patch sequence is reshaped back to its square grid before pooling, so
    neighbouring patches are averaged with their neighbours rather than with
    whatever happened to sit beside them in a flattened sequence.
    """
    import torch.nn.functional as F

    count, n_patches, channels = patches.shape
    side = int(round(n_patches ** 0.5))
    if side * side != n_patches:
        # Not a square grid; fall back to the mean rather than mis-shaping it.
        return patches.mean(dim=1)

    square = patches.transpose(1, 2).reshape(count, channels, side, side)
    pooled = F.adaptive_avg_pool2d(square, grid)
    return pooled.flatten(2).transpose(1, 2).reshape(count, grid * grid * channels)


def embed_slices(encoder, slices, grid: int = None) -> "torch.Tensor":
    """(N, 3, H, W) -> (N, embed_dim(grid)): CLS plus pooled patch tokens.

    Uses forward_features so the patch tokens are reachable at all; the model's
    default pooled output discards them.
    """
    grid = POOL_GRID if grid is None else grid

    with torch.no_grad():
        tokens = encoder.forward_features(normalize(slices))

        if tokens.ndim == 2:
            # A backbone that already pools; repeat so the width is predictable.
            repeats = embed_dim(grid) // tokens.shape[-1]
            return tokens.repeat(1, repeats)

        cls = tokens[:, 0]
        prefix = getattr(encoder, "num_prefix_tokens", 1)
        patches = tokens[:, prefix:]

        if grid > 1:
            return torch.cat([cls, pool_patches(patches, grid)], dim=-1)
        return torch.cat([cls, patches.mean(dim=1)], dim=-1)


class CachedEmbeddingBackbone(nn.Module if nn is not None else object):
    """Adapter so KneeNet can consume precomputed embeddings unchanged.

    KneeNet expects a module mapping slices to features. When features are
    already computed, the "backbone" is an identity that reshapes -- which
    keeps one model definition for both the cached and the end-to-end path.
    """

    def __init__(self, feature_dim: int = EMBED_DIM):
        super().__init__()
        self.feature_dim = feature_dim

    def forward(self, x):
        return x
