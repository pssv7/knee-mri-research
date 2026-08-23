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

**What is stored per slice.** The CLS token concatenated with the mean of the
patch tokens: 768 dimensions from DINOv2-S. Storing the full 16x16 patch grid
would preserve more spatial detail, which matters for the small local findings
(a meniscal tear is a thin line tens of pixels across), but costs ~115 GB
against ~0.9 GB. The mean-patch half is a hedge: it carries some of what the
CLS token discards. If meniscus and fracture lag the regional findings, a
spatially pooled variant is the first thing to try.
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

# DINOv2-S: 384-dim tokens, and we keep CLS plus mean-patch.
TOKEN_DIM = 384
EMBED_DIM = TOKEN_DIM * 2

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


def embed_slices(encoder, slices) -> "torch.Tensor":
    """(N, 3, H, W) -> (N, 768): CLS concatenated with the mean patch token.

    Uses forward_features so the patch tokens are reachable. Taking the model's
    default pooled output would discard them, and the mean patch is the half
    that retains anything spatial.
    """
    with torch.no_grad():
        tokens = encoder.forward_features(normalize(slices))

        if tokens.ndim == 2:
            # A backbone that already pools; duplicate so the width is stable.
            return torch.cat([tokens, tokens], dim=-1)

        cls = tokens[:, 0]
        prefix = getattr(encoder, "num_prefix_tokens", 1)
        patches = tokens[:, prefix:].mean(dim=1)
        return torch.cat([cls, patches], dim=-1)


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
