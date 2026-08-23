"""Study-level model: slices to series to study to twelve logits.

The backbone is deliberately the least interesting part. It is swappable, and
which one wins is settled by measurement rather than by argument -- at 4,349
studies with mined labels, the aggregation above it matters more.

    (B, S, D, H, W) uint8 volumes
      -> backbone over slices          (B*S*D, F)
      -> attention pool over slices    (B*S, F)
      -> FiLM on sequence metadata     (B*S, F)
      -> transformer over series       (B, F)      masked
      -> head                          (B, 12)

**Why FiLM on the sequence metadata.** Findings are sequence-specific: effusion
and Baker's cysts read off fluid-sensitive series, cartilage loss off
fat-suppressed ones, ACL tears off sagittals. `Fluid_Sensitive`,
`Fat_Suppression`, and the plane are given in both train and test, so pooling
every series through one undifferentiated path throws away free signal.

**Why masking is load-bearing.** Series counts run from one to a dozen. Padded
rows must be excluded from both the attention pool and the transformer, or a
short study is diluted by rows of zeros that the model reads as real series.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src import config
from src.model.dataset import META_COLUMNS

# Adjacent slices stacked into the three channels an ImageNet-pretrained
# backbone expects. Gives each slice a little through-plane context for free.
SLICE_CONTEXT = 3


def stack_2p5d(volumes: torch.Tensor) -> torch.Tensor:
    """(N, D, H, W) -> (N*D, 3, H, W) using each slice and its neighbours.

    Edge slices repeat, which is preferable to zero padding: a black plane at
    the edge of a volume is a feature the model would learn to key on.
    """
    n, depth, height, width = volumes.shape
    previous = torch.cat([volumes[:, :1], volumes[:, :-1]], dim=1)
    following = torch.cat([volumes[:, 1:], volumes[:, -1:]], dim=1)
    stacked = torch.stack([previous, volumes, following], dim=2)
    return stacked.reshape(n * depth, SLICE_CONTEXT, height, width)


class AttentionPool(nn.Module):
    """Masked attention pooling over a sequence of embeddings.

    Preferred to mean pooling because the informative slices are a minority --
    a meniscal tear appears on a few slices of a stack of 24, and averaging
    dilutes it by the rest.
    """

    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, N, F), mask: (B, N) with True for real entries.
        weights = self.score(x).squeeze(-1)
        if mask is not None:
            weights = weights.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(weights, dim=1).unsqueeze(-1)
        return (x * weights).sum(dim=1)


class FiLM(nn.Module):
    """Feature-wise modulation of series embeddings by sequence metadata."""

    def __init__(self, dim: int, meta_dim: int = len(META_COLUMNS)):
        super().__init__()
        self.to_scale_shift = nn.Sequential(
            nn.Linear(meta_dim, dim), nn.ReLU(inplace=True),
            nn.Linear(dim, dim * 2))

    def forward(self, x: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        scale, shift = self.to_scale_shift(meta).chunk(2, dim=-1)
        # Centre the scale on 1 so an untrained FiLM is close to the identity.
        return x * (1.0 + scale) + shift


class KneeNet(nn.Module):
    """Backbone plus the aggregation stack, emitting twelve study-level logits.

    `backbone` is any module mapping (N, 3, H, W) to (N, feature_dim). Passing
    a frozen encoder is supported and is the intended first configuration: with
    the trunk frozen, slice embeddings are computed once and the stack above
    trains in minutes rather than hours.
    """

    def __init__(self, backbone: Optional[nn.Module], feature_dim: int,
                 n_labels: int = None, series_layers: int = 2,
                 n_heads: int = 8, dropout: float = 0.2,
                 freeze_backbone: bool = False):
        super().__init__()
        # None is valid on the frozen path, where the trunk ran offline.
        self.backbone = backbone if backbone is not None else nn.Identity()
        self.feature_dim = feature_dim
        n_labels = n_labels or len(config.LABELS)

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        self.slice_pool = AttentionPool(feature_dim)
        self.film = FiLM(feature_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=n_heads,
            dim_feedforward=feature_dim * 2, dropout=dropout,
            batch_first=True, norm_first=True)
        self.series_encoder = nn.TransformerEncoder(layer, num_layers=series_layers)
        self.study_pool = AttentionPool(feature_dim)

        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, n_labels))

    def encode_series(self, volumes: torch.Tensor) -> torch.Tensor:
        """(N, D, H, W) uint8 or float -> (N, F) one embedding per series."""
        count, depth = volumes.shape[0], volumes.shape[1]
        if volumes.dtype == torch.uint8:
            volumes = volumes.float().div_(255.0)

        slices = stack_2p5d(volumes)
        features = self.backbone(slices)
        features = features.reshape(count, depth, self.feature_dim)
        return self.slice_pool(features)

    def aggregate(self, series: torch.Tensor, meta: torch.Tensor,
                  mask: torch.Tensor) -> torch.Tensor:
        """(B, S, F) series embeddings -> (B, n_labels).

        Everything above the backbone. Shared by the end-to-end path and the
        frozen path so the two cannot drift into different architectures.
        """
        series = self.film(series, meta)

        # The transformer takes True for positions to IGNORE, the inverse of
        # our mask.
        encoded = self.series_encoder(series, src_key_padding_mask=~mask)
        # A fully padded row emits NaN from the attention; zero it before
        # pooling so one short study cannot poison the batch.
        encoded = torch.nan_to_num(encoded)

        pooled = self.study_pool(encoded, mask=mask)
        return self.head(pooled)

    def forward_features(self, features: torch.Tensor, meta: torch.Tensor,
                         mask: torch.Tensor) -> torch.Tensor:
        """Frozen path: (B, S, D, F) precomputed slice embeddings.

        The backbone has already run, so this starts at the slice pooling.
        """
        batch, series, depth, dim = features.shape
        pooled = self.slice_pool(features.reshape(batch * series, depth, dim))
        return self.aggregate(pooled.reshape(batch, series, dim), meta, mask)

    def forward(self, volumes: torch.Tensor, meta: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        # volumes: (B, S, D, H, W), meta: (B, S, M), mask: (B, S)
        batch, series = volumes.shape[0], volumes.shape[1]

        flat = volumes.reshape(batch * series, *volumes.shape[2:])
        embeddings = self.encode_series(flat).reshape(batch, series, self.feature_dim)
        return self.aggregate(embeddings, meta, mask)


def weighted_bce(logits: torch.Tensor, targets: torch.Tensor,
                 weights: torch.Tensor) -> torch.Tensor:
    """BCE against soft targets, weighted per study and finding.

    Targets are probabilities rather than 0/1 -- they come from two extractors
    that disagree often enough that thresholding would teach the model to
    reproduce extraction errors. Weights carry how much each target is
    believed: full for agreement, less for conflict, least for silence.
    """
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    total = weights.sum()
    if total <= 0:
        return loss.mean() * 0.0
    return (loss * weights).sum() / total
