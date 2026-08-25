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
                 freeze_backbone: bool = False,
                 input_dim: Optional[int] = None):
        super().__init__()
        # None is valid on the frozen path, where the trunk ran offline.
        self.backbone = backbone if backbone is not None else nn.Identity()
        self.feature_dim = feature_dim
        n_labels = n_labels or len(config.LABELS)

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        # Spatially pooled embeddings are wide (6,528 for a 4x4 grid), and
        # running the whole stack at that width would put ~85M parameters in
        # FiLM alone. A projection keeps the aggregation at a sane size.
        self.project = (nn.Linear(input_dim, feature_dim)
                        if input_dim and input_dim != feature_dim
                        else nn.Identity())

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

    def pool_study(self, series: torch.Tensor, meta: torch.Tensor,
                   mask: torch.Tensor) -> torch.Tensor:
        """(B, S, F) series embeddings -> (B, F) one embedding per study.

        The study representation the head reads. Exposed separately so
        auxiliary objectives (report-embedding regression) can supervise the
        representation without their parameters entering this module -- the
        exported state dict must stay loadable by a bare `KneeNet`.
        """
        series = self.film(series, meta)

        # The transformer takes True for positions to IGNORE, the inverse of
        # our mask.
        encoded = self.series_encoder(series, src_key_padding_mask=~mask)
        # A fully padded row emits NaN from the attention; zero it before
        # pooling so one short study cannot poison the batch.
        encoded = torch.nan_to_num(encoded)

        return self.study_pool(encoded, mask=mask)

    def aggregate(self, series: torch.Tensor, meta: torch.Tensor,
                  mask: torch.Tensor) -> torch.Tensor:
        """(B, S, F) series embeddings -> (B, n_labels).

        Everything above the backbone. Shared by the end-to-end path and the
        frozen path so the two cannot drift into different architectures.
        """
        return self.head(self.pool_study(series, meta, mask))

    def forward_features(self, features: torch.Tensor, meta: torch.Tensor,
                         mask: torch.Tensor) -> torch.Tensor:
        """Frozen path: (B, S, D, F) precomputed slice embeddings.

        The backbone has already run, so this starts at the slice pooling.
        """
        batch, series, depth, _ = features.shape
        projected = self.project(features.reshape(batch * series, depth, -1))
        pooled = self.slice_pool(projected)
        return self.aggregate(pooled.reshape(batch, series, self.feature_dim),
                              meta, mask)

    def forward_pooled(self, volumes: torch.Tensor, meta: torch.Tensor,
                       mask: torch.Tensor):
        """End-to-end path returning (logits, pooled study embedding).

        The training loop wants both: logits for the label losses, the pooled
        representation for auxiliary supervision. Computing them together
        avoids a second pass through the backbone.
        """
        batch, series = volumes.shape[0], volumes.shape[1]
        flat = volumes.reshape(batch * series, *volumes.shape[2:])
        embeddings = self.encode_series(flat).reshape(batch, series, self.feature_dim)
        pooled = self.pool_study(embeddings, meta, mask)
        return self.head(pooled), pooled

    def forward(self, volumes: torch.Tensor, meta: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        # volumes: (B, S, D, H, W), meta: (B, S, M), mask: (B, S)
        logits, _ = self.forward_pooled(volumes, meta, mask)
        return logits


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


def pairwise_rank_loss(scores: torch.Tensor, targets: torch.Tensor,
                       ref_scores: torch.Tensor, ref_targets: torch.Tensor,
                       weights: Optional[torch.Tensor] = None,
                       ref_weights: Optional[torch.Tensor] = None,
                       min_gap: float = 0.25) -> torch.Tensor:
    """Per-finding RankNet loss between a batch and a reference set.

    The metric is macro AUC -- a ranking statistic -- while BCE optimises
    calibration; this term optimises the ordering directly. Pairs are formed
    per finding between the current batch (`scores`/`targets`, (B, L), with
    gradient) and a reference set ((Q, L), typically detached scores from
    recent steps), because at one study per rank there are no pairs *within*
    a batch to rank.

    Only pairs whose soft targets differ by at least `min_gap` count: the
    targets are extractor probabilities, and ordering two studies the
    extractor itself barely distinguishes would train the model to reproduce
    extraction noise. Surviving pairs are weighted by the target gap and by
    both studies' label weights, so a confident 0.9-vs-0.1 pair outweighs a
    marginal one. Returns a zero that still touches `scores` when no pair
    qualifies, so the graph is identical either way.
    """
    zero = scores.sum() * 0.0
    if ref_scores.numel() == 0:
        return zero

    gap = targets.unsqueeze(1) - ref_targets.unsqueeze(0)      # (B, Q, L)
    pair_weight = gap.abs() * (gap.abs() >= min_gap)
    if weights is not None and ref_weights is not None:
        pair_weight = (pair_weight * weights.unsqueeze(1)
                       * ref_weights.unsqueeze(0))

    total = pair_weight.sum()
    if total <= 0:
        return zero

    margin = scores.unsqueeze(1) - ref_scores.unsqueeze(0)     # (B, Q, L)
    loss = F.softplus(-torch.sign(gap) * margin)
    return (loss * pair_weight).sum() / total


def load_knee_state(path, map_location="cpu") -> dict:
    """Read a `KneeNet` state dict from either checkpoint format.

    The end-to-end path writes two things over its life: a bare state dict
    from `torch.save(model.state_dict(), ...)`, and a Lightning checkpoint,
    which nests the same tensors under `state_dict` and prefixes every key
    with `model.` because `KneeNet` is an attribute of the wrapper. The
    submission must not care which it was handed -- requiring a manual
    conversion step is requiring a step that can be forgotten, and it was.

    Lightning checkpoints also carry optimiser state and callback objects,
    which `weights_only=True` refuses, so that is attempted first and relaxed
    only on failure. These are files this project wrote.
    """
    try:
        obj = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:       # torch too old to know the argument
        obj = torch.load(path, map_location=map_location)
    except Exception:       # noqa: BLE001 - a Lightning checkpoint
        obj = torch.load(path, map_location=map_location, weights_only=False)

    state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    if any(key.startswith("model.") for key in state):
        state = {key[len("model."):]: value for key, value in state.items()
                 if key.startswith("model.")}
    if not state:
        raise ValueError(f"no weights found in {path}")
    return state
