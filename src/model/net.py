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

from typing import Optional, Sequence

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


# The statistics DINOv2 was pretrained under. `embed.py` has always applied
# these on the frozen path; the end-to-end path fed raw [0,1] and ConvNeXt
# absorbed the shift into its first layer over twelve epochs. A pre-norm ViT
# is far less forgiving, so the same trunk was being fed a distribution it
# never saw during pretraining.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# NORMALISATION_BUFFERS -- why input_mean/input_std are persistent.
#
# They were registered with persistent=False on the argument that they are
# constants of the backbone family rather than learned state. That was true
# and it was still wrong: absent from the state dict, they have to be supplied
# again at every construction site, a site that forgets raises nothing, and a
# strict load restores every weight while leaving normalisation off. Two
# scoring sites forgot. The 24-slice DINOv2 run logged gold 0.864 from the
# in-memory model and 0.823 from the same weights reloaded -- an
# ImageNet-normalised trunk scored on raw [0,1].
#
# So they travel now, and `KneeNet._load_from_state_dict` handles both
# directions: a checkpoint that carries them imposes them on whatever the
# caller built, and one written before this change loads strictly without
# them. ConvNeXt is unaffected either way -- its defaults are None, so its
# state dict gains no keys and every existing reference number stands.

# How far below the head's learning rate a pretrained trunk should sit. The
# optimiser puts every parameter in one group at OneCycle's peak of
# `lr * max_lr_multiplier` = 3e-4, which a supervised ConvNeXt survives and a
# self-supervised ViT does not: it loses its pretrained features during warmup
# and then relearns them from 4,349 studies, which is the one regime ViTs are
# worst in. Measured symptom: gold 0.6 at epoch 4 where ConvNeXt reached
# 0.729, and *below* the same trunk frozen (0.770).
VIT_BACKBONE_LR_SCALE = 0.1


def backbone_defaults(name: str) -> dict:
    """Input normalisation and trunk learning-rate scale for a backbone family.

    Returns ConvNeXt's existing behaviour unchanged -- no normalisation, one
    learning rate -- so the 0.861 fold-1 reference stays comparable. That
    includes ConvNeXt-V2, whose FCMAE weights are a drop-in for V1 at the same
    `timm` interface, and DINOv3's ConvNeXt variants: keeping the convnet path
    identical is what makes a backbone swap a one-variable experiment.

    ViT-family trunks are the exception on both axes, and both were measured
    rather than argued -- see VIT_BACKBONE_LR_SCALE.
    """
    lowered = name.lower()
    is_vit = "vit" in lowered or ("dinov" in lowered and "convnext" not in lowered)
    if is_vit:
        return {"input_mean": IMAGENET_MEAN, "input_std": IMAGENET_STD,
                "backbone_lr_scale": VIT_BACKBONE_LR_SCALE}
    return {"input_mean": None, "input_std": None, "backbone_lr_scale": 1.0}


def build_backbone(name: str, pretrained: bool = True,
                   image_size: Optional[int] = None):
    """Create a timm trunk mapping (N, 3, H, W) -> (N, num_features).

    Every backbone this project runs is constructed through here so the
    training and submission notebooks cannot build different trunks from the
    same name -- notebook 10 hard-coded ConvNeXt for a while, which would have
    met a DINOv2 checkpoint with a shape error at submission time.

    ViT-family weights (DINOv2 among them) are published at a fixed input size
    and timm's patch embedding refuses anything else, so the cache's size is
    passed explicitly and the position embeddings are interpolated -- 224 is
    16x14, which divides cleanly. ConvNets accept any size and reject the
    argument, hence the fallback rather than an unconditional pass.
    """
    import timm

    image_size = config.CACHE_SIZE if image_size is None else image_size
    kwargs = dict(pretrained=pretrained, num_classes=0,
                  in_chans=SLICE_CONTEXT)
    try:
        return timm.create_model(name, img_size=image_size, **kwargs)
    except TypeError:
        return timm.create_model(name, **kwargs)


class AttentionPool(nn.Module):
    """Masked attention pooling over a sequence of embeddings.

    Preferred to mean pooling because the informative slices are a minority --
    a meniscal tear appears on a few slices of a stack of 24, and averaging
    dilutes it by the rest. This is the attention-based MIL pooling of Ilse et
    al., "Attention-based Deep Multiple Instance Learning" (ICML 2018,
    arXiv:1802.04712), equation 8: a trainable weighted average, permutation
    invariant, with the weights produced by a two-layer scorer.

    `gated=True` selects the same paper's equation 9, which replaces the single
    tanh branch with `tanh(Vh) * sigmoid(Uh)`. The stated motivation is that
    tanh is close to linear over most of its range, so it cannot express the
    non-linear "how relevant is this instance" function well; the sigmoid
    branch supplies a multiplicative gate that can suppress an instance
    outright. It is a strictly larger family -- a gate saturated at 1
    reproduces the ungated pool -- and it costs one extra `dim x hidden`
    matrix per pool.

    The two variants produce different state dict keys on purpose. A gated
    checkpoint loaded into an ungated model raises on the missing keys rather
    than scoring quietly through the wrong pooling, which is the failure mode
    every other geometry mismatch in this project has had.
    """

    def __init__(self, dim: int, hidden: int = 128, gated: bool = False):
        super().__init__()
        self.gated = gated
        if gated:
            self.attend = nn.Linear(dim, hidden)
            self.gate = nn.Linear(dim, hidden)
            self.weigh = nn.Linear(hidden, 1)
        else:
            self.score = nn.Sequential(
                nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def scores(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, F) -> (B, N) unnormalised attention logits."""
        if not self.gated:
            return self.score(x).squeeze(-1)
        return self.weigh(torch.tanh(self.attend(x))
                          * torch.sigmoid(self.gate(x))).squeeze(-1)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, N, F), mask: (B, N) with True for real entries.
        weights = self.scores(x)
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
                 input_dim: Optional[int] = None,
                 input_mean=None, input_std=None,
                 gated_pool: bool = False,
                 residual_pairs=None, residual_beta=None):
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

        # These now travel in the state dict, and `_load_from_state_dict`
        # below reconciles the two directions with older files. See
        # NORMALISATION_BUFFERS for why the earlier persistent=False cost a
        # measured run.
        #
        # Registered as None rather than assigned as a plain attribute: a
        # plain attribute of the same name makes a later register_buffer
        # raise "attribute already exists", and adopting a checkpoint's
        # statistics is exactly that late registration.
        if input_mean is not None:
            self.register_buffer(
                "input_mean", torch.tensor(input_mean).view(1, -1, 1, 1))
            self.register_buffer(
                "input_std", torch.tensor(input_std).view(1, -1, 1, 1))
        else:
            self.register_buffer("input_mean", None)
            self.register_buffer("input_std", None)

        self.slice_pool = AttentionPool(feature_dim, gated=gated_pool)
        self.film = FiLM(feature_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=n_heads,
            dim_feedforward=feature_dim * 2, dropout=dropout,
            batch_first=True, norm_first=True)
        self.series_encoder = nn.TransformerEncoder(layer, num_layers=series_layers)
        self.study_pool = AttentionPool(feature_dim, gated=gated_pool)

        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, n_labels))

        # Residual parameterisation: for each (child, parent) pair,
        #
        #     child_logit = beta * stopgrad(parent_logit) + g(features)
        #
        # with g the head's own child column. The plan's Stage 1: the
        # synovitis head reconstructs from the effusion head at R^2 0.55 with
        # a coefficient of +0.843, so effusion is most of what it computes.
        # Handing it that term for free, detached, leaves g nothing to learn
        # except the part effusion does not explain -- which is the part the
        # conditional metric measures.
        #
        # `beta` is calibrated on out-of-fold predictions, not derived from
        # r_true = 0.403: a rank correlation between two binary labels does
        # not map onto a logit coefficient. See
        # `src.model.conditional.calibrate_coupling`.
        #
        # Registered as None when unconfigured, exactly like the
        # normalisation buffers above and for the same reason -- a run that
        # does not use this writes an unchanged state dict, and a run that
        # does carries the coupling into the file so a bare
        # `KneeNet(backbone, dim)` at a scoring site reproduces it instead of
        # silently scoring an uncoupled model.
        if residual_pairs:
            pairs = torch.tensor(list(residual_pairs), dtype=torch.long)
            if pairs.ndim != 2 or pairs.shape[1] != 2:
                raise ValueError(
                    "residual_pairs must be (child, parent) index pairs, got "
                    f"shape {tuple(pairs.shape)}")
            if residual_beta is None:
                raise ValueError("residual_pairs given without residual_beta")
            values = (list(residual_beta) if hasattr(residual_beta, "__len__")
                      else [float(residual_beta)] * len(pairs))
            beta = torch.tensor(values, dtype=torch.float32)
            if len(beta) != len(pairs):
                raise ValueError(
                    f"residual_beta has {len(beta)} values for "
                    f"{len(pairs)} pairs")
            self.register_buffer("residual_pairs", pairs)
            self.register_buffer("residual_beta", beta)
        else:
            self.register_buffer("residual_pairs", None)
            self.register_buffer("residual_beta", None)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Reconcile the normalisation buffers with checkpoints either side.

        Three cases, and the middle one is the bug this exists for:

        - **The file carries the statistics and this model has none.** Adopt
          them. A bare `KneeNet(backbone, dim)` built at a scoring site now
          normalises the way the run that wrote the file did, instead of
          feeding an ImageNet-normalised trunk raw [0,1] and reading as a
          worse checkpoint.
        - **The file predates this change and this model has statistics.**
          Keep what the caller passed. Every checkpoint written before
          2026-08-28 is missing the two keys, and a strict load must not
          start failing on files this project already shipped.
        - **The file carries statistics and this model has different ones.**
          The file wins, for the same reason as the first case: the weights
          and the statistics they were trained under belong together, and the
          construction site is the thing that has been wrong before.

        Adopting is a `register_buffer` on a slot registered as None in
        `__init__`, which is why it is registered there rather than assigned.
        """
        for name in ("input_mean", "input_std",
                     "residual_pairs", "residual_beta"):
            key = prefix + name
            if key in state_dict:
                self.register_buffer(name, state_dict[key].clone())
            elif self._buffers.get(name) is not None:
                # Pre-fill so the strict load below sees no missing key.
                state_dict[key] = self._buffers[name]
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs)

    def encode_series(self, volumes: torch.Tensor) -> torch.Tensor:
        """(N, D, H, W) uint8 or float -> (N, F) one embedding per series."""
        count, depth = volumes.shape[0], volumes.shape[1]
        if volumes.dtype == torch.uint8:
            volumes = volumes.float().div_(255.0)

        slices = stack_2p5d(volumes)
        if self.input_mean is not None:
            slices = (slices - self.input_mean) / self.input_std
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

    def couple(self, logits: torch.Tensor) -> torch.Tensor:
        """Add the detached parent term to each configured child logit.

        A no-op unless the model was built (or loaded) with residual pairs, so
        every checkpoint written before this existed behaves identically.

        The parent term is detached: the point is to relieve the child of
        learning effusion, not to let the synovitis objective reach back and
        reshape the effusion head, which is the one head on this finding that
        is reliable.
        """
        if self.residual_pairs is None:
            return logits
        child = self.residual_pairs[:, 0]
        parent = self.residual_pairs[:, 1]
        addition = torch.zeros_like(logits)
        addition[:, child] = (self.residual_beta.to(logits.dtype)
                              * logits[:, parent].detach())
        return logits + addition

    def aggregate(self, series: torch.Tensor, meta: torch.Tensor,
                  mask: torch.Tensor) -> torch.Tensor:
        """(B, S, F) series embeddings -> (B, n_labels).

        Everything above the backbone. Shared by the end-to-end path and the
        frozen path so the two cannot drift into different architectures.
        """
        return self.couple(self.head(self.pool_study(series, meta, mask)))

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
        return self.couple(self.head(pooled)), pooled

    def forward(self, volumes: torch.Tensor, meta: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        # volumes: (B, S, D, H, W), meta: (B, S, M), mask: (B, S)
        logits, _ = self.forward_pooled(volumes, meta, mask)
        return logits


def weighted_bce(logits: torch.Tensor, targets: torch.Tensor,
                 weights: torch.Tensor, label_smoothing: float = 0.0,
                 pos_weight: Optional[torch.Tensor] = None,
                 symmetric_beta: float = 0.0) -> torch.Tensor:
    """BCE against soft targets, weighted per study and finding.

    Targets are probabilities rather than 0/1 -- they come from two extractors
    that disagree often enough that thresholding would teach the model to
    reproduce extraction errors. Weights carry how much each target is
    believed: full for agreement, less for conflict, least for silence.

    Three optional terms, all inert at their defaults so the reference runs
    stay bit-comparable. Each addresses something the plan names:

    `label_smoothing` pulls every target toward 0.5 by `eps`. The mined labels
    disagree with image-derived ground truth about 18% of the time, and a
    confident target that is wrong 18% of the time is what late training
    memorises. Note that soft targets already carry some of this effect --
    smoothing on top of them is a floor on the target's distance from 0.5, not
    the usual one-hot correction.

    `pos_weight` scales the positive term per finding. The metric is macro
    AUC, so a rare finding costs exactly as much as a common one; without this
    the loss is dominated by whatever is prevalent. Passed straight through to
    torch, which applies it to soft targets correctly.

    `symmetric_beta` adds the reverse cross-entropy term of Wang et al.,
    "Symmetric Cross Entropy for Robust Learning with Noisy Labels" (ICCV
    2019). RCE swaps the roles of prediction and target, which bounds the loss
    a wrong label can contribute: a confidently mislabelled study cannot drag
    the gradient the way plain CE lets it. Targets are clamped off 0 and 1
    because RCE takes their log; the clamp is what the paper's `A` constant
    does.
    """
    if label_smoothing > 0:
        targets = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing

    loss = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none")

    if symmetric_beta > 0:
        probs = torch.sigmoid(logits)
        clamped = targets.clamp(1e-4, 1.0 - 1e-4)
        reverse = -(probs * torch.log(clamped)
                    + (1.0 - probs) * torch.log1p(-clamped))
        loss = loss + symmetric_beta * reverse

    total = weights.sum()
    if total <= 0:
        return loss.mean() * 0.0
    return (loss * weights).sum() / total


def pairwise_rank_loss(scores: torch.Tensor, targets: torch.Tensor,
                       ref_scores: torch.Tensor, ref_targets: torch.Tensor,
                       weights: Optional[torch.Tensor] = None,
                       ref_weights: Optional[torch.Tensor] = None,
                       min_gap: float = 0.25,
                       pair_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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

    `pair_mask` drops pairs before weighting. Broadcastable to (B, Q, L), so
    it can restrict which findings are ranked, which pairs are comparable, or
    both. `stratified_rank_loss` below builds one; at None every pair counts
    and this is the loss the reference runs used, unchanged.
    """
    zero = scores.sum() * 0.0
    if ref_scores.numel() == 0:
        return zero

    gap = targets.unsqueeze(1) - ref_targets.unsqueeze(0)      # (B, Q, L)
    pair_weight = gap.abs() * (gap.abs() >= min_gap)
    if weights is not None and ref_weights is not None:
        pair_weight = (pair_weight * weights.unsqueeze(1)
                       * ref_weights.unsqueeze(0))
    if pair_mask is not None:
        pair_weight = pair_weight * pair_mask.to(pair_weight.dtype)

    total = pair_weight.sum()
    if total <= 0:
        return zero

    margin = scores.unsqueeze(1) - ref_scores.unsqueeze(0)     # (B, Q, L)
    loss = F.softplus(-torch.sign(gap) * margin)
    return (loss * pair_weight).sum() / total


def finding_mask(findings: Sequence[str], n_labels: int = None,
                 device=None) -> torch.Tensor:
    """(1, 1, L) mask selecting the findings a loss is allowed to touch.

    The stratified objective is for synovitis. Applying it to all twelve
    findings would trade away ranking on eleven that are already fine, on a
    stratification that means nothing for them.
    """
    n_labels = n_labels or len(config.LABELS)
    mask = torch.zeros(n_labels, dtype=torch.float32, device=device)
    for name in findings:
        mask[config.LABELS.index(name)] = 1.0
    return mask.view(1, 1, -1)


def caliper_pairs(strata: torch.Tensor, ref_strata: torch.Tensor,
                  caliper: float) -> torch.Tensor:
    """(B, Q, 1) mask of pairs whose stratifier scores are within `caliper`.

    Matching on a continuous score rather than splitting it at a threshold,
    because the mined effusion label carries its own extraction error and a
    hard split inherits it at the boundary. The predicted effusion score is
    continuous, scores AUC 0.888, and is available at every training step.
    """
    distance = (strata.unsqueeze(1) - ref_strata.unsqueeze(0)).abs()
    return (distance <= caliper).unsqueeze(-1)


def stratified_rank_loss(scores: torch.Tensor, targets: torch.Tensor,
                         ref_scores: torch.Tensor, ref_targets: torch.Tensor,
                         strata: torch.Tensor, ref_strata: torch.Tensor,
                         findings: Sequence[str], caliper: float = 0.15,
                         weights: Optional[torch.Tensor] = None,
                         ref_weights: Optional[torch.Tensor] = None,
                         min_gap: float = 0.25) -> torch.Tensor:
    """RankNet restricted to pairs from the same effusion stratum.

    Conditional AUC *is* within-stratum pair concordance, so this is not an
    analogy to the metric -- it is the metric, differentiably. Pairs the
    conditional number ignores are the ones effusion already orders correctly,
    and training on them is what produced a synovitis head that reconstructs
    from the effusion head at R^2 0.55.

    The known risk, stated in the plan and not solved here: stripping effusion
    removes the one reliable cue and leaves a harder task supervised by labels
    that are silent in 82.2% of reports against a true prevalence of 46.6%. If
    label quality binds, this loss makes the finding worse, and the conditional
    metric is what will show it.

    `strata` is the detached predicted effusion probability for the current
    batch and `ref_strata` the same for the queued reference studies.
    """
    mask = caliper_pairs(strata, ref_strata, caliper) * finding_mask(
        findings, scores.shape[-1], scores.device)
    return pairwise_rank_loss(scores, targets, ref_scores, ref_targets,
                              weights, ref_weights, min_gap=min_gap,
                              pair_mask=mask)


def coupling_penalty(scores: torch.Tensor, ref_scores: torch.Tensor,
                     child: int, parent: int, target: float) -> torch.Tensor:
    """Push corr(child logit, parent logit) toward `target`, not toward zero.

    The findings genuinely co-occur: on the gold set the odds ratio is 6.09
    (p = 0.003) and the label correlation is 0.403, against the model's 0.651.
    Driving the two heads to independence optimises toward a value the truth
    contradicts, so `target` is required and is meant to be the observed
    coupling rather than 0.

    Batches here are one study per rank, so the correlation is computed over
    the current batch concatenated with the detached reference queue. Returns
    a zero that still touches `scores` when there are too few studies to
    estimate a correlation, so the graph does not change shape between steps.
    """
    zero = scores.sum() * 0.0
    pool = (torch.cat([scores, ref_scores], dim=0)
            if ref_scores.numel() else scores)
    if pool.shape[0] < 8:
        return zero

    a = pool[:, child] - pool[:, child].mean()
    b = pool[:, parent] - pool[:, parent].mean()
    denominator = a.norm() * b.norm()
    if denominator <= 0:
        return zero
    return ((a * b).sum() / denominator - target) ** 2


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
