"""Lightning wrapper for the end-to-end fine-tuning run.

Kaggle offers a T4 x2 accelerator, and the training loop is GPU-bound at
~220 images/second, so a second device is close to a free halving of
wall-clock. Lightning is here for the distributed plumbing, not for the
abstraction: the semantics below are a faithful port of the hand-written loop
in notebook 08, and any difference in the resulting score is a bug rather
than a design choice.

**Everything in this module lives here rather than in a notebook cell.**
Lightning's `ddp_notebook` strategy forks, so children inherit memory and the
module is not pickled on that path -- but dataloader workers and the spawn
start method both do pickle, and a class defined in a notebook cell belongs to
`__main__` and does not survive that. Keeping these definitions importable as
`src.model.lightning` makes the module correct under either start method; the
bootstrap writes the files to disk before importing them. The same rule put
the cache's process-pool worker in `src/data/cache.py`.

A related consequence of fork, and the reason the notebook is careful about
it: **CUDA must not be initialised before `Trainer.fit`**. A fork inheriting a
live CUDA context fails in the child. `torch.cuda.get_device_name` and
`get_device_properties` both call `_lazy_init`; `device_count` and
`is_available` do not.

Three DDP hazards are handled explicitly, because each one runs clean and
silently scores worse:

**Memory-mapped shards are never serialised.** The cache is ~29 GB across
three memmaps. Holding them as attributes would carry the array bytes into any
process the dataset is pickled into. They are opened lazily per process and
dropped from `__getstate__`, which also means each dataloader worker maps the
files itself rather than sharing a handle across a fork.

**AUC is computed over gathered predictions.** It is a ranking metric, so
averaging two per-rank AUCs is not the AUC of the whole set. Predictions and
targets are all-gathered before scoring. The distributed sampler pads the
final batch by repeating samples, so at most one study per rank is counted
twice out of ~870 -- below the resolution of the measurement.

**The effective batch is held constant.** Each rank receives `batch_size`
studies, so the effective batch is `batch_size x devices x accumulation`.
Halving accumulation when moving from one device to two keeps the optimiser
seeing the same eight studies per step, and keeps the OneCycleLR step count
unchanged.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from src import config
from src.model import augment as augment_mod
from src.model import averaging
from src.model import conditional
from src.model import dataset as knee_dataset
from src.model import net

try:  # Lightning 2.x ships under both names depending on the install.
    import lightning.pytorch as pl
except ImportError:  # pragma: no cover - exercised only on Kaggle/Colab
    import pytorch_lightning as pl


def auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """ROC AUC via the rank statistic, with ties averaged.

    Duplicated from `labels.calibrate` rather than imported: only the model
    package is materialised into the training notebook, and a metric that
    disagrees with the one the labels were scored by would be worse than the
    duplication.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    positives = y_true == 1
    n_pos, n_neg = int(positives.sum()), int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    unique, inverse, counts = np.unique(scores, return_inverse=True,
                                        return_counts=True)
    sums = np.zeros(len(unique))
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]

    return (ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def macro_auc(truth: np.ndarray, probs: np.ndarray) -> float:
    """Mean per-finding AUC, ignoring findings with no positives present."""
    scores = [auc(truth[:, i], probs[:, i]) for i in range(truth.shape[1])]
    return float(np.nanmean(scores))


class StudyVolumeDataset(Dataset):
    """Study volumes with an even slice subsample and intensity jitter.

    Slices are subsampled evenly rather than at random so a validation score
    is reproducible. Horizontal flips are deliberately not used: after
    canonicalisation a flip re-mirrors the knee and swaps medial with lateral,
    silently corrupting four of the twelve labels.
    """

    def __init__(self, study_index, studies: Sequence[str],
                 cache_paths: Sequence[str], slices_per_series: int,
                 augment=False):
        self.index = study_index
        self.studies = list(studies)
        self.cache_paths = list(cache_paths)
        self.slices_per_series = slices_per_series
        # `augment` is either a bool -- the legacy single intensity scale that
        # every measured result was trained under -- or an AugmentConfig. The
        # bool path is kept byte-for-byte rather than expressed as a config so
        # that reproducing an old run does not depend on the two being
        # equivalent. See src/model/augment.py.
        self.augment = augment
        self.keep = np.linspace(0, config.CACHE_SLICES - 1,
                                slices_per_series).round().astype(int)
        self._shards: Optional[Dict[str, np.ndarray]] = None

    @property
    def shards(self) -> Dict[str, np.ndarray]:
        """Open the memmaps on first use, once per process."""
        if self._shards is None:
            self._shards = {path: np.load(path, mmap_mode="r")
                            for path in self.cache_paths}
        return self._shards

    def __getstate__(self):
        # Never serialise the memmaps into a spawned process or a dataloader
        # worker; each one reopens them from the paths instead.
        state = self.__dict__.copy()
        state["_shards"] = None
        return state

    def __len__(self) -> int:
        return len(self.studies)

    def __getitem__(self, i: int):
        item = knee_dataset.load_study(self.index, self.studies[i], self.shards)

        available = item["volumes"].shape[1]
        if available > self.slices_per_series:
            # A thinned cache is already at the target count; only subsample a
            # full one, or the second cut would drop slices twice.
            item["volumes"] = item["volumes"][:, self.keep]
        elif available < self.slices_per_series:
            # The attention pool masks over the slice axis, so a short volume
            # trains happily on fewer slices than configured and says nothing.
            raise ValueError(
                f"cache supplies {available} slices, slices_per_series is "
                f"{self.slices_per_series} -- attach the full 24-slice "
                "shards, or lower the setting")

        if isinstance(self.augment, augment_mod.AugmentConfig):
            # Seeded from the study index and the epoch-independent worker RNG
            # so two workers never draw the same stream, and a re-run with the
            # same seed is reproducible.
            item = augment_mod.augment_study(
                item, np.random.default_rng(np.random.randint(0, 2**31)),
                self.augment)
        elif self.augment:
            # Legacy path: one intensity scale for the whole study. Every
            # result on record was trained through exactly this.
            scale = np.random.uniform(0.9, 1.1)
            item["volumes"] = np.clip(
                item["volumes"].astype(np.float32) * scale,
                0, 255).astype(np.uint8)
        return item


class KneeLitModule(pl.LightningModule):
    """The fine-tuning step, loss, schedule, and epoch-level scoring.

    Two validation dataloaders are expected, and which is which matters:
    index 0 is the validation fold scored against mined labels binarised at
    0.5, and it is the only thing allowed to select a checkpoint. Index 1 is
    the gold 58, logged every epoch because it is nearly free and shows
    whether the run is converging -- and never monitored. Selecting on gold
    turns the project's one honest measuring stick into a model-selection set;
    an earlier version of this loop did exactly that and inflated its own
    reported figure by an estimated 0.01-0.03.
    """

    def __init__(self, backbone: torch.nn.Module, feature_dim: int,
                 lr: float = 1e-4, weight_decay: float = 1e-4,
                 total_steps: int = 1000, max_lr_multiplier: float = 3.0,
                 report_dim: Optional[int] = None,
                 lambda_report: float = 0.0, lambda_rank: float = 0.0,
                 rank_queue_size: int = 256, rank_min_gap: float = 0.25,
                 aux_mode: str = "fixed", aux_gate: str = "binary",
                 backbone_lr_scale: float = 1.0,
                 input_mean=None, input_std=None, gated_pool: bool = False,
                 schedule: str = "onecycle",
                 warmup_steps: Optional[int] = None,
                 label_smoothing: float = 0.0, symmetric_beta: float = 0.0,
                 pos_weight: Optional[Sequence[float]] = None,
                 lambda_stratified_rank: float = 0.0,
                 stratified_findings: Sequence[str] = ("Synovitis",),
                 stratify_by: str = "Effusion",
                 stratified_caliper: float = 0.15,
                 lambda_coupling: float = 0.0,
                 coupling_target: Optional[float] = None,
                 residual_beta: Optional[float] = None):
        super().__init__()
        if aux_mode not in ("fixed", "cosine"):
            raise ValueError(f"aux_mode must be fixed|cosine, got {aux_mode!r}")
        if aux_gate not in ("binary", "weighted"):
            raise ValueError(f"aux_gate must be binary|weighted, got {aux_gate!r}")
        if schedule not in ("onecycle", "cosine"):
            raise ValueError(
                f"schedule must be onecycle|cosine, got {schedule!r}")
        if warmup_steps is not None and not 0 <= warmup_steps < total_steps:
            raise ValueError(
                f"warmup_steps {warmup_steps} must be under total_steps "
                f"{total_steps}")
        if lambda_coupling > 0 and coupling_target is None:
            raise ValueError(
                "lambda_coupling needs an explicit coupling_target. The "
                "findings genuinely co-occur (gold OR 6.09, p = 0.003); a "
                "penalty toward independence optimises to a value the truth "
                "contradicts. Pass the observed coupling.")
        for name in tuple(stratified_findings) + (stratify_by,):
            if name not in config.LABELS:
                raise ValueError(f"unknown finding {name!r}")

        self.save_hyperparameters(ignore=["backbone"])
        # Stage 1's residual parameterisation, built into the model so it
        # travels in the checkpoint rather than having to be reconstructed at
        # every scoring site. See KneeNet.couple.
        residual_pairs = None
        if residual_beta is not None:
            residual_pairs = [(config.LABELS.index(name),
                               config.LABELS.index(stratify_by))
                              for name in stratified_findings]
        self.model = net.KneeNet(backbone, feature_dim,
                                 input_mean=input_mean, input_std=input_std,
                                 gated_pool=gated_pool,
                                 residual_pairs=residual_pairs,
                                 residual_beta=residual_beta)
        self.backbone_lr_scale = backbone_lr_scale
        self.lr = lr
        self.weight_decay = weight_decay
        self.total_steps = total_steps
        self.max_lr_multiplier = max_lr_multiplier
        self.lambda_report = lambda_report
        self.lambda_rank = lambda_rank
        self.rank_queue_size = rank_queue_size
        self.rank_min_gap = rank_min_gap
        self.aux_mode = aux_mode
        self.aux_gate = aux_gate
        self.schedule = schedule
        self.warmup_steps = warmup_steps
        self.label_smoothing = label_smoothing
        self.symmetric_beta = symmetric_beta
        self.lambda_stratified_rank = lambda_stratified_rank
        self.stratified_findings = tuple(stratified_findings)
        self.stratify_by = stratify_by
        self.stratified_index = config.LABELS.index(stratify_by)
        self.stratified_caliper = stratified_caliper
        self.lambda_coupling = lambda_coupling
        self.coupling_target = coupling_target

        # A buffer rather than a plain tensor: it has to follow the module to
        # the GPU, and it is not learned. Registered as None when unused so
        # the state dict is unchanged for every run that does not set it --
        # and it lives on the wrapper, not on `KneeNet`, so it never reaches
        # the exported checkpoint the submission loads.
        if pos_weight is not None:
            self.register_buffer(
                "pos_weight", torch.tensor(list(pos_weight), dtype=torch.float32))
        else:
            self.register_buffer("pos_weight", None)

        # The report head is an attribute of the *wrapper*, deliberately: the
        # exported state dict keeps only `model.`-prefixed keys, so auxiliary
        # parameters vanish at export and notebook 10 keeps loading a bare
        # KneeNet. Created only when actually trained against -- under DDP a
        # parameter that never receives gradient is an error, not a no-op.
        if report_dim is not None and lambda_report > 0:
            self.report_head = torch.nn.Sequential(
                torch.nn.LayerNorm(feature_dim),
                torch.nn.Linear(feature_dim, report_dim))
        else:
            self.report_head = None

        # Cross-batch queue for the ranking loss. At one study per rank there
        # are no pairs inside a batch, so current logits are ranked against
        # detached logits from recent steps (XBM-style). Slightly stale scores
        # are the price; per-rank and reset each epoch.
        self._rank_queue: List[Dict[str, torch.Tensor]] = []

        # Not `_buffers`: nn.Module reserves that name for registered
        # buffers, and shadowing it corrupts state_dict and .to(device).
        self._val_outputs: Dict[int, List[Dict[str, torch.Tensor]]] = {0: [], 1: []}

    def forward(self, volumes, meta, mask):
        return self.model(volumes, meta, mask)

    def on_train_epoch_start(self):
        self._rank_queue = []

    def _report_loss(self, pooled: torch.Tensor, batch) -> torch.Tensor:
        """Cosine distance between the projected study embedding and the
        report's sentence embedding, over studies that have one.

        The head must run even when the whole batch lacks reports: DDP
        requires every parameter to join every backward pass.
        """
        predicted = self.report_head(pooled.float())
        if "report" not in batch:
            return predicted.sum() * 0.0
        predicted = torch.nn.functional.normalize(predicted, dim=-1)
        target = torch.nn.functional.normalize(batch["report"].float(), dim=-1)
        distance = 1.0 - (predicted * target).sum(dim=-1)
        mask = batch["report_mask"].float()
        return (distance * mask).sum() / mask.sum().clamp(min=1.0)

    def _rank_reference(self):
        """The queued detached scores both ranking losses rank against."""
        if not self._rank_queue:
            return None
        return {key: torch.cat([q[key] for q in self._rank_queue])
                for key in ("scores", "targets", "weights", "strata")}

    def _rank_push(self, logits: torch.Tensor, batch) -> None:
        """Add this step's studies to the queue, once per training step.

        Pushed after every ranking term has been computed, so a study never
        ranks against itself. `strata` is the predicted probability of the
        stratifying finding -- effusion -- carried because the stratified loss
        needs the reference studies' effusion scores to decide which pairs are
        comparable, and recomputing them would mean a second forward pass over
        studies that have already left the batch.
        """
        self._rank_queue.append({
            "scores": logits.detach().float(),
            "targets": batch["targets"].detach().float(),
            "weights": batch["weights"].detach().float(),
            "strata": torch.sigmoid(
                logits[:, self.stratified_index].detach().float()),
        })
        while sum(len(q["scores"]) for q in self._rank_queue) > self.rank_queue_size:
            self._rank_queue.pop(0)

    def _rank_loss(self, logits: torch.Tensor, batch) -> torch.Tensor:
        scores = logits.float()
        reference = self._rank_reference()
        if reference is None:
            return scores.sum() * 0.0
        return net.pairwise_rank_loss(
            scores, batch["targets"].float(),
            reference["scores"], reference["targets"],
            batch["weights"].float(), reference["weights"],
            min_gap=self.rank_min_gap)

    def _stratified_loss(self, logits: torch.Tensor, batch) -> torch.Tensor:
        """Stage 1: rank only within an effusion stratum.

        Conditional AUC is within-stratum pair concordance, so this optimises
        the metric the finding is judged on rather than a proxy for it. Pairs
        are matched on the model's own predicted effusion score, detached: the
        mined effusion label has its own error, and a hard split would inherit
        it at the boundary.
        """
        scores = logits.float()
        reference = self._rank_reference()
        if reference is None:
            return scores.sum() * 0.0
        strata = torch.sigmoid(scores[:, self.stratified_index].detach())
        return net.stratified_rank_loss(
            scores, batch["targets"].float(),
            reference["scores"], reference["targets"],
            strata, reference["strata"],
            self.stratified_findings, self.stratified_caliper,
            batch["weights"].float(), reference["weights"],
            min_gap=self.rank_min_gap)

    def _coupling_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Hold the head-to-head coupling at the truth's value, not at zero."""
        reference = self._rank_reference()
        queued = (reference["scores"] if reference is not None
                  else logits.new_zeros((0, logits.shape[1])))
        total = logits.sum() * 0.0
        for name in self.stratified_findings:
            total = total + net.coupling_penalty(
                logits.float(), queued, config.LABELS.index(name),
                self.stratified_index, self.coupling_target)
        return total

    def _aux_cosine(self, main_loss: torch.Tensor, aux_loss: torch.Tensor,
                    pooled: torch.Tensor) -> torch.Tensor:
        """cos between the two losses' gradients at the shared bottleneck.

        Du et al., "Adapting Auxiliary Losses Using Gradient Similarity"
        (arXiv:1812.02224), Algorithm 1: apply the auxiliary gradient to the
        shared parameters only while it points into the same half-space as
        the main gradient, which guarantees convergence to the main task's
        critical points and blocks negative transfer.

        The paper measures that cosine over the shared parameters theta.
        Here every shared parameter is reached *only* through `pooled` --
        `forward_pooled` returns `head(pooled), pooled`, and both losses hang
        off that one tensor -- so the shared gradients are J^T a and J^T b for
        the same Jacobian J = d pooled/d theta. Taking the cosine at the
        bottleneck asks the same question ("does the auxiliary want to move
        the shared representation the way the main task does?") for the price
        of two backward passes through the two small heads instead of two
        through ConvNeXt. It is a proxy, not an identity: J^T can rotate the
        pair, so agreement here is not arithmetically agreement at theta.
        The sign is what Algorithm 1 consumes, and the paper itself keeps
        only the sign because minibatch cosines are too noisy to trust
        beyond it.

        retain_graph is required: the real backward still has to run
        afterwards over the same graph.

        Both losses are scaled by a constant first. Training runs in
        `16-mixed`, and these probe gradients bypass Lightning's GradScaler,
        so an unscaled fp16 backward can flush small values to zero -- which
        reads as cosine 0, i.e. "aligned", i.e. silently back to a fixed
        lambda. Cosine is scale-invariant, so scaling both sides costs
        nothing and keeps the values off the floor.
        """
        scale = 1024.0
        grad_main, = torch.autograd.grad(main_loss * scale, pooled,
                                         retain_graph=True)
        grad_aux, = torch.autograd.grad(aux_loss * scale, pooled,
                                        retain_graph=True)
        cosine = torch.nn.functional.cosine_similarity(
            grad_main.flatten().float(), grad_aux.flatten().float(), dim=0)
        # Ranks must agree, or the "binary" gate is silently a soft one: DDP
        # averages the two ranks' gradients, so a per-rank decision would
        # apply the auxiliary at half strength whenever they disagreed.
        # torch.distributed rather than self.all_gather, which requires an
        # attached Trainer and would make this untestable in isolation.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            cosine = cosine.clone()
            torch.distributed.all_reduce(cosine, op=torch.distributed.ReduceOp.SUM)
            cosine = cosine / torch.distributed.get_world_size()
        return cosine

    def _aux_weight(self, cosine: torch.Tensor) -> torch.Tensor:
        """Algorithm 1 (binary) or Algorithm 2 (cosine-weighted)."""
        if self.aux_gate == "weighted":
            return cosine.clamp(min=0.0)
        return (cosine >= 0).to(cosine.dtype)

    def training_step(self, batch, batch_idx):
        logits, pooled = self.model.forward_pooled(
            batch["volumes"], batch["meta"], batch["mask"])
        loss = net.weighted_bce(
            logits, batch["targets"], batch["weights"],
            label_smoothing=self.label_smoothing,
            pos_weight=self.pos_weight,
            symmetric_beta=self.symmetric_beta)
        self.log("bce_loss", loss, on_step=False, on_epoch=True,
                 sync_dist=True, batch_size=len(logits))
        main_loss = loss

        if self.report_head is not None:
            report_loss = self._report_loss(pooled, batch)
            weight = self.lambda_report
            if self.aux_mode == "cosine":
                cosine = self._aux_cosine(main_loss, report_loss, pooled)
                weight = weight * self._aux_weight(cosine)
                # The diagnostic worth having even when the gate never fires:
                # a report loss that is mostly anti-aligned explains a fixed
                # lambda hurting the leaderboard while helping validation.
                self.log("report_cosine", cosine, on_step=False, on_epoch=True,
                         sync_dist=True, batch_size=len(logits))
                self.log("report_gate", (cosine >= 0).float(), on_step=False,
                         on_epoch=True, sync_dist=True, batch_size=len(logits))
            loss = loss + weight * report_loss
            self.log("report_loss", report_loss, on_step=False, on_epoch=True,
                     sync_dist=True, batch_size=len(logits))

        if self.lambda_rank > 0:
            rank_loss = self._rank_loss(logits, batch)
            weight = self.lambda_rank
            if self.aux_mode == "cosine":
                cosine = self._aux_cosine(main_loss, rank_loss, pooled)
                weight = weight * self._aux_weight(cosine)
                self.log("rank_cosine", cosine, on_step=False, on_epoch=True,
                         sync_dist=True, batch_size=len(logits))
            loss = loss + weight * rank_loss
            self.log("rank_loss", rank_loss, on_step=False, on_epoch=True,
                     sync_dist=True, batch_size=len(logits))

        if self.lambda_stratified_rank > 0:
            stratified_loss = self._stratified_loss(logits, batch)
            weight = self.lambda_stratified_rank
            if self.aux_mode == "cosine":
                cosine = self._aux_cosine(main_loss, stratified_loss, pooled)
                weight = weight * self._aux_weight(cosine)
                self.log("stratified_cosine", cosine, on_step=False,
                         on_epoch=True, sync_dist=True, batch_size=len(logits))
            loss = loss + weight * stratified_loss
            self.log("stratified_rank_loss", stratified_loss, on_step=False,
                     on_epoch=True, sync_dist=True, batch_size=len(logits))

        if self.lambda_coupling > 0:
            coupling_loss = self._coupling_loss(logits)
            loss = loss + self.lambda_coupling * coupling_loss
            self.log("coupling_loss", coupling_loss, on_step=False,
                     on_epoch=True, sync_dist=True, batch_size=len(logits))

        if self.lambda_rank > 0 or self.lambda_stratified_rank > 0 \
                or self.lambda_coupling > 0:
            self._rank_push(logits, batch)

        self.log("train_loss", loss, prog_bar=True, on_step=True,
                 on_epoch=True, sync_dist=True, batch_size=len(logits))
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        logits = self(batch["volumes"], batch["meta"], batch["mask"])
        self._val_outputs[dataloader_idx].append({
            "probs": torch.sigmoid(logits.float()),
            "targets": batch["targets"].float(),
        })

    def _gathered(self, dataloader_idx: int):
        """All-gather one buffer and return (truth, probs) as numpy."""
        entries = self._val_outputs[dataloader_idx]
        if not entries:
            return None, None
        probs = torch.cat([e["probs"] for e in entries])
        targets = torch.cat([e["targets"] for e in entries])

        # all_gather returns (world_size, N, L) when every rank holds the same
        # N, which the distributed sampler guarantees by padding.
        probs = self.all_gather(probs)
        targets = self.all_gather(targets)
        if probs.ndim == 3:
            probs = probs.reshape(-1, probs.shape[-1])
            targets = targets.reshape(-1, targets.shape[-1])
        return (targets.cpu().numpy(), probs.cpu().numpy())

    def _log_conditional(self, prefix: str, truth, probs) -> None:
        """Log conditional AUC beside the pooled number, per stratified finding.

        The plan's §3, applied where it is cheapest to apply: pooled AUC on
        synovitis is reproducible from the effusion label alone (0.698 against
        the head's 0.701), so an epoch-level pooled number cannot show whether
        the run is learning the finding or learning effusion. Conditional AUC
        is the same statistic restricted to pairs effusion does not already
        separate.

        Stratified on the model's own predicted effusion, split at its median
        -- there is no gold effusion label in the mined validation fold, and
        on the gold set the prediction gives 0.617 against the label's 0.630,
        so nothing material is lost by using the one that always exists.

        Monitoring only, like `gold_macro`. Selecting a checkpoint on any
        gold-derived number is what inflated an earlier version of this
        project's reported figure.
        """
        for name in self.stratified_findings:
            index = config.LABELS.index(name)
            strata = conditional.strata_from_scores(probs[:, self.stratified_index])
            value = conditional.conditional_auc(
                truth[:, index], probs[:, index], strata)
            if np.isfinite(value):
                self.log(f"{prefix}_{name}_conditional", float(value),
                         rank_zero_only=False)
            pooled = auc(truth[:, index], probs[:, index])
            if np.isfinite(pooled):
                self.log(f"{prefix}_{name}_pooled", float(pooled),
                         rank_zero_only=False)

    def on_validation_epoch_end(self):
        val_truth, val_probs = self._gathered(0)
        if val_truth is not None:
            # Mined labels are soft; binarise for a ranking metric.
            binarised = (val_truth >= 0.5).astype(int)
            val_macro = macro_auc(binarised, val_probs)
            self.log("val_macro", val_macro, prog_bar=True, rank_zero_only=False)
            self._log_conditional("val", binarised, val_probs)

        gold_truth, gold_probs = self._gathered(1)
        if gold_truth is not None:
            gold_macro = macro_auc(gold_truth.astype(int), gold_probs)
            # Monitoring only. Never pass this to a checkpoint callback.
            self.log("gold_macro", gold_macro, prog_bar=True,
                     rank_zero_only=False)
            self._log_conditional("gold", gold_truth.astype(int), gold_probs)

        self._val_outputs = {0: [], 1: []}

    def configure_optimizers(self):
        # One group unless a trunk asks for a slower one. The aggregation is
        # trained from scratch and wants the full rate; a pretrained trunk at
        # the same rate is fine for a supervised convnet and destroys a
        # self-supervised ViT's features during OneCycle's warmup. Scale 1.0
        # takes the original single-group path unchanged, so every ConvNeXt
        # result stays comparable.
        if self.backbone_lr_scale == 1.0:
            optimiser = torch.optim.AdamW(self.parameters(), lr=self.lr,
                                          weight_decay=self.weight_decay)
            max_lr = self.lr * self.max_lr_multiplier
        else:
            trunk_ids = {id(p) for p in self.model.backbone.parameters()}
            trunk = [p for p in self.parameters() if id(p) in trunk_ids]
            rest = [p for p in self.parameters() if id(p) not in trunk_ids]
            trunk_lr = self.lr * self.backbone_lr_scale
            optimiser = torch.optim.AdamW(
                [{"params": trunk, "lr": trunk_lr},
                 {"params": rest, "lr": self.lr}],
                lr=self.lr, weight_decay=self.weight_decay)
            max_lr = [trunk_lr * self.max_lr_multiplier,
                      self.lr * self.max_lr_multiplier]

        scheduler = self._scheduler(optimiser, max_lr)
        return {
            "optimizer": optimiser,
            # Stepped per optimiser step, so accumulation is already folded in.
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def _scheduler(self, optimiser, max_lr):
        """OneCycle or cosine, with warmup pinned to an absolute step count.

        **The restretch is the thing being fixed here.** `OneCycleLR`'s
        `pct_start` is a *fraction* of the run, and it defaults to 0.3. At
        EPOCHS=12 that puts the learning-rate peak near epoch 3.6; at
        EPOCHS=20 it moves to epoch 6. Two runs of different lengths are
        therefore at different points of a differently-shaped schedule at the
        same epoch number, and "epoch 9 was best" does not mean the same thing
        in each. Every cross-length comparison this project has drawn was
        reading through that.

        Setting `warmup_steps` to an absolute count fixes the peak in place
        regardless of run length. Leaving it None keeps the 0.3 fraction, so
        the twelve-epoch reference runs stay reproducible.

        `schedule="cosine"` replaces the cycle with linear warmup then cosine
        decay -- the schedule the RSNA 2022/2023 winning solutions used for
        fine-tuning, and one with no restretch artefact to reason about at
        all. It peaks at the same `lr * max_lr_multiplier` as OneCycle so the
        two are comparable at matched warmup.
        """
        warmup = self.warmup_steps
        if self.schedule == "onecycle":
            pct_start = 0.3 if warmup is None else warmup / self.total_steps
            return torch.optim.lr_scheduler.OneCycleLR(
                optimiser, max_lr=max_lr, total_steps=self.total_steps,
                pct_start=pct_start)

        warmup = int(round(0.3 * self.total_steps)) if warmup is None else warmup
        peak = self.max_lr_multiplier
        decay_steps = max(1, self.total_steps - warmup)

        def factor(step: int) -> float:
            # Returned as a multiple of the group's base lr, which is already
            # `lr * backbone_lr_scale` for the trunk group -- so the two
            # groups keep their ratio through the whole schedule.
            if step < warmup:
                return peak * (step + 1) / max(1, warmup)
            progress = min(1.0, (step - warmup) / decay_steps)
            return peak * 0.5 * (1.0 + np.cos(np.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimiser, factor)


def optimiser_steps(n_studies: int, batch_size: int, devices: int,
                    accumulate: int, epochs: int) -> int:
    """Total OneCycleLR steps, which must match what Lightning will take.

    A schedule sized for the wrong number of steps either finishes its cycle
    early and trains the tail at a near-zero learning rate, or never reaches
    its peak. Neither raises.
    """
    per_rank_batches = int(np.ceil(n_studies / (batch_size * max(devices, 1))))
    return epochs * (per_rank_batches // accumulate + 1)


class WeightAverage(pl.Callback):
    """Maintain EMA and SWA of the `KneeNet` weights and write both out.

    This is the selection-free half of the answer to the biased-selector
    problem described in `src/model/averaging.py`: rather than trusting mined
    validation to name an epoch, average the trajectory and ship the average.
    The best-validation checkpoint is still written by `ModelCheckpoint`
    alongside these, so a run produces three candidates and the comparison
    between them is the measurement.

    **Updated on optimiser steps, not on batches.** With gradient
    accumulation, most batches leave the weights untouched, and folding an
    unchanged model into the EMA `accumulate` times per step would make the
    effective decay `decay ** accumulate` -- a shorter memory than asked for,
    varying with a knob that is supposed to be numerically neutral.
    `on_before_optimizer_step` fires exactly when the weights are about to
    change, so counting there tracks the schedule's own step count.

    **Rank zero only.** DDP keeps every rank's weights identical after each
    step, so every rank would compute the same average; doing it once saves
    the copies. The written files therefore exist only on rank zero, which is
    also the only rank whose `/kaggle/working` survives.

    **The averages are written from inside `fit`, in `on_train_end`, and read
    back off disk afterwards.** Under `ddp_notebook` the ranks are forked
    children: the callback object the notebook constructed is *not* the object
    that accumulated anything, and asking it for its EMA afterwards returns an
    empty average with no error. This is the same trap that leaves
    `best_model_score` as None on that strategy. So the file paths are fixed
    at construction and the notebook globs for what appeared.
    """

    def __init__(self, prefix: str, decay: float = 0.999,
                 swa_fraction: float = 0.25,
                 total_steps: Optional[int] = None):
        if not 0.0 <= swa_fraction <= 1.0:
            raise ValueError(f"swa_fraction must be in [0, 1], got {swa_fraction}")
        self.prefix = prefix
        self.decay = decay
        self.swa_fraction = swa_fraction
        self.total_steps = total_steps
        self.average: Optional[averaging.WeightAverage] = None
        self.written: Dict[str, str] = {}

    @property
    def ema_path(self) -> str:
        return f"{self.prefix}_ema.pt"

    @property
    def swa_path(self) -> str:
        return f"{self.prefix}_swa.pt"

    def _swa_from(self, trainer) -> Optional[int]:
        """The step the equal-weight tail average starts at."""
        total = self.total_steps
        if total is None:
            total = getattr(trainer, "estimated_stepping_batches", None)
        if not total or self.swa_fraction <= 0:
            return None
        # +1 because `update` counts from one: at total=4 and fraction=0.5 the
        # window must open at step 3, so that steps 3 and 4 -- the last half --
        # are what gets averaged, not steps 2 through 4.
        return int(round(float(total) * (1.0 - self.swa_fraction))) + 1

    def on_train_start(self, trainer, module):
        if not trainer.is_global_zero:
            return
        self.average = averaging.WeightAverage(
            module.model, decay=self.decay, swa_from=self._swa_from(trainer))

    def on_before_optimizer_step(self, trainer, module, optimizer):
        if self.average is not None:
            self.average.update(module.model)

    def on_train_end(self, trainer, module):
        # Wrapped, on the same rule the notebook's export cell follows: by the
        # time this fires, hours of GPU time exist as the monitored checkpoint
        # `ModelCheckpoint` already wrote, and an exception raised here
        # propagates out of `trainer.fit` and takes the export and scoring
        # cells with it. The averages are an experiment; the run is not.
        try:
            self.export()
        except Exception as exc:  # noqa: BLE001 - the checkpoint is safe
            print(f"WeightAverage could not write its averages ({exc}). "
                  "The monitored checkpoint is unaffected; only the "
                  "selection-free comparison is lost.")

    def export(self) -> Dict[str, str]:
        """Write the averages as bare `KneeNet` state dicts.

        The same format `export_from_checkpoint` produces, so notebook 10
        loads them with no idea they came from an average. Returns
        {kind: path} for what was actually written -- SWA is absent when the
        run was too short for any step to fall inside the tail window.
        """
        self.written = {}
        if self.average is None:
            return self.written

        torch.save(self.average.ema_state(), self.ema_path)
        self.written["ema"] = self.ema_path

        swa = self.average.swa_state()
        if swa is not None:
            torch.save(swa, self.swa_path)
            self.written["swa"] = self.swa_path
        return self.written


class EpochReport(pl.Callback):
    """One line per epoch, from rank zero only.

    Under DDP every rank runs this code, so an unguarded print produces one
    copy of the log per GPU and makes a run look twice as long as it is.

    **This reads `callback_metrics` in `on_validation_end`, not in
    `on_validation_epoch_end`.** Lightning runs callback hooks before the
    LightningModule's hook of the same name, so a callback asking for
    `val_macro` at `on_validation_epoch_end` arrives before
    `KneeLitModule.on_validation_epoch_end` has computed it and reads an
    empty dict -- printing `nan` for every epoch of an otherwise healthy run.
    `ModelCheckpoint` selects in `on_validation_end` for the same reason, so
    checkpoint selection was never affected by this; only the log was.
    """

    def __init__(self):
        self.rows: List[Dict[str, float]] = []

    def on_validation_end(self, trainer, module):
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        metrics = trainer.callback_metrics
        val = float(metrics.get("val_macro", float("nan")))
        gold = float(metrics.get("gold_macro", float("nan")))
        self.rows.append({"epoch": trainer.current_epoch + 1,
                          "val_macro": val, "gold_macro": gold})
        best = max(row["val_macro"] for row in self.rows)
        marker = "  <- best" if val >= best else ""
        print(f"  epoch {trainer.current_epoch + 1}: val macro {val:.3f}  "
              f"gold {gold:.3f} (monitor only){marker}")


class BatchProgress(pl.Callback):
    """Plain-text tqdm over training batches, rank zero only.

    Lightning's own progress bar resolves to the notebook-widget flavour of
    tqdm inside Kaggle, and the widget never reaches the log of a
    Save & Run All -- eleven minutes pass per epoch with no sign of life.
    Plain tqdm writes carriage-returned text to stdout, which survives both
    the interactive view and the batch log, so the Trainer disables its own
    bar and takes this instead. `mininterval` keeps the batch log from
    bloating: one refresh per interval, not one per batch.
    """

    def __init__(self, refresh_seconds: float = 15.0):
        self.refresh_seconds = refresh_seconds
        self._bar = None

    def on_train_epoch_start(self, trainer, module):
        if not trainer.is_global_zero:
            return
        import sys
        from tqdm import tqdm
        self._bar = tqdm(
            total=int(trainer.num_training_batches),
            desc=f"epoch {trainer.current_epoch + 1}/{trainer.max_epochs}",
            unit="batch", file=sys.stdout,
            mininterval=self.refresh_seconds, leave=False)

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if self._bar is not None:
            self._bar.update(1)

    def on_train_epoch_end(self, trainer, module):
        if self._bar is not None:
            self._bar.close()
            self._bar = None


def export_from_checkpoint(ckpt_path: str, out_path: str) -> int:
    """Convert a Lightning checkpoint into the plain state dict for scoring.

    Read from the file rather than from a module in memory: under the spawn
    strategy the process that called `fit` is not the process that trained,
    and reaching for in-memory weights there is how a run silently ships its
    initialisation. Returns the number of tensors written.
    """
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    state = {key[len("model."):]: value
             for key, value in state_dict.items()
             if key.startswith("model.")}
    if not state:
        raise ValueError(
            f"no 'model.' keys in {ckpt_path}; found "
            f"{list(state_dict)[:3]}")
    torch.save(state, out_path)
    return len(state)


def export_state_dict(module: KneeLitModule, path: str) -> None:
    """Write a plain `KneeNet` state dict, the format the submission loads.

    Lightning prefixes every key with `model.` because `KneeNet` is an
    attribute of the wrapper. Notebook 10 constructs a bare `KneeNet` and
    calls `load_state_dict`, so the prefix has to come off here rather than
    become a surprise at submission time.
    """
    state = {key[len("model."):]: value
             for key, value in module.state_dict().items()
             if key.startswith("model.")}
    torch.save(state, path)
