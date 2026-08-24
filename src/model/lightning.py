"""Lightning wrapper for the end-to-end fine-tuning run.

Kaggle offers a T4 x2 accelerator, and the training loop is GPU-bound at
~220 images/second, so a second device is close to a free halving of
wall-clock. Lightning is here for the distributed plumbing, not for the
abstraction: the semantics below are a faithful port of the hand-written loop
in notebook 08, and any difference in the resulting score is a bug rather
than a design choice.

**Everything in this module lives here rather than in a notebook cell for one
reason.** Lightning's notebook-safe DDP strategy spawns worker processes and
pickles the dataset and module into them. A class defined in a notebook cell
belongs to `__main__` and does not survive that trip -- this project has
already paid for that lesson once, which is why the cache's process-pool
worker lives in `src/data/cache.py`. The bootstrap writes these files to disk
and imports them as `src.model.lightning`, so the spawn finds them by name.

Three DDP hazards are handled explicitly, because each one runs clean and
silently scores worse:

**Memory-mapped shards are never pickled.** The cache is ~29 GB across three
memmaps. Holding them as attributes would serialise them into every spawned
process. They are opened lazily per process and dropped from `__getstate__`.

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
                 augment: bool = False):
        self.index = study_index
        self.studies = list(studies)
        self.cache_paths = list(cache_paths)
        self.slices_per_series = slices_per_series
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

        if self.augment:
            # Intensity jitter only. A geometric flip would swap medial and
            # lateral; rotation is safe but not part of this measurement.
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
                 total_steps: int = 1000, max_lr_multiplier: float = 3.0):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        self.model = net.KneeNet(backbone, feature_dim)
        self.lr = lr
        self.weight_decay = weight_decay
        self.total_steps = total_steps
        self.max_lr_multiplier = max_lr_multiplier
        # Not `_buffers`: nn.Module reserves that name for registered
        # buffers, and shadowing it corrupts state_dict and .to(device).
        self._val_outputs: Dict[int, List[Dict[str, torch.Tensor]]] = {0: [], 1: []}

    def forward(self, volumes, meta, mask):
        return self.model(volumes, meta, mask)

    def training_step(self, batch, batch_idx):
        logits = self(batch["volumes"], batch["meta"], batch["mask"])
        loss = net.weighted_bce(logits, batch["targets"], batch["weights"])
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

    def on_validation_epoch_end(self):
        val_truth, val_probs = self._gathered(0)
        if val_truth is not None:
            # Mined labels are soft; binarise for a ranking metric.
            val_macro = macro_auc((val_truth >= 0.5).astype(int), val_probs)
            self.log("val_macro", val_macro, prog_bar=True, rank_zero_only=False)

        gold_truth, gold_probs = self._gathered(1)
        if gold_truth is not None:
            gold_macro = macro_auc(gold_truth.astype(int), gold_probs)
            # Monitoring only. Never pass this to a checkpoint callback.
            self.log("gold_macro", gold_macro, prog_bar=True,
                     rank_zero_only=False)

        self._val_outputs = {0: [], 1: []}

    def configure_optimizers(self):
        optimiser = torch.optim.AdamW(self.parameters(), lr=self.lr,
                                      weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimiser, max_lr=self.lr * self.max_lr_multiplier,
            total_steps=self.total_steps)
        return {
            "optimizer": optimiser,
            # Stepped per optimiser step, so accumulation is already folded in.
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


def optimiser_steps(n_studies: int, batch_size: int, devices: int,
                    accumulate: int, epochs: int) -> int:
    """Total OneCycleLR steps, which must match what Lightning will take.

    A schedule sized for the wrong number of steps either finishes its cycle
    early and trains the tail at a near-zero learning rate, or never reaches
    its peak. Neither raises.
    """
    per_rank_batches = int(np.ceil(n_studies / (batch_size * max(devices, 1))))
    return epochs * (per_rank_batches // accumulate + 1)


class EpochReport(pl.Callback):
    """One line per epoch, from rank zero only.

    Under DDP every rank runs this code, so an unguarded print produces one
    copy of the log per GPU and makes a run look twice as long as it is.
    """

    def __init__(self):
        self.rows: List[Dict[str, float]] = []

    def on_validation_epoch_end(self, trainer, module):
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
