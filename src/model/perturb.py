"""Controlled input variants for a sensitivity probe -- never new labels.

Each variant is a deliberate, small change to the exact tensor the model
reads: `load_study` output, already mirrored onto the canonical right knee and
subsampled to 16 slices. The probe compares each study's prediction on a
variant with its prediction on the unmodified input. Nothing here says what
the knee actually shows; a variant that moves a prediction measures what the
model reacts to, not whether the original or the variant is right.

The grid is declared here, before any run, so the reading cannot be tuned to
the result. Two variants need care in interpretation:

- **`flip_lr`** mirrors every series with `augment.flip_laterality` -- the
  plane-aware axis the training augmentation and the submission's flip TTA
  use -- and its predictions are permuted back with `lateral_swap_order()`
  before comparison. MCL has no lateral counterpart among the labels, so after
  a flip the model is asked about the MCL where the lateral structures now
  sit. An MCL change there measures position reliance against anatomy
  reading; it is not a broken invariance, and the summary marks it.
- **`side_toggle`** inverts the `is_right` covariate on series whose side is
  known and leaves the pixels alone. It separates reliance on the side
  classifier's covariate from reliance on image laterality.
"""

import hashlib
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

from src import config
from src.model import augment
from src.model.conditional import logit
from src.model.dataset import META_COLUMNS

IS_RIGHT = META_COLUMNS.index("is_right")
SIDE_KNOWN = META_COLUMNS.index("side_known")

DEPTH = 16                      # the probed model's slices per series
FLOOR_P95_MULTIPLE = 3.0        # "sensitive": p95 |dp| above this x floor ...
MIN_SENSITIVE_P95 = 0.05        # ... and above this absolute level
REPRODUCTION_TOLERANCE = 0.01   # identity vs saved predictions, max |dp|
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260924

# Findings whose flip expectation is undefined: no lateral twin among labels.
FLIP_UNPAIRED = ("MCL",)


def _copy(item: Dict) -> Dict:
    return {**item, "volumes": item["volumes"].copy(), "meta": item["meta"].copy()}


def shift_intensity(item: Dict, delta: float) -> Dict:
    """Add `delta` 8-bit levels to every voxel, clipped and rounded once."""
    out = _copy(item)
    working = item["volumes"].astype(np.float32) + delta
    out["volumes"] = np.clip(working, 0, 255).astype(np.uint8)
    return out


def add_noise(item: Dict, sigma: float, seed: int) -> Dict:
    """Gaussian noise of `sigma` 8-bit levels from one seeded generator."""
    out = _copy(item)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, size=item["volumes"].shape).astype(np.float32)
    working = item["volumes"].astype(np.float32) + noise
    out["volumes"] = np.clip(working, 0, 255).astype(np.uint8)
    return out


def drop_slices(item: Dict, keep: np.ndarray) -> Dict:
    """Keep the same slice indices in every series of the study.

    Indices stay ascending so `stack_2p5d` still sees real neighbours; the
    slice encoder's positions come from `linspace` over whatever depth arrives.
    """
    keep = np.asarray(keep)
    if keep.ndim != 1 or len(keep) == 0 or (np.diff(keep) <= 0).any():
        raise ValueError("keep must be ascending, distinct slice indices")
    out = _copy(item)
    out["volumes"] = np.ascontiguousarray(item["volumes"][:, keep])
    return out


def toggle_side(item: Dict) -> Dict:
    """Invert `is_right` where the side is known; unknown sides stay unknown."""
    out = _copy(item)
    known = out["meta"][:, SIDE_KNOWN] > 0.5
    out["meta"][known, IS_RIGHT] = 1.0 - out["meta"][known, IS_RIGHT]
    return out


def flip(item: Dict) -> Dict:
    """Correct laterality flip; `is_right` stays as acquired, as in training."""
    return augment.flip_laterality(_copy(item), None)


def noise_seed(study: str, variant: str) -> int:
    """Fixed per (study, variant), independent of run order and process."""
    digest = hashlib.sha256(f"{study}|{variant}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _shift(delta):
    return lambda item, study: shift_intensity(item, delta)


def _noise(sigma, name):
    return lambda item, study: add_noise(item, sigma, noise_seed(study, name))


def _slices(keep):
    return lambda item, study: drop_slices(item, keep)


# name -> (family, transform). Declared before any run; order is output order.
VARIANTS: Dict[str, tuple] = {"identity": ("identity", lambda item, study: _copy(item))}
for _delta in (-20, -10, -5, 5, 10, 20):
    VARIANTS[f"shift_{_delta:+d}"] = ("intensity_shift", _shift(_delta))
for _sigma in (2, 5, 10):
    VARIANTS[f"noise_{_sigma}"] = ("noise", _noise(_sigma, f"noise_{_sigma}"))
VARIANTS["slices_even12"] = ("slice_removal", _slices(
    np.linspace(0, DEPTH - 1, 12).round().astype(int)))
VARIANTS["slices_alternate8"] = ("slice_removal", _slices(np.arange(0, DEPTH, 2)))
VARIANTS["slices_central12"] = ("slice_removal", _slices(np.arange(2, DEPTH - 2)))
VARIANTS["flip_lr"] = ("laterality_flip", lambda item, study: flip(item))
VARIANTS["side_toggle"] = ("side_covariate", lambda item, study: toggle_side(item))


def apply(name: str, item: Dict, study: str) -> Dict:
    """Return a new item; the input item is never modified."""
    if item["volumes"].ndim != 4 or item["volumes"].shape[1] != DEPTH:
        raise ValueError(f"expected (series, {DEPTH}, H, W) volumes")
    transform: Callable = VARIANTS[name][1]
    return transform(item, study)


def align(name: str, probabilities: np.ndarray) -> np.ndarray:
    """Map a variant's outputs back onto the original finding order."""
    probabilities = np.asarray(probabilities)
    if name == "flip_lr":
        return probabilities[..., augment.lateral_swap_order()]
    return probabilities


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ranks_a = pd.Series(a).rank().to_numpy()
    ranks_b = pd.Series(b).rank().to_numpy()
    if ranks_a.std() == 0 or ranks_b.std() == 0:
        return float("nan")
    return float(np.corrcoef(ranks_a, ranks_b)[0, 1])


def _wide(frame: pd.DataFrame, variant: str, studies) -> np.ndarray:
    rows = frame[frame.variant == variant].set_index(config.ID_COL)
    if rows.index.duplicated().any() or set(rows.index) != set(studies):
        raise ValueError(f"{variant}: every study must appear exactly once")
    return rows.loc[studies, config.LABELS].to_numpy(float)


def reproduction(identity: np.ndarray, reference: np.ndarray) -> Dict:
    worst = float(np.abs(identity - reference).max())
    return {"max_abs_dp": worst, "tolerance": REPRODUCTION_TOLERANCE,
            "passed": worst <= REPRODUCTION_TOLERANCE}


def summarise(predictions: pd.DataFrame, reference: Optional[pd.DataFrame] = None) -> Dict:
    """Per variant x finding consistency against the identity pass.

    `predictions` is long format (study, variant, twelve raw probabilities);
    flipped outputs are aligned here. `reference` is the saved prediction of
    the same model; identity against it gives the noise floor and the
    reproduction check. Without it the floor is taken as zero.
    """
    studies = sorted(predictions.loc[predictions.variant == "identity", config.ID_COL])
    base = _wide(predictions, "identity", studies)
    if reference is not None:
        saved = reference.set_index(config.ID_COL).loc[studies, config.LABELS].to_numpy(float)
        floor = np.percentile(np.abs(base - saved), 95, axis=0)
        check = reproduction(base, saved)
    else:
        floor = np.zeros(len(config.LABELS))
        check = None
    resamples = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, len(studies), size=(BOOTSTRAP_DRAWS, len(studies)))
    rows = []
    for name, (family, _) in VARIANTS.items():
        if name == "identity" or not (predictions.variant == name).any():
            continue
        variant = align(name, _wide(predictions, name, studies))
        dp = variant - base
        dlogit = logit(variant) - logit(base)
        means = dlogit[resamples].mean(axis=1)
        for j, finding in enumerate(config.LABELS):
            p95 = float(np.percentile(np.abs(dp[:, j]), 95))
            rows.append({
                "variant": name, "family": family, "finding": finding,
                "median_abs_dp": float(np.median(np.abs(dp[:, j]))),
                "p95_abs_dp": p95,
                "mean_dlogit": float(dlogit[:, j].mean()),
                "dlogit_95ci": [float(np.percentile(means[:, j], 2.5)),
                                float(np.percentile(means[:, j], 97.5))],
                "spearman": _spearman(base[:, j], variant[:, j]),
                "sensitive": p95 > max(MIN_SENSITIVE_P95, FLOOR_P95_MULTIPLE * floor[j]),
                "note": ("flip expectation undefined: no lateral counterpart label"
                         if name == "flip_lr" and finding in FLIP_UNPAIRED else ""),
            })
    return {
        "declared": {"variants": {k: v[0] for k, v in VARIANTS.items()},
                     "sensitive_rule": f"p95_abs_dp > max({MIN_SENSITIVE_P95}, "
                                       f"{FLOOR_P95_MULTIPLE} x identity-vs-reference p95)",
                     "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED,
                                   "unit": "study"},
                     "labels_used": False},
        "studies": len(studies),
        "reproduction": check,
        "floor_p95_abs_dp": dict(zip(config.LABELS, map(float, floor))),
        "rows": rows,
    }
