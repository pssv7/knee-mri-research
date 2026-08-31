"""Look at the studies the model gets wrong.

Every aggregate number this project has produced says *how much* is wrong and
none of them say *what*. The worst bugs on record were not modelling errors --
slice order reversed by `InstanceNumber`, sagittal mirroring flipping the wrong
axis -- and both raised nothing, scored plausibly, and cost about 0.03 each.
That class is found by looking at the pixels, not by reading a macro.

**What gets rendered is the model's own view**, not the raw DICOM: the volumes
that come out of `dataset.load_study`, after laterality mirroring and the slice
subsample the training loop applies. If the mirroring is flipping the wrong
axis, or a series is blank, or the slices run backwards, it is visible here and
nowhere else.

Each sheet is one study: series down the page, slices across, every row
labelled with the metadata the model was conditioned on (plane, fluid
sensitivity, fat suppression, predicted side). The title carries the findings
this study got most wrong, so the eye knows what it is looking for.

Kept in `src/` rather than `tools/` because the cache lives on Kaggle and
Colab, so the code has to travel into the notebook's bootstrap.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np

from src import config
from src.model.dataset import META_COLUMNS


def per_finding_errors(truth: np.ndarray, preds: np.ndarray) -> np.ndarray:
    """(n_studies, n_findings) each study's share of the discordant pairs.

    AUC is the fraction of (positive, negative) pairs the model orders
    correctly. So the natural per-study error is the fraction of *its own*
    pairs it gets wrong: for a positive, the share of negatives ranked above
    it; for a negative, the share of positives ranked below it. Ties count a
    half, exactly as they do in the AUC itself.

    That makes this the study's contribution to `1 - AUC`, and it is zero under
    a perfect separation however the scores are distributed. An earlier version
    measured distance from an idealised rank instead, which charged a study for
    being one of many tied positives -- a perfectly separating model scored
    nonzero, and the sheets would have led with studies that were not wrong.

    Findings with only one class present are skipped: with 58 studies that
    happens, and there are no pairs to be concordant about.
    """
    truth = np.asarray(truth, dtype=float)
    preds = np.asarray(preds, dtype=float)
    if truth.shape != preds.shape:
        raise ValueError(f"truth {truth.shape} against preds {preds.shape}")

    out = np.zeros_like(truth, dtype=float)
    for column in range(truth.shape[1]):
        y = truth[:, column] == 1
        if y.all() or not y.any():
            continue
        scores = preds[:, column]
        positives, negatives = np.sort(scores[y]), np.sort(scores[~y])

        # For a positive: negatives strictly above it, plus half the ties.
        above = len(negatives) - np.searchsorted(negatives, scores[y], "right")
        ties = (np.searchsorted(negatives, scores[y], "right")
                - np.searchsorted(negatives, scores[y], "left"))
        out[y, column] = (above + 0.5 * ties) / len(negatives)

        # For a negative: positives strictly below it, plus half the ties.
        below = np.searchsorted(positives, scores[~y], "left")
        ties = (np.searchsorted(positives, scores[~y], "right")
                - np.searchsorted(positives, scores[~y], "left"))
        out[~y, column] = (below + 0.5 * ties) / len(positives)
    return out


def study_rank_errors(truth: np.ndarray, preds: np.ndarray) -> np.ndarray:
    """(n_studies,) the same error summed over the twelve findings.

    How much one study drags the macro down -- the right ordering to look at,
    rather than "most confident mistake", because AUC is a ranking statistic
    and does not care how confident anything was.
    """
    return per_finding_errors(truth, preds).sum(axis=1)


def describe_series(meta_row: Sequence[float]) -> str:
    """A one-line label for a series, from the vector the model was given.

    Reading it back out of `meta` rather than off the manifest is deliberate:
    this shows what the MODEL was told, which is the thing that can be wrong.
    """
    keys = {name: i for i, name in enumerate(META_COLUMNS)}
    plane = ("sag" if meta_row[keys["is_sagittal"]] > 0.5 else
             "cor" if meta_row[keys["is_coronal"]] > 0.5 else
             "ax" if meta_row[keys["is_axial"]] > 0.5 else "plane?")
    bits = [plane]
    if meta_row[keys["Fluid_Sensitive"]] > 0.5:
        bits.append("fluid")
    if meta_row[keys["Fat_Suppression"]] > 0.5:
        bits.append("fatsat")
    if meta_row[keys["side_known"]] > 0.5:
        bits.append("R" if meta_row[keys["is_right"]] > 0.5 else "L")
    else:
        bits.append("side?")
    return " ".join(bits)


def save_study_sheet(path, item: Dict, title: str = "",
                     columns: int = 8, dpi: int = 90) -> str:
    """Write one study as a contact sheet: series down, slices across.

    `item` is whatever `dataset.load_study` returned -- so the pixels here are
    exactly the pixels the backbone saw. Slices are sampled evenly across each
    series rather than taken from the front, because the front of a stack is
    usually empty and a sheet of black tells you nothing.

    matplotlib is imported inside the function: this module is bootstrapped
    into notebooks that never render anything, and an import at module scope
    would make them all depend on it.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    volumes, meta = item["volumes"], item["meta"]
    n_series = int(volumes.shape[0])
    if n_series == 0:
        raise ValueError("study has no series to render")
    depth = int(volumes.shape[1])
    picks = np.linspace(0, depth - 1, min(columns, depth)).round().astype(int)
    picks = np.unique(picks)

    figure, axes = plt.subplots(
        n_series, len(picks),
        figsize=(1.3 * len(picks), 1.35 * n_series), squeeze=False)
    for row in range(n_series):
        for col, slice_index in enumerate(picks):
            axis = axes[row][col]
            axis.imshow(volumes[row, slice_index], cmap="gray",
                        vmin=0, vmax=255)
            axis.set_xticks([]); axis.set_yticks([])
            if col == 0:
                axis.set_ylabel(describe_series(meta[row]), fontsize=6,
                                rotation=0, ha="right", va="center", labelpad=28)
            if row == 0:
                axis.set_title(f"z{slice_index}", fontsize=6)
    if title:
        figure.suptitle(title, fontsize=8)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96 if title else 1.0))
    figure.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return str(path)


def sheet_title(study: str, truth_row, pred_row, errors_row,
                worst: int = 3) -> str:
    """`study … | Synovitis truth 1 pred 0.04 | …` for the worst few findings."""
    order = np.argsort(errors_row)[::-1][:worst]
    parts = [f"{config.LABELS[i]} truth {truth_row[i]:.0f} "
             f"pred {pred_row[i]:.2f}" for i in order if errors_row[i] > 0]
    return f"{study[-12:]}  |  " + "  |  ".join(parts)


def render_failures(index, shards, truth_frame, preds_frame, out_dir,
                    limit: int = 8, best: int = 2) -> List[str]:
    """Render the worst studies, plus a couple of the best for comparison.

    The controls matter as much as the failures. A sheet of a study the model
    got right is what tells you whether the odd thing you are looking at in a
    failure is actually odd, or just what this dataset looks like.
    """
    import os
    from src.model import dataset as knee_dataset

    studies = [s for s in truth_frame.index if s in set(index.studies)]
    truth = truth_frame.loc[studies, config.LABELS].to_numpy(dtype=float)
    preds = preds_frame.loc[studies, config.LABELS].to_numpy(dtype=float)

    totals = study_rank_errors(truth, preds)
    per_finding = per_finding_errors(truth, preds)
    order = np.argsort(totals)[::-1]
    chosen = list(order[:limit]) + list(order[::-1][:best])

    os.makedirs(out_dir, exist_ok=True)
    written = []
    for rank, i in enumerate(chosen):
        study = studies[i]
        kind = "worst" if rank < limit else "best"
        item = knee_dataset.load_study(index, study, shards)
        title = (f"[{kind} {rank if kind == 'worst' else rank - limit}] "
                 f"rank error {totals[i]:.2f}  "
                 + sheet_title(study, truth[i], preds[i], per_finding[i]))
        path = os.path.join(out_dir, f"{kind}{rank:02d}_{study[-12:]}.png")
        written.append(save_study_sheet(path, item, title=title))
    return written
