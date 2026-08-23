"""Cross-validation splits and the gold holdout.

Two rules govern every split here, and both exist because the obvious
alternative produces a number that looks better and means less.

**Split by study, never by series.** A study's ~5.5 series image the same knee
in the same session. A random series split puts near-duplicates on both sides
and reports an accuracy the model does not have.

**The 58 gold studies never enter training.** They are the only radiologist
labels in the dataset and the only honest measurement available. Everything
else is trained on mined labels, and a model evaluated against mined labels can
score well by reproducing the extractor's mistakes. `assert_no_gold_leakage`
is called by the training entrypoint rather than trusted to discipline.
"""

from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src import config

N_FOLDS = 5
SEED = 0


def load_gold_uids(train_csv=None) -> set:
    """Study UIDs of the 58 radiologist-labelled studies."""
    frame = pd.read_csv(train_csv or config.TRAIN_CSV)
    labelled = frame[frame[config.LABELS[0]].notna()]
    return set(labelled[config.ID_COL])


def assert_no_gold_leakage(train_uids: Sequence[str], gold_uids: set) -> None:
    """Fail loudly if a gold study appears in training data.

    Called from the training entrypoint. The failure this guards against is
    silent -- a leaked gold study inflates the one number being used to decide
    whether the model works.
    """
    leaked = set(train_uids) & gold_uids
    if leaked:
        raise ValueError(
            f"{len(leaked)} gold studies leaked into training: "
            f"{sorted(leaked)[:3]}..."
        )


def stratification_key(labels: pd.DataFrame) -> pd.Series:
    """A coarse key for balancing folds across a 12-label problem.

    Full multi-label stratification over 2^12 combinations would leave most
    cells empty at 4,349 studies. The count of positive findings captures the
    axis that matters -- studies with many findings are rare and should not
    cluster in one fold.
    """
    positives = (labels[config.LABELS] > 0.5).sum(axis=1)
    return positives.clip(upper=4)


def make_folds(labels: pd.DataFrame, n_folds: int = N_FOLDS,
               seed: int = SEED) -> pd.DataFrame:
    """Assign each study to a fold, stratified by positive-finding count.

    Returns the frame with a `fold` column added.
    """
    frame = labels.copy().reset_index(drop=True)
    key = stratification_key(frame)
    rng = np.random.default_rng(seed)

    frame["fold"] = -1
    for value in key.unique():
        rows = np.flatnonzero(key.to_numpy() == value)
        rng.shuffle(rows)
        frame.loc[rows, "fold"] = np.arange(len(rows)) % n_folds

    if (frame["fold"] < 0).any():
        raise RuntimeError("some studies were not assigned a fold")
    return frame


def iter_folds(labels: pd.DataFrame,
               n_folds: int = N_FOLDS) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Yield (train, validation) frames for each fold."""
    if "fold" not in labels.columns:
        labels = make_folds(labels, n_folds=n_folds)

    for fold in range(n_folds):
        validation = labels[labels["fold"] == fold]
        train = labels[labels["fold"] != fold]
        yield train.reset_index(drop=True), validation.reset_index(drop=True)


def series_for_studies(manifest: pd.DataFrame,
                       study_uids: Sequence[str]) -> pd.DataFrame:
    """All cached series belonging to a set of studies."""
    wanted = set(study_uids)
    return manifest[manifest[config.ID_COL].isin(wanted)].reset_index(drop=True)


def summarise(labels: pd.DataFrame, n_folds: int = N_FOLDS) -> pd.DataFrame:
    """Per-fold positive rates, to confirm the folds are actually comparable."""
    if "fold" not in labels.columns:
        labels = make_folds(labels, n_folds=n_folds)

    rows = []
    for fold in range(n_folds):
        subset = labels[labels["fold"] == fold]
        row = {"fold": fold, "studies": len(subset)}
        for name in config.LABELS:
            row[name] = float((subset[name] > 0.5).mean())
        rows.append(row)
    return pd.DataFrame(rows).set_index("fold")
