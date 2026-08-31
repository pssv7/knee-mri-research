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


def scanner_key(index: pd.DataFrame) -> pd.Series:
    """A per-study scanner identity, from the header scan's index.

    The dataset spans 16-19 sites on five continents, and a CNN reads scanner
    signature as readily as it reads pathology. A fold split that puts the same
    scanner on both sides therefore reports an accuracy that does not survive
    a new site: the plan's figure is ~0.053 of inflation from random K-fold,
    with one measured model showing a 0.136 grouped-vs-random gap.

    Site is not given, so manufacturer + model + field strength stands in for
    it. That is coarser than a site id -- two hospitals with the same scanner
    collapse into one group, which makes the split *more* conservative, not
    less -- and it is what the DICOM headers actually carry.

    Takes the frame `src/data/index.py` writes (one row per instance or
    series, carrying `StudyInstanceUID`), and returns one key per study,
    indexed by study UID. A study whose headers name none of the three fields
    gets "unknown", and those all land in one group together.
    """
    fields = [c for c in ("Manufacturer", "ManufacturerModelName",
                          "MagneticFieldStrength") if c in index.columns]
    if not fields:
        raise ValueError(
            "the index carries none of Manufacturer / ManufacturerModelName / "
            f"MagneticFieldStrength; it has {list(index.columns)[:8]}")

    # fillna after astype: pandas keeps a missing value missing through
    # astype(str) rather than rendering it "nan", and the join below would
    # then fail on a float instead of producing a key.
    joined = index[fields].astype(str).fillna("nan").agg("|".join, axis=1)
    joined = joined.where(~joined.str.fullmatch(r"(nan\|?)+"), "unknown")
    # The first series scanned decides the study. A study imaged on two
    # scanners is rare and either group is defensible; taking the first keeps
    # the assignment deterministic.
    return joined.groupby(index[config.ID_COL]).first()


def make_folds(labels: pd.DataFrame, n_folds: int = N_FOLDS,
               seed: int = SEED,
               groups: Optional[pd.Series] = None) -> pd.DataFrame:
    """Assign each study to a fold, stratified by positive-finding count.

    Returns the frame with a `fold` column added.

    `groups` maps study UID -> group id (see `scanner_key`). When given, a
    whole group lands in one fold: the split becomes leave-scanner-out rather
    than leave-study-out, so validation measures transfer to unseen equipment
    instead of rewarding memorised scanner signature.

    Grouping costs balance. Groups are sized very unequally -- one scanner may
    hold a quarter of the corpus -- so groups are assigned largest-first to
    whichever fold is currently smallest, which is the standard greedy
    approximation and the best available without splitting a group. Check
    `summarise()` afterwards: badly imbalanced folds are the price of grouping,
    and worth knowing before a run rather than after.

    **Changing this re-cuts every fold**, and every reference number on record
    (fold 1: val 0.872 / gold 0.861 / LB 0.889) belongs to the ungrouped
    split. Adopt it when the fold ensemble is rebuilt, not in the middle of an
    experiment that is being compared against those numbers.
    """
    frame = labels.copy().reset_index(drop=True)
    key = stratification_key(frame)
    rng = np.random.default_rng(seed)

    frame["fold"] = -1
    if groups is None:
        for value in key.unique():
            rows = np.flatnonzero(key.to_numpy() == value)
            rng.shuffle(rows)
            frame.loc[rows, "fold"] = np.arange(len(rows)) % n_folds
    else:
        mapped = frame[config.ID_COL].map(groups).fillna("unknown")
        sizes = mapped.value_counts()
        # Shuffle before sorting so equal-sized groups do not always fall the
        # same way, then place the largest first into the emptiest fold.
        order = sizes.sample(frac=1.0, random_state=seed).sort_values(
            ascending=False, kind="mergesort").index
        counts = np.zeros(n_folds, dtype=int)
        for group in order:
            fold = int(np.argmin(counts))
            rows = np.flatnonzero(mapped.to_numpy() == group)
            frame.loc[rows, "fold"] = fold
            counts[fold] += len(rows)

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
