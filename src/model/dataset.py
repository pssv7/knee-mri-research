"""Study-level dataset over the cached volumes.

Joins four sources into one training sample per study:

    cache shards      the uint8 volumes, memory-mapped
    manifest          which shard row holds which series
    fused labels      12 soft targets and 12 sample weights per study
    side predictions  L/R per series, from the tag or the classifier

A sample is a whole study, not a series. The twelve findings are study-level
labels, and a single series often cannot support them -- effusion reads off a
fluid-sensitive sequence, patellofemoral OA off a different one -- so the model
has to see the study's series together.

**Laterality is handled two ways, depending on the path.**

For the end-to-end path, `mirror_volume` reflects left knees onto the canonical
side at load time, along the axis the acquisition plane dictates: slice order
for sagittal, where medial-lateral is the slice axis, and columns for coronal
and axial, where it lies in-plane. Doing it at load time rather than in the
cache means side detection can improve without rebuilding 29 GB.

For the frozen path this is not available -- the embeddings are computed from
the volume, so the image can no longer be flipped. There the side travels to
the model as a covariate (`is_right`, `side_known`) instead. That still
addresses the original concern, which was that the model should not have to
*infer* the side from cues that are not reliably present; it is handed the
answer either way. Which of the two performs better is an experiment, not a
settled question.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src import config

# Plane -> which axis of a (slices, rows, cols) volume runs medial-to-lateral.
PLANE_MIRROR_AXIS = {
    "Sagittal": 0,   # medial-lateral is the slice axis
    "Coronal": 2,    # medial-lateral lies in-plane
    "Axial": 2,
}
DEFAULT_MIRROR_AXIS = 2

# Series metadata fed to the model as conditioning. Findings are
# sequence-specific, and these three are given in both train and test.
# `is_right` and `side_known` carry laterality as a covariate rather than
# normalising it away by mirroring. With precomputed frozen embeddings the
# image can no longer be flipped -- the features are already computed -- so the
# side is handed to the model instead, which addresses the original concern
# (the model should not have to *infer* the side) by a different route.
# Mirroring remains available for the end-to-end path and is an A/B to run,
# not a settled question.
META_COLUMNS = ["Fluid_Sensitive", "Fat_Suppression", "is_sagittal",
                "is_coronal", "is_axial", "is_right", "side_known"]


def mirror_volume(volume: np.ndarray, side: Optional[str],
                  plane: Optional[str]) -> np.ndarray:
    """Reflect a left knee onto the canonical side, along the correct axis."""
    if side is None or (isinstance(side, float) and np.isnan(side)):
        # Unknown side: leave as acquired. Guessing swaps medial with lateral
        # on four of the twelve findings.
        return volume
    if str(side).upper() == config.CANONICAL_SIDE:
        return volume

    axis = PLANE_MIRROR_AXIS.get(str(plane), DEFAULT_MIRROR_AXIS)
    return np.flip(volume, axis=axis).copy()


def series_metadata(row: pd.Series) -> np.ndarray:
    """Conditioning vector for one series.

    Fluid sensitivity and fat suppression are kept separate: the data
    description notes they are often correlated but not equivalent, so
    collapsing them would discard the cases where they disagree.
    """
    plane = str(row.get("Anatomical_Plane", ""))
    side = row.get("side_predicted")
    known = isinstance(side, str) and side.upper() in ("L", "R")
    return np.array([
        float(row.get("Fluid_Sensitive", 0) or 0),
        float(row.get("Fat_Suppression", 0) or 0),
        float(plane == "Sagittal"),
        float(plane == "Coronal"),
        float(plane == "Axial"),
        float(known and side.upper() == "R"),
        float(known),
    ], dtype=np.float32)


class StudyIndex:
    """Everything needed to assemble a study, resolved once up front.

    Built from frames rather than files so it can be constructed in a test
    without touching disk.
    """

    def __init__(self, manifest: pd.DataFrame, series_meta: pd.DataFrame,
                 labels: Optional[pd.DataFrame] = None,
                 sides: Optional[pd.DataFrame] = None,
                 max_series: Optional[int] = None):
        frame = manifest.merge(
            series_meta[["SeriesInstanceUID", "Fluid_Sensitive",
                         "Fat_Suppression", "Anatomical_Plane"]],
            on="SeriesInstanceUID", how="left")

        if sides is not None:
            frame = frame.merge(
                sides[["SeriesInstanceUID", "side_predicted"]],
                on="SeriesInstanceUID", how="left")
        else:
            frame["side_predicted"] = frame.get("laterality_resolved")

        # Drop series whose volume failed to build; their row in the cache is
        # zeros and would train the model on blank images.
        if "error" in frame.columns:
            frame = frame[frame["error"].isna()].reset_index(drop=True)

        self.series = frame
        self.max_series = max_series
        self.labels = labels

        self.studies: List[str] = sorted(frame[config.ID_COL].unique())
        self._by_study = {
            study: group.reset_index(drop=True)
            for study, group in frame.groupby(config.ID_COL)
        }

        if labels is not None:
            keyed = labels.set_index(config.ID_COL)
            self.studies = [s for s in self.studies if s in keyed.index]
            self._labels = keyed
        else:
            self._labels = None

    def __len__(self) -> int:
        return len(self.studies)

    def series_for(self, study: str) -> pd.DataFrame:
        """Series of a study, capped deterministically when a cap is set.

        The cap exists for the efficiency track: inference cost scales with
        series count, and a study with a dozen series contributes far more
        runtime than one with three. Selection is by position rather than at
        random so a prediction is reproducible.
        """
        rows = self._by_study[study]
        if self.max_series is not None and len(rows) > self.max_series:
            keep = np.linspace(0, len(rows) - 1, self.max_series).round().astype(int)
            rows = rows.iloc[np.unique(keep)].reset_index(drop=True)
        return rows

    def targets_for(self, study: str):
        """(targets, weights) for a study, or (None, None) when unlabelled."""
        if self._labels is None or study not in self._labels.index:
            return None, None
        row = self._labels.loc[study]
        targets = np.array([float(row[name]) for name in config.LABELS],
                           dtype=np.float32)
        weights = np.array(
            [float(row.get(f"{name}_weight", 1.0)) for name in config.LABELS],
            dtype=np.float32)
        return targets, weights


def load_study(index: StudyIndex, study: str, shards: Dict[str, np.ndarray]) -> Dict:
    """Assemble one study: mirrored volumes, conditioning, targets, weights."""
    rows = index.series_for(study)

    volumes, meta = [], []
    for _, row in rows.iterrows():
        shard = shards[row["cache_path"]]
        volume = np.asarray(shard[int(row["position"])])
        volume = mirror_volume(volume, row.get("side_predicted"),
                               row.get("Anatomical_Plane"))
        volumes.append(volume)
        meta.append(series_metadata(row))

    targets, weights = index.targets_for(study)
    return {
        config.ID_COL: study,
        "volumes": np.stack(volumes) if volumes else np.zeros(
            (0, config.CACHE_SLICES, config.CACHE_SIZE, config.CACHE_SIZE),
            dtype=np.uint8),
        "meta": np.stack(meta) if meta else np.zeros((0, len(META_COLUMNS)),
                                                     dtype=np.float32),
        "targets": targets,
        "weights": weights,
    }


def load_study_features(index: StudyIndex, study: str,
                        shards: Dict[str, np.ndarray]) -> Dict:
    """Assemble one study from precomputed slice embeddings.

    Mirroring is unavailable here -- the features were computed from the image
    and cannot be flipped after the fact -- so laterality reaches the model
    through `series_metadata` instead.
    """
    rows = index.series_for(study)

    features, meta = [], []
    for _, row in rows.iterrows():
        shard = shards[row["cache_path"]]
        features.append(np.asarray(shard[int(row["position"])], dtype=np.float32))
        meta.append(series_metadata(row))

    targets, weights = index.targets_for(study)
    return {
        config.ID_COL: study,
        "volumes": np.stack(features) if features else np.zeros((0, 1, 1),
                                                                dtype=np.float32),
        "meta": np.stack(meta) if meta else np.zeros((0, len(META_COLUMNS)),
                                                     dtype=np.float32),
        "targets": targets,
        "weights": weights,
    }


def collate_studies(batch: Sequence[Dict]) -> Dict:
    """Pad a batch of studies to the largest series count, with a mask.

    Series counts vary from one to a dozen, so the batch is padded and masked
    rather than truncated. The mask must reach the attention pooling: without
    it, padding rows are averaged in as if they were real series.
    """
    import torch

    longest = max(len(item["volumes"]) for item in batch)

    # Per-series shape is taken from the data rather than assumed, so the same
    # collate serves raw volumes (D, H, W) and cached embeddings (D, F).
    reference = next((item["volumes"] for item in batch if len(item["volumes"])),
                     None)
    if reference is None:
        raise ValueError("every study in the batch is empty")
    per_series = reference.shape[1:]
    dtype = reference.dtype

    volumes = np.zeros((len(batch), longest, *per_series), dtype=dtype)
    meta = np.zeros((len(batch), longest, len(META_COLUMNS)), dtype=np.float32)
    mask = np.zeros((len(batch), longest), dtype=bool)

    for i, item in enumerate(batch):
        count = len(item["volumes"])
        if count:
            volumes[i, :count] = item["volumes"]
            meta[i, :count] = item["meta"]
            mask[i, :count] = True

    out = {
        config.ID_COL: [item[config.ID_COL] for item in batch],
        "volumes": torch.from_numpy(volumes),
        "meta": torch.from_numpy(meta),
        "mask": torch.from_numpy(mask),
    }

    if batch[0]["targets"] is not None:
        out["targets"] = torch.from_numpy(
            np.stack([item["targets"] for item in batch]))
        out["weights"] = torch.from_numpy(
            np.stack([item["weights"] for item in batch]))
    return out
