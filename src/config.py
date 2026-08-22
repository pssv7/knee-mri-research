"""Central configuration: label names, cache geometry, and environment paths.

Everything here is imported by both the training pipeline and the submission
notebook, so it must stay dependency-free apart from the standard library.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
# Exact submission spelling. The apostrophe in "Baker's" and the spaces are
# part of the column names -- do not normalise them anywhere that touches a
# submission file.
LABELS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

ID_COL = "StudyInstanceUID"
SUBMISSION_COLUMNS = [ID_COL] + LABELS

# Compartment-specific labels. A horizontal flip swaps medial and lateral, so
# any augmentation that mirrors the image must also swap these pairs.
LATERAL_LABEL_PAIRS = [
    ("Medial Meniscus", "Lateral Meniscus"),
    ("Medial OA", "Lateral OA"),
]

# ---------------------------------------------------------------------------
# Cache geometry
# ---------------------------------------------------------------------------
# Kept as constants so a higher-resolution variant is a config change rather
# than a rewrite. Sizing: 224 * 224 * 24 bytes ~= 1.2 MB/series,
# x 24,371 series ~= 30 GB, which fits Kaggle's 200 GB private dataset limit.
CACHE_SIZE = 224          # in-plane resolution after resampling
CACHE_SLICES = 24         # slices sampled per series, centred on the volume
CACHE_DTYPE = "uint8"

# Robust intensity normalisation percentiles. MRI has no standardised units,
# so every volume is scaled against its own distribution.
CLIP_PCT_LOW = 1.0
CLIP_PCT_HIGH = 99.0

# Canonical laterality: left knees are mirrored to match right knees so the
# medial compartment always falls on the same side of the image.
CANONICAL_SIDE = "R"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
COMPETITION = "rsna-knee-abnormality-detection"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# What a competition data root must contain to be recognised.
_MARKERS = ("train_series", "test_series")


def _detect_data_root() -> Path:
    """Locate the competition data across Kaggle, Colab, and local runs.

    An explicit KNEE_DATA_ROOT wins. Otherwise Kaggle's input directory is
    searched by content rather than by name: the mount is not reliably at
    /kaggle/input/<competition-slug>, and guessing the name silently falls
    through to the local fallback, which then fails much later with a confusing
    path in the message.
    """
    env = os.environ.get("KNEE_DATA_ROOT")
    if env:
        return Path(env)

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        exact = kaggle_input / COMPETITION
        if exact.exists():
            return exact
        # Search two levels: Kaggle mounts competition data under
        # /kaggle/input/competitions/<slug>/ as well as directly under
        # /kaggle/input/<slug>/, and which one you get is not predictable.
        try:
            for child in sorted(kaggle_input.iterdir()):
                if not child.is_dir():
                    continue
                if any((child / m).exists() for m in _MARKERS):
                    return child
                for grandchild in sorted(child.iterdir()):
                    if grandchild.is_dir() and any(
                            (grandchild / m).exists() for m in _MARKERS):
                        return grandchild
        except OSError:
            pass

    return PROJECT_ROOT / "data"


DATA_ROOT = _detect_data_root()
META_DIR = PROJECT_ROOT / "data" / "meta"
CACHE_DIR = Path(os.environ.get("KNEE_CACHE_DIR", PROJECT_ROOT / "data" / "cache"))

TRAIN_CSV = META_DIR / "train.csv"
TRAIN_SERIES_CSV = META_DIR / "train_series.csv"
TEST_CSV = META_DIR / "test.csv"
TEST_SERIES_CSV = META_DIR / "test_series.csv"
SAMPLE_SUBMISSION_CSV = META_DIR / "sample_submission.csv"

INDEX_PARQUET = META_DIR / "index.parquet"

ON_KAGGLE = Path("/kaggle").exists()
