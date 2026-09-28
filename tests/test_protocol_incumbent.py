"""Fixed recipe and held-out alignment checks for the incumbent comparison."""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from tools.compare_protocol_incumbent import fixed_predictions, completed_protocol_directory


def family(value):
    frame = pd.DataFrame(value, index=["b", "a"], columns=config.LABELS)
    frame["fold"] = [3, 1]
    return frame


def test_fixed_equal_family_weights_and_index_alignment():
    frames = [family(v) for v in (.1, .3, .8)]
    frames[1] = frames[1].iloc[::-1]
    originals = [f.copy() for f in frames]
    baseline, candidate = fixed_predictions(frames)
    assert baseline.index.tolist() == ["a", "b"]
    assert np.allclose(baseline[config.LABELS], .2)
    assert np.allclose(candidate[config.LABELS], .4)
    assert baseline.fold.tolist() == candidate.fold.tolist() == [1, 3]
    for actual, expected in zip(frames, originals):
        pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize("problem", ["missing", "duplicate", "fold", "nan", "range"])
def test_rejects_unaligned_or_invalid_predictions(problem):
    frames = [family(.1), family(.3), family(.8)]
    if problem == "missing":
        frames[2] = frames[2].iloc[:1]
    elif problem == "duplicate":
        frames[2].index = ["a", "a"]
    elif problem == "fold":
        frames[2].loc["a", "fold"] = 0
    else:
        frames[2].loc["a", config.LABELS[0]] = np.nan if problem == "nan" else 1.1
    with pytest.raises(ValueError):
        fixed_predictions(frames)


@pytest.mark.parametrize("fold", range(5))
def test_protocol_result_requires_completed_receipt(tmp_path, fold):
    suffix = "results" if fold in (1, 3) else "comparison_download"
    root = tmp_path / f"outputs/kaggle_protocol_fold{fold}_{suffix}"
    root.mkdir(parents=True)
    receipt = root / ("pair_progress.json" if fold in (1, 3) else "completion_progress.json")
    progress = {"state": "failed", "fold": fold, "completed": [f"protocol_fold{fold}"]}
    receipt.write_text(json.dumps(progress))
    with pytest.raises(ValueError, match="not completed"):
        completed_protocol_directory(tmp_path, fold)
    progress["state"] = "complete"
    receipt.write_text(json.dumps(progress))
    assert completed_protocol_directory(tmp_path, fold) == root / f"protocol_fold{fold}"
    if fold not in (1, 3):
        progress["completed"] = ["ddp_smoke"]
        receipt.write_text(json.dumps(progress))
        with pytest.raises(ValueError, match="unfinished fold"):
            completed_protocol_directory(tmp_path, fold)
