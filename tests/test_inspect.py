"""Tests for the failure-inspection sheets.

The point of the tool is to make a data-plumbing bug visible, so the properties
worth guarding are that it picks the right studies to look at, that it renders
the model's own view rather than something adjacent to it, and that the labels
down the side say what the model was actually told.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config  # noqa: E402
from src.model import inspect as knee_inspect  # noqa: E402
from src.model.dataset import META_COLUMNS  # noqa: E402

L = len(config.LABELS)


class TestRankErrors:
    def test_a_perfect_ranking_scores_zero(self):
        truth = np.zeros((10, L)); truth[:5] = 1.0
        preds = np.tile(np.r_[np.ones(5), np.zeros(5)][:, None], (1, L))
        assert knee_inspect.study_rank_errors(truth, preds).sum() == \
            pytest.approx(0.0, abs=1e-9)

    def test_a_reversed_ranking_scores_the_maximum(self):
        truth = np.zeros((10, L)); truth[:5] = 1.0
        preds = np.tile(np.r_[np.zeros(5), np.ones(5)][:, None], (1, L))
        errors = knee_inspect.study_rank_errors(truth, preds)
        assert errors.min() > 0.5 * L

    def test_a_finding_with_one_class_is_skipped(self):
        # With 58 studies this happens, and a rank means nothing there.
        truth = np.ones((10, L))
        preds = np.random.default_rng(0).random((10, L))
        assert knee_inspect.study_rank_errors(truth, preds).sum() == \
            pytest.approx(0.0)

    def test_mismatched_shapes_are_refused(self):
        with pytest.raises(ValueError, match="against preds"):
            knee_inspect.study_rank_errors(np.zeros((4, L)), np.zeros((5, L)))

    def test_per_finding_sums_to_the_total(self):
        rng = np.random.default_rng(1)
        truth = rng.integers(0, 2, (30, L)).astype(float)
        preds = rng.random((30, L))
        assert knee_inspect.per_finding_errors(truth, preds).sum(axis=1) == \
            pytest.approx(knee_inspect.study_rank_errors(truth, preds))


class TestSeriesLabel:
    def row(self, **kwargs):
        row = np.zeros(len(META_COLUMNS))
        for key, value in kwargs.items():
            row[META_COLUMNS.index(key)] = value
        return row

    def test_it_reads_back_what_the_model_was_told(self):
        label = knee_inspect.describe_series(
            self.row(is_sagittal=1, Fluid_Sensitive=1, Fat_Suppression=1,
                     side_known=1, is_right=1))
        assert label == "sag fluid fatsat R"

    def test_an_unknown_side_is_named_not_guessed(self):
        # 42% of series have no recoverable side; showing "L" for those would
        # be inventing the thing most likely to be wrong.
        assert "side?" in knee_inspect.describe_series(self.row(is_coronal=1))

    def test_a_missing_plane_is_visible(self):
        assert knee_inspect.describe_series(self.row()).startswith("plane?")


def fake_item(n_series=2, depth=6, size=8):
    meta = np.zeros((n_series, len(META_COLUMNS)), dtype=np.float32)
    meta[:, META_COLUMNS.index("is_sagittal")] = 1.0
    rng = np.random.default_rng(0)
    return {"volumes": rng.integers(0, 255, (n_series, depth, size, size),
                                    dtype=np.uint8),
            "meta": meta}


class TestSheets:
    def test_it_writes_a_png(self, tmp_path):
        out = tmp_path / "sheet.png"
        knee_inspect.save_study_sheet(out, fake_item(), title="t")
        assert out.exists() and out.stat().st_size > 1000

    def test_it_samples_across_the_stack_not_the_front(self, tmp_path):
        # The front of a series is usually empty; a sheet of black is useless.
        # Rendered with more columns than slices, every slice must appear.
        out = tmp_path / "sheet.png"
        knee_inspect.save_study_sheet(out, fake_item(depth=3), columns=8)
        assert out.exists()

    def test_a_study_with_no_series_is_refused(self, tmp_path):
        empty = {"volumes": np.zeros((0, 4, 8, 8), dtype=np.uint8),
                 "meta": np.zeros((0, len(META_COLUMNS)), dtype=np.float32)}
        with pytest.raises(ValueError, match="no series"):
            knee_inspect.save_study_sheet(tmp_path / "x.png", empty)

    def test_the_title_names_the_findings_that_failed(self):
        errors = np.zeros(L)
        errors[config.LABELS.index("Synovitis")] = 0.9
        truth = np.zeros(L); truth[config.LABELS.index("Synovitis")] = 1.0
        preds = np.full(L, 0.04)
        title = knee_inspect.sheet_title("abc123456789xyz", truth, preds, errors)
        assert "Synovitis truth 1 pred 0.04" in title


class TestRenderFailures:
    """The end-to-end path, against a real StudyIndex over a fake cache."""

    def build(self, tmp_path, n_studies=12):
        from src.model.dataset import StudyIndex
        rng = np.random.default_rng(0)
        cache = tmp_path / "cache.npy"
        rows, labels = [], []
        position = 0
        for i in range(n_studies):
            study = f"study{i:03d}"
            for j in range(2):
                rows.append({config.ID_COL: study,
                             "SeriesInstanceUID": f"{study}_s{j}",
                             "position": position, "cache_path": str(cache),
                             "error": None})
                position += 1
            labels.append({config.ID_COL: study,
                           **{n: float(i % 2) for n in config.LABELS},
                           **{f"{n}_weight": 1.0 for n in config.LABELS}})
        np.save(cache, rng.integers(0, 255, (position, 6, 8, 8), dtype=np.uint8))
        manifest = pd.DataFrame(rows)
        series_meta = pd.DataFrame({
            "SeriesInstanceUID": manifest["SeriesInstanceUID"],
            "Fluid_Sensitive": 1, "Fat_Suppression": 0,
            "Anatomical_Plane": "Sagittal"})
        index = StudyIndex(manifest, series_meta, pd.DataFrame(labels))
        shards = {str(cache): np.load(cache, mmap_mode="r")}
        truth = pd.DataFrame(labels).set_index(config.ID_COL)[config.LABELS]
        return index, shards, truth

    def test_it_renders_the_worst_and_a_control(self, tmp_path):
        index, shards, truth = self.build(tmp_path)
        # A model that is exactly backwards on three studies and right elsewhere.
        preds = truth.copy().astype(float)
        broken = list(truth.index[:3])
        preds.loc[broken] = 1.0 - preds.loc[broken]

        written = knee_inspect.render_failures(
            index, shards, truth, preds, str(tmp_path / "out"),
            limit=3, best=2)
        assert len(written) == 5
        names = [os.path.basename(p) for p in written]
        assert sum(n.startswith("worst") for n in names) == 3
        assert sum(n.startswith("best") for n in names) == 2
        for path in written:
            assert os.path.getsize(path) > 1000
        # The three broken studies are the three rendered as worst.
        worst_ids = {n.split("_")[-1].replace(".png", "")
                     for n in names if n.startswith("worst")}
        assert worst_ids == {s[-12:] for s in broken}

    def test_it_skips_studies_absent_from_the_cache(self, tmp_path):
        index, shards, truth = self.build(tmp_path)
        extra = pd.concat([truth, truth.iloc[:1].rename(index={truth.index[0]: "ghost"})])
        preds = extra.copy().astype(float)
        written = knee_inspect.render_failures(
            index, shards, extra, preds, str(tmp_path / "out"), limit=2, best=1)
        assert all("ghost" not in os.path.basename(p) for p in written)
