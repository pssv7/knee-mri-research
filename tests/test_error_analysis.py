"""Tests for the error-analysis tool.

The tool exists to answer a question the macro cannot: is the model uniformly
mediocre, or is it losing a definable slice of studies? So the properties worth
guarding are that it FINDS a planted subgroup effect, that it names the right
studies as worst, and that it refuses to report a subgroup too small to mean
anything -- the failure mode being an eye-catching 0.55 computed from six
studies.
"""

import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import config  # noqa: E402
from tools import error_analysis as ea  # noqa: E402


def synthetic(n=200, seed=0, broken_when_few_series=False):
    """Studies, truth, predictions and a series table, with an optional flaw."""
    rng = np.random.default_rng(seed)
    ids = [f"study{i:04d}" for i in range(n)]
    n_series = rng.integers(3, 9, size=n)

    truth = pd.DataFrame(
        {name: rng.integers(0, 2, size=n).astype(float) for name in config.LABELS},
        index=pd.Index(ids, name=config.ID_COL))

    # A good model: score = truth plus noise, so AUC is high everywhere.
    preds = {}
    for name in config.LABELS:
        signal = truth[name].to_numpy() + rng.normal(0, 0.35, size=n)
        if broken_when_few_series:
            # Planted flaw: studies with under five series are scored at random.
            broken = n_series < 5
            signal = np.where(broken, rng.normal(0, 1, size=n), signal)
        preds[name] = signal
    preds = pd.DataFrame(preds, index=truth.index)

    rows = []
    for study, count in zip(ids, n_series):
        for j in range(count):
            rows.append({config.ID_COL: study,
                         "SeriesInstanceUID": f"{study}_s{j}",
                         "Fluid_Sensitive": int(j % 2),
                         "Fat_Suppression": int(j % 3 == 0),
                         "Anatomical_Plane": ["Sagittal", "Coronal", "Axial"][j % 3]})
    return truth, preds, pd.DataFrame(rows)


class TestAucSe:
    def test_it_reproduces_the_plan_s_worked_number(self):
        # The plan computes SE ~= 0.061 at AUC 0.86 with 17 positives and 41
        # negatives -- the calculation behind "58 studies cannot resolve 0.015".
        assert ea.auc_se(0.86, 17, 41) == pytest.approx(0.061, abs=0.002)

    def test_it_shrinks_as_the_sample_grows(self):
        assert ea.auc_se(0.86, 170, 410) < ea.auc_se(0.86, 17, 41)

    def test_a_degenerate_split_is_nan(self):
        assert np.isnan(ea.auc_se(0.86, 0, 41))
        assert np.isnan(ea.auc_se(float("nan"), 17, 41))


class TestAuc:
    def test_it_matches_the_training_loop(self):
        from src.model import lightning as knee_lightning
        rng = np.random.default_rng(1)
        y, s = rng.integers(0, 2, 300), rng.random(300)
        assert ea.auc(y, s) == pytest.approx(knee_lightning.auc(y, s))

    def test_a_single_class_is_nan(self):
        assert np.isnan(ea.auc(np.ones(10), np.arange(10)))


class TestStudyFeatures:
    def test_it_counts_series_and_planes(self, tmp_path):
        _, _, series = synthetic(n=5)
        path = tmp_path / "s.csv"
        series.to_csv(path, index=False)
        features = ea.study_features(path)
        assert features["n_series"].sum() == len(series)
        assert (features["n_sagittal"] + features["n_coronal"]
                + features["n_axial"]).equals(features["n_series"])


class TestSubgroups:
    def test_a_planted_flaw_shows_up_in_the_right_bucket(self, tmp_path):
        truth, preds, series = synthetic(n=400, broken_when_few_series=True)
        path = tmp_path / "s.csv"; series.to_csv(path, index=False)
        features = ea.study_features(path)
        table = ea.subgroup_table(truth, preds, features, "n_series",
                                  [0, 5, 99])
        below, above = table.loc["[0,5)", "macro"], table.loc["[5,99)", "macro"]
        assert below < above - 0.15, (below, above)

    def test_a_healthy_model_shows_no_subgroup_gap(self, tmp_path):
        truth, preds, series = synthetic(n=400, broken_when_few_series=False)
        path = tmp_path / "s.csv"; series.to_csv(path, index=False)
        features = ea.study_features(path)
        table = ea.subgroup_table(truth, preds, features, "n_series", [0, 5, 99])
        gap = abs(table.loc["[0,5)", "macro"] - table.loc["[5,99)", "macro"])
        assert gap < 0.05

    def test_a_bucket_too_small_to_mean_anything_is_refused(self, tmp_path):
        # The failure this guards: an eye-catching 0.55 computed from six
        # studies, reported next to numbers from four hundred.
        truth, preds, series = synthetic(n=400)
        path = tmp_path / "s.csv"; series.to_csv(path, index=False)
        features = ea.study_features(path)
        table = ea.subgroup_table(truth, preds, features, "n_series",
                                  [0, 1, 99])
        assert np.isnan(table.loc["[0,1)", "macro"])
        assert table.loc["[0,1)", "n"] == 0


class TestWorstStudies:
    def test_it_names_the_studies_that_are_actually_wrong(self, tmp_path):
        truth, preds, series = synthetic(n=200)
        path = tmp_path / "s.csv"; series.to_csv(path, index=False)
        features = ea.study_features(path)

        # Corrupt three studies: every finding predicted exactly backwards.
        victims = list(truth.index[:3])
        for name in config.LABELS:
            preds.loc[victims, name] = np.where(
                truth.loc[victims, name] == 1, -9.0, 9.0)

        worst = ea.worst_studies(truth, preds, features, limit=5)
        assert set(victims) <= set(worst.index)

    def test_it_reports_which_finding_drove_each_failure(self, tmp_path):
        truth, preds, series = synthetic(n=200)
        path = tmp_path / "s.csv"; series.to_csv(path, index=False)
        features = ea.study_features(path)
        victim = truth.index[7]
        preds.loc[victim, "Synovitis"] = -9.0 if truth.loc[victim, "Synovitis"] == 1 else 9.0
        worst = ea.worst_studies(truth, preds, features, limit=200)
        assert worst.loc[victim, "worst_finding"] == "Synovitis"


class TestCommandLine:
    def _write(self, tmp_path):
        truth, preds, series = synthetic(n=120, broken_when_few_series=True)
        meta = tmp_path / "meta"; meta.mkdir()
        gold = truth.reset_index()
        gold["Report"] = "x"
        gold.to_csv(meta / "train.csv", index=False)
        series.to_csv(meta / "train_series.csv", index=False)
        out = preds.reset_index()
        out.to_parquet(tmp_path / "preds.parquet", index=False)
        return meta, tmp_path / "preds.parquet"

    def test_it_runs_end_to_end(self, tmp_path):
        meta, preds = self._write(tmp_path)
        result = subprocess.run(
            [sys.executable, os.path.join(ROOT, "tools", "error_analysis.py"),
             "--predictions", str(preds), "--truth", "gold",
             "--meta-dir", str(meta)],
            capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert "per finding" in result.stdout
        assert "macro by study descriptor" in result.stdout
        assert "worst" in result.stdout
        # Every AUC carries its standard error, so nobody reads a gap without it.
        assert "+/- se" in result.stdout

    def test_mismatched_inputs_fail_loudly(self, tmp_path):
        meta, _ = self._write(tmp_path)
        other = pd.DataFrame({config.ID_COL: ["nobody"],
                              **{n: [0.5] for n in config.LABELS}})
        path = tmp_path / "other.parquet"
        other.to_parquet(path, index=False)
        result = subprocess.run(
            [sys.executable, os.path.join(ROOT, "tools", "error_analysis.py"),
             "--predictions", str(path), "--truth", "gold",
             "--meta-dir", str(meta)],
            capture_output=True, text=True)
        assert result.returncode != 0
        assert "no studies in common" in result.stdout + result.stderr
