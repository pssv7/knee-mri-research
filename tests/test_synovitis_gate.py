"""Tests for the Stage 0 gate tool.

The tool's job is to keep a decision honest, so the properties guarded are the
ones that would let it be read the wrong way: the pre-registered cutoffs must
not drift, a gated score must not pass the gate, and the tool must refuse the
one combination that would quietly blur the strata -- gold stratification
against mined labels.
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
from tools import synovitis_gate as gate  # noqa: E402


def synthetic(tmp_path, n=120, seed=0, gated=True):
    """Truth and predictions where synovitis is, or is not, only effusion."""
    rng = np.random.default_rng(seed)
    ids = [f"study{i:04d}" for i in range(n)]
    effusion = rng.integers(0, 2, size=n)
    synovitis = np.where(rng.random(n) < np.where(effusion == 1, 0.7, 0.3), 1, 0)

    truth = pd.DataFrame({name: rng.integers(0, 2, size=n).astype(float)
                          for name in config.LABELS},
                         index=pd.Index(ids, name=config.ID_COL))
    truth["Effusion"] = effusion.astype(float)
    truth["Synovitis"] = synovitis.astype(float)

    effusion_score = 0.5 + 0.4 * effusion + rng.normal(0, 0.02, size=n)
    preds = pd.DataFrame({name: rng.random(n) for name in config.LABELS},
                         index=truth.index)
    preds["Effusion"] = effusion_score
    preds["Synovitis"] = (effusion_score if gated else
                          np.clip(0.2 + 0.5 * synovitis
                                  + rng.normal(0, 0.08, size=n), 0.01, 0.99))

    meta = tmp_path / "meta"
    meta.mkdir(exist_ok=True)
    frame = truth.reset_index()
    frame.to_csv(meta / "train.csv", index=False)
    path = tmp_path / ("gated.parquet" if gated else "real.parquet")
    preds.reset_index().to_parquet(path, index=False)
    return meta, path


class TestTheGateCutoffs:
    """Fixed in code so they cannot be chosen after seeing a number."""

    def test_a_confident_interval_above_chance_is_clear(self):
        assert gate.gate_reading({"auc": 0.78, "lo": 0.61, "hi": 0.90,
                                  "p_at_or_below_chance": 0.001}) == "clear"

    def test_sitting_on_chance_is_chance(self):
        assert gate.gate_reading({"auc": 0.51, "lo": 0.38, "hi": 0.64,
                                  "p_at_or_below_chance": 0.44}) == "chance"

    def test_the_reference_run_is_ambiguous(self):
        # 0.630, CI [0.464, 0.786], P(<= 0.5) = 0.06. Expected at n=58, and
        # the reason the plan gates on it rather than acting on it.
        assert gate.gate_reading({"auc": 0.630, "lo": 0.464, "hi": 0.786,
                                  "p_at_or_below_chance": 0.060}) == "ambiguous"

    def test_a_barely_significant_interval_is_not_promoted_to_clear(self):
        assert gate.gate_reading({"auc": 0.62, "lo": 0.505, "hi": 0.74,
                                  "p_at_or_below_chance": 0.06}) == "ambiguous"

    def test_every_reading_names_an_action(self):
        for reading in ("clear", "chance", "ambiguous"):
            why, action = gate.GATE_ACTIONS[reading]
            assert why and action

    def test_the_chance_action_names_both_branches_the_plan_gives(self):
        # §4 says Stage 3, §10 says Stage 4. The conflict is surfaced, not
        # silently resolved by the tool.
        _, action = gate.GATE_ACTIONS["chance"]
        assert "Stage 3" in action and "Stage 4" in action


class TestItRefusesTheUnsafeCombination:
    def test_gold_strata_against_mined_labels(self, tmp_path):
        meta, path = synthetic(tmp_path)
        with pytest.raises(SystemExit):
            gate.main(["--predictions", str(path), "--truth", "mined",
                       "--stratify-on", "gold", "--meta-dir", str(meta)])


class TestEndToEnd:
    def _run(self, meta, path, extra=()):
        result = subprocess.run(
            [sys.executable, os.path.join(ROOT, "tools", "synovitis_gate.py"),
             "--predictions", str(path), "--meta-dir", str(meta),
             "--draws", "400", *extra],
            capture_output=True, text=True, cwd=ROOT)
        assert result.returncode == 0, result.stderr
        return result.stdout

    def test_a_gated_model_does_not_pass_the_gate(self, tmp_path):
        meta, path = synthetic(tmp_path, gated=True)
        output = self._run(meta, path)
        assert "CLEAR" not in output
        assert "conditional" in output

    def test_a_real_signal_passes(self, tmp_path):
        meta, path = synthetic(tmp_path, n=300, gated=False)
        assert "CLEAR" in self._run(meta, path)

    def test_it_prints_pooled_beside_conditional_and_never_macro(self, tmp_path):
        meta, path = synthetic(tmp_path)
        output = self._run(meta, path)
        assert "pooled" in output
        assert "macro" in output.lower()   # only as the standing rule against it
        assert "macro AUC" not in output

    def test_a_paired_comparison_reports_both_metrics(self, tmp_path):
        meta, gated = synthetic(tmp_path, gated=True)
        _, real = synthetic(tmp_path, gated=False)
        output = self._run(meta, real, ("--baseline", str(gated)))
        assert "paired against" in output
        assert output.count("P(not better)") == 2
