from datetime import datetime, timedelta, timezone
import json

import numpy as np
import pandas as pd
import pytest

from src import config
from src.labels import audit_sampling as sampling
from src.labels import audit_stats
from tools import audit_progress as ap
from tools import build_blinded_packets as bp

ID = config.ID_COL
T0 = datetime(2026, 9, 24, 9, tzinfo=timezone.utc)
SMALL = sampling.Settings(controls=3)


@pytest.mark.parametrize("k, n, lo, hi", [(0, 50, 0.0, 0.0714), (5, 50, 0.0435, 0.2136),
                                          (10, 50, 0.1124, 0.3304), (50, 50, 0.9286, 1.0)])
def test_wilson_matches_reference_values(k, n, lo, hi):
    got = audit_stats.wilson(k, n)
    assert got == pytest.approx((lo, hi), abs=5e-4)


def test_trigger_fires_only_above_width_020():
    assert not audit_stats.trigger(5, 50)["fires"]      # width ~0.170
    assert audit_stats.trigger(10, 50)["fires"]         # width ~0.218


def test_wilson_rejects_impossible_counts():
    with pytest.raises(ValueError):
        audit_stats.wilson(3, 2)


@pytest.fixture
def root(tmp_path, monkeypatch):
    uids = [f"u{i:02}" for i in range(20)]
    (tmp_path / "data/meta").mkdir(parents=True)
    pd.DataFrame({ID: uids, "Report": [f"Synthetic {i}." for i in range(20)]}).to_csv(
        tmp_path / "data/meta/train.csv", index=False)
    out = tmp_path / "outputs/label_quality/audit_v1"
    out.mkdir(parents=True)
    sample = pd.DataFrame({ID: uids[:6], "finding": "PF OA",
                           "role": ["control"] * 4 + ["suspected_false_positive"] * 2,
                           "language": "en"})
    sample.to_parquet(out / "sample.parquet", index=False)
    pd.DataFrame({ID: ["u19"], "finding": ["PF OA"]}).to_parquet(out / "reserve.parquet", index=False)
    manifest = {"audit": {"pairs": {"PF OA | control | random_draw": 4},
                          "population_pairs": {"PF OA": 19}}}
    (tmp_path / "docs/label_quality").mkdir(parents=True)
    (tmp_path / "docs/label_quality/manifest_v1.json").write_text(json.dumps(manifest))
    pop = pd.DataFrame({ID: uids, "finding": "PF OA", "fold": 0, "target": 0.3, "weight": 1.0,
                        "oof": 0.5, "language": "en"})
    monkeypatch.setattr(ap.audit, "load_pairs", lambda r: ({"PF OA": pop}, set()))
    monkeypatch.setattr(ap.baseline, "verify", lambda r, v: ([], []))
    return tmp_path


def keys_with_stage_b(root, correct, batch="A"):
    """Four pass-1 controls plus two enriched cases, with a stage B carrying `correct`."""
    sample = pd.read_parquet(root / ap.audit.OUT / "sample.parquet")
    keys = sample.assign(case_id=[f"K{i:06}" for i in range(len(sample))], **{"pass": 1}, batch=batch,
                         prepared_at=T0.isoformat(), report_sha256="x")
    p = bp.paths(root, 1, batch)
    p["keys"].parent.mkdir(parents=True, exist_ok=True)
    keys.to_parquet(p["keys"], index=False)
    p["stage_a"].parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"case_id": keys.case_id, "ingested_at": T0.isoformat()}).to_parquet(p["stage_a"], index=False)
    pd.DataFrame({"case_id": keys.case_id, "final_target_correct": correct,
                  "disposition": "keep", "ingested_at": T0.isoformat()}).to_parquet(p["stage_b"], index=False)
    return keys


def test_assessment_waits_for_every_original_control(root):
    sample = pd.read_parquet(root / ap.audit.OUT / "sample.parquet")
    keys_with_stage_b(root, [True] * 6)
    p = bp.paths(root, 1, "A")
    pd.read_parquet(p["stage_b"]).iloc[1:].to_parquet(p["stage_b"], index=False)
    assert ap.assess(root, T0) == {}


def test_assessment_is_stored_once_and_never_recomputed(root):
    keys_with_stage_b(root, [False, False, True, True, True, True])
    first = ap.assess(root, T0)["PF OA"]
    assert first["errors"] == 2 and first["controls"] == 4 and first["fires"]
    p = bp.paths(root, 1, "A")
    pd.read_parquet(p["stage_b"]).assign(final_target_correct=True).to_parquet(p["stage_b"], index=False)
    assert ap.assess(root, T0 + timedelta(days=1))["PF OA"] == first


def test_unjudged_readings_are_not_counted_as_errors(root):
    keys_with_stage_b(root, [None, None, False, True, True, True])
    assert ap.assess(root, T0)["PF OA"]["errors"] == 1


def test_no_extra_draw_without_a_firing_assessment(root, monkeypatch):
    with pytest.raises(ValueError, match="not assessed"):
        ap.draw_extra(root, "PF OA", T0, SMALL)
    # Four synthetic controls can't give an interval under 0.20 wide (0/4 spans ~0.49),
    # so widen the threshold to exercise the not-firing branch.
    original = audit_stats.trigger
    monkeypatch.setattr(ap.audit_stats, "trigger", lambda k, n: original(k, n, width=0.6))
    keys_with_stage_b(root, [True] * 6)
    with pytest.raises(ValueError, match="within 0.20|no extra draw"):
        ap.draw_extra(root, "PF OA", T0, SMALL)


def test_extra_draw_is_uniform_outside_the_original_controls_and_happens_once(root):
    keys_with_stage_b(root, [False, False, True, True, True, True])
    chosen = ap.draw_extra(root, "PF OA", T0, SMALL)
    assert len(chosen) == 3 and not set(chosen[ID]) & {"u00", "u01", "u02", "u03"} and "u19" not in set(chosen[ID])
    assert np.allclose(chosen.inclusion_probability, (4 + 3) / 19)
    record = json.loads((root / ap.EXTRA_RECORD).read_text())
    assert record["PF OA"]["pairs"] == 3
    with pytest.raises(FileExistsError):
        ap.draw_extra(root, "PF OA", T0, SMALL)


def test_extra_draw_refuses_a_changed_population(root, monkeypatch):
    keys_with_stage_b(root, [False, False, True, True, True, True])
    manifest = json.loads((root / ap.audit.MANIFEST).read_text())
    manifest["audit"]["population_pairs"]["PF OA"] = 25
    (root / ap.audit.MANIFEST).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="no longer matches"):
        ap.draw_extra(root, "PF OA", T0, SMALL)


def test_already_reviewed_extra_controls_skip_the_new_batch(root):
    keys_with_stage_b(root, [False, False, True, True, True, True])
    chosen = ap.draw_extra(root, "PF OA", T0, SMALL)
    pending = bp.pending_extra_controls(root)
    assert set(pending[ID]) == set(chosen[~chosen.already_reviewed][ID])


def test_status_reports_the_re_review_date(root):
    keys_with_stage_b(root, [True] * 6)
    table = ap.status(root)
    assert table.iloc[0].re_review_due == (T0 + timedelta(days=7)).date().isoformat()
