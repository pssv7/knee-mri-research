from datetime import datetime, timedelta, timezone
import json

import numpy as np
import pandas as pd
import pytest

from src import config
from src.labels import audit_stats
from tools import audit_report as ar
from tools import build_blinded_packets as bp

ID = config.ID_COL
T0 = datetime(2026, 9, 25, 9, tzinfo=timezone.utc)
QUOTE = "tear of the lateral patellar facet cartilage"


def a_row(case, state, quote=QUOTE):
    return {"case_id": case, "finding": "PF OA", "report_states": state, "anatomical_location": "patella",
            "certainty": "definite", "severity_text": "", "severity_norm": "not stated",
            "evidence_quote": quote, "note": "", "ingested_at": T0.isoformat(), "reviewer": "participant",
            "source_sha256": "x"}


def b_row(case, disposition, target, correct, error_class="missed_mention", proposed="1.0"):
    return {"case_id": case, "finding": "PF OA", "extractor_correct": "no", "error_class": error_class,
            "disposition": disposition, "proposed_target": proposed if disposition == "correct_target" else "",
            "proposed_weight": "", "rationale": "x" if disposition != "keep" else "",
            "final_target_correct": correct, "old_target": target, "old_weight": 1.0,
            "ingested_at": T0.isoformat(), "reviewer": "participant", "source_sha256": "x"}


def write_batch(root, pass_no, batch, keys, a_rows, b_rows):
    p = bp.paths(root, pass_no, batch)
    for key in ("keys", "stage_a", "stage_b"):
        p[key].parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(keys).to_parquet(p["keys"], index=False)
    pd.DataFrame(a_rows).to_parquet(p["stage_a"], index=False)
    if b_rows is not None:
        pd.DataFrame(b_rows).to_parquet(p["stage_b"], index=False)


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(ar.sampling, "PRIORITY", ("PF OA",))
    uids = [f"u{i}" for i in range(8)]
    roles = ["control"] * 6 + ["suspected_false_positive"] * 2
    out = tmp_path / "outputs/label_quality/audit_v1"
    out.mkdir(parents=True)
    pd.DataFrame({ID: uids, "finding": "PF OA", "role": roles, "provenance": "random_draw",
                  "inclusion_probability": 6 / 100, "language": ["en", "el", "en", "tr", "en", "en", "en", "en"],
                  "scanner": "S", "weight_bucket": "1.0", "extractor_call": "no"}).to_parquet(
        out / "sample.parquet", index=False)
    (tmp_path / "docs/label_quality").mkdir(parents=True)
    (tmp_path / "docs/label_quality/manifest_v1.json").write_text(json.dumps(
        {"audit": {"pairs": {"PF OA | control | random_draw": 6}}, "secondary_queue_pairs": 0}))
    # Pass 1: controls u0-u2 read as present with target 0.2 (missed positives, all proposed),
    # u3-u5 negated with target 0.1 (correct, keep). Enriched u6-u7 wrong too, but must not count.
    keys1 = [{"case_id": f"K{i}", ID: u, "finding": "PF OA", "role": r, "pass": 1, "batch": "A"}
             for i, (u, r) in enumerate(zip(uids, roles))]
    a1 = [a_row(f"K{i}", "present" if i in (0, 1, 2, 6, 7) else "negated") for i in range(8)]
    b1 = [b_row(f"K{i}", "correct_target", 0.2, False) if i in (0, 1, 2, 6, 7)
          else b_row(f"K{i}", "keep", 0.1, True, error_class="none") for i in range(8)]
    write_batch(tmp_path, 1, "A", keys1, a1, b1)
    return tmp_path


def second_pass(root, agree_on=(0, 1), complete=True):
    """Re-review every control; the first two proposals agree, the third does not."""
    keys = [{"case_id": f"R{i}", ID: f"u{i}", "finding": "PF OA", "role": "control", "pass": 2, "batch": "R1"}
            for i in range(6 if complete else 2)]
    a = [a_row(k["case_id"], "present" if i in (0, 1, 2) else "negated") for i, k in enumerate(keys)]
    b = []
    for i, k in enumerate(keys):
        if i in agree_on:
            b.append(b_row(k["case_id"], "correct_target", 0.2, False))
        elif i == 2:
            b.append(b_row(k["case_id"], "keep", 0.2, False, error_class="unresolved"))
        else:
            b.append(b_row(k["case_id"], "keep", 0.1, True, error_class="none"))
    write_batch(root, 2, "R1", keys, a, b)


def test_control_rates_use_only_random_controls(root):
    summary, _ = ar.summarise(root, T0)
    c = summary["controls"]["PF OA"]
    assert c["with_stage_b"] == 6 and c["incorrect"]["k"] == 3 and c["missed_positive"]["k"] == 3
    assert c["false_positive"]["k"] == 0
    assert c["incorrect"]["lower"] == pytest.approx(audit_stats.wilson(3, 6)[0])
    assert summary["enriched"]["PF OA | suspected_false_positive"]["incorrect"] == 2


def test_undecided_until_proposals_are_re_reviewed(root):
    summary, _ = ar.summarise(root, T0)
    assert summary["controls"]["PF OA"]["awaiting_pass2"] == 3
    assert summary["decision"]["decision"] == "undecided"


def test_only_agreed_target_corrections_count(root, monkeypatch):
    second_pass(root)
    summary, _ = ar.summarise(root, T0 + timedelta(days=8))
    c = summary["controls"]["PF OA"]
    assert c["agreed_target_errors"] == 2 and c["decidable"]
    assert summary["decision"]["decision"] == "no-go"           # 2 < 5
    monkeypatch.setattr(ar, "THRESHOLD", 2)
    assert ar.summarise(root, T0 + timedelta(days=8))[0]["decision"] == {"decision": "go", "findings": ["PF OA"]}


def test_incomplete_after_the_deadline_is_audit_only(root):
    second_pass(root, complete=False)
    late = datetime(2026, 10, 11, 9, tzinfo=timezone.utc)
    assert ar.summarise(root, late)[0]["decision"]["decision"] == "audit-only"


def test_agreement_is_split_between_controls_and_proposed_changes(root):
    second_pass(root)
    summary, _ = ar.summarise(root, T0 + timedelta(days=8))
    controls = summary["agreement"]["controls"]
    assert controls["pairs"] == 6 and controls["state_raw"] == 1.0
    assert controls["disposition_raw"] == pytest.approx(5 / 6)
    assert "proposed_changes" not in summary["agreement"]      # enriched proposals not yet re-reviewed


def test_report_never_quotes_or_names_a_study(root):
    second_pass(root)
    summary, _ = ar.summarise(root, T0 + timedelta(days=8))
    text = ar.markdown(summary)
    assert QUOTE not in text and "u0" not in text
    assert "NO-GO" in text and "never a rate" in text


def test_wilson_interval_covers_about_95_percent():
    rng = np.random.default_rng(0)
    for p in (0.05, 0.1, 0.3):
        hits = 0
        for k in rng.binomial(50, p, size=3000):
            lo, hi = audit_stats.wilson(int(k), 50)
            hits += lo <= p <= hi
        assert 0.92 <= hits / 3000 <= 0.985


def test_kappa_and_go_no_go_edges():
    assert audit_stats.cohen_kappa("aabb", "aabb") == 1.0
    assert audit_stats.cohen_kappa("aaaa", "aaaa") is None
    assert audit_stats.cohen_kappa("ab", "ba") == pytest.approx(-1.0)
    decided = {"A": {"decidable": True, "agreed_target_errors": 5}, "B": {"decidable": False, "agreed_target_errors": 0}}
    assert audit_stats.go_no_go(decided)["decision"] == "go"
    pending = {"A": {"decidable": True, "agreed_target_errors": 4}, "B": {"decidable": False, "agreed_target_errors": 0}}
    assert audit_stats.go_no_go(pending)["decision"] == "undecided"
    assert audit_stats.go_no_go(pending, deadline_passed=True)["decision"] == "audit-only"
