from datetime import datetime, timedelta, timezone
import json

import numpy as np
import pandas as pd
import pytest

from src import config
from src.labels import translate
from tools import build_blinded_packets as bp
from tools import ingest_ledger

ID = config.ID_COL
T0 = datetime(2026, 9, 24, 9, tzinfo=timezone.utc)


def report_for(i):
    return f"Synthetic report {i}. Lateral meniscus: radial tear. Bone marrow normal. Patella cartilage grade I."


@pytest.fixture
def root(tmp_path):
    n = 12
    uids = [f"u{i:02}" for i in range(n)]
    meta = tmp_path / "data/meta"
    meta.mkdir(parents=True)
    pd.DataFrame({ID: uids, "Report": [report_for(i) for i in range(n)]}).to_csv(meta / "train.csv", index=False)
    labels = pd.DataFrame({ID: uids})
    calls = pd.DataFrame({ID: uids})
    probs = pd.DataFrame({ID: uids})
    for f in config.LABELS:
        labels[f], labels[f + "_weight"] = 0.123, 0.777
        calls[f], calls[f + "_severity"], calls[f + "_evidence"], calls[f + "_confidence"] = 0.0, "none", "", 0.5
        probs[f], probs[f + "_confidence"] = 0.456, 0.5
    labels.to_parquet(meta / "train_labels.parquet", index=False)
    calls.to_parquet(meta / "llm_labels.parquet", index=False)
    probs.to_parquet(meta / "llm_probs.parquet", index=False)
    roles = ["control"] * 6 + ["suspected_false_positive"] * 3 + ["suspected_missed_positive"] * 3
    sample = pd.DataFrame({ID: uids, "finding": "Lateral Meniscus", "role": roles,
                           "language": ["en"] * 9 + ["el"] * 3, "oof": 0.918, "target": 0.123,
                           "inclusion_probability": 0.0421})
    out = tmp_path / "outputs/label_quality/audit_v1"
    out.mkdir(parents=True)
    sample.to_parquet(out / "sample.parquet", index=False)
    return tmp_path


@pytest.fixture(autouse=True)
def small_batch(monkeypatch):
    monkeypatch.setattr(bp, "BATCH_A_ENRICHED", 2)


def translations_for(root, batch):
    p = bp.paths(root, 1, batch)
    requests = [json.loads(l) for l in p["requests"].read_text(encoding="utf-8").splitlines() if l]
    results = root / f"{batch}_translations.jsonl"
    results.write_text("".join(json.dumps({"case_id": r["case_id"], "translation": "EN " + r["report"],
                                           "finish_reason": "stop", "model": "stub"}) + "\n"
                               for r in requests), encoding="utf-8")
    return results


def fill_stage_a(root, batch, pass_no=1):
    sheet = pd.read_csv(bp.paths(root, pass_no, batch)["packet"] / "stage_a.csv", dtype=str, keep_default_na=False)
    sheet["report_states"] = "present"
    sheet["anatomical_location"] = "lateral meniscus"
    sheet["certainty"] = "definite"
    sheet["severity_norm"] = "not stated"
    sheet["evidence_quote"] = "radial tear"
    path = root / f"filled_a_{pass_no}_{batch}.csv"
    sheet.to_csv(path, index=False)
    return path


def fill_stage_b(root, batch, pass_no=1):
    sheet = pd.read_csv(bp.paths(root, pass_no, batch)["packet"] / "stage_b.csv", dtype=str, keep_default_na=False)
    sheet["extractor_correct"] = "no"
    sheet["error_class"] = "missed_mention"
    sheet["disposition"] = "correct_target"
    sheet["proposed_target"] = "1.0"
    sheet["rationale"] = "explicit tear"
    path = root / f"filled_b_{pass_no}_{batch}.csv"
    sheet.to_csv(path, index=False)
    return path


def test_batch_a_mixes_every_control_with_enriched_cases(root):
    keys, n_requests = bp.prepare_pass1(root, "A", T0)
    assert (keys.role == "control").sum() == 6 and (keys.role != "control").sum() == 2
    b, _ = bp.prepare_pass1(root, "B", T0)
    assert not set(keys[ID]) & set(b[ID]) and len(keys) + len(b) == 12
    assert keys.case_id.str.fullmatch(r"K[0-9A-F]{6}").all()


def test_prepared_batches_are_never_redrawn(root):
    bp.prepare_pass1(root, "A", T0)
    with pytest.raises(FileExistsError):
        bp.prepare_pass1(root, "A", T0)


def test_stage_a_packet_carries_only_the_allowed_fields(root):
    bp.prepare_pass1(root, "A", T0)
    bp.prepare_pass1(root, "B", T0)
    bp.add_translations(root, 1, "B", translations_for(root, "B"))
    bp.render(root, 1, "B")
    folder = bp.paths(root, 1, "B")["packet"]
    blob = "".join(p.read_text(encoding="utf-8") for p in folder.iterdir())
    for hidden in ["0.123", "0.777", "0.918", "0.456", "0.0421", "suspected", "control", "u0", "u1",
                   "missed_positive", "false_positive"]:
        assert hidden not in blob
    keys = pd.read_parquet(bp.paths(root, 1, "B")["keys"])
    assert all(c in blob for c in keys.case_id)
    assert not any(p.name.endswith(".parquet") for p in folder.iterdir())


def test_non_english_reports_need_a_translation_before_rendering(root):
    bp.prepare_pass1(root, "A", T0)
    keys, n_requests = bp.prepare_pass1(root, "B", T0)
    assert n_requests == (keys.language != "en").sum() > 0
    with pytest.raises(ValueError, match="no translation yet"):
        bp.render(root, 1, "B")


def test_translation_files_are_checked_before_use(root):
    bp.prepare_pass1(root, "A", T0)
    bp.prepare_pass1(root, "B", T0)
    bad = root / "bad.jsonl"
    bad.write_text(json.dumps({"case_id": "KXXXXXX", "translation": "x", "finish_reason": "stop"}) + "\n")
    with pytest.raises(ValueError, match="translation file rejected"):
        bp.add_translations(root, 1, "B", bad)


def test_reveal_comes_only_after_the_blind_reading(root):
    bp.prepare_pass1(root, "A", T0)
    bp.render(root, 1, "A", allow_missing=True)
    with pytest.raises(ValueError, match="stage A has not been ingested"):
        bp.reveal(root, 1, "A")
    ingest_ledger.ingest(root, 1, "A", "a", fill_stage_a(root, "A"), now=T0)
    bp.reveal(root, 1, "A")
    page = (bp.paths(root, 1, "A")["packet"] / "reveal.html").read_text(encoding="utf-8")
    assert "0.123" in page and "0.918" not in page and "control" not in page and "suspected" not in page


def test_ingestion_rejects_a_sheet_with_any_bad_row_and_never_overwrites(root, capsys):
    bp.prepare_pass1(root, "A", T0)
    bp.render(root, 1, "A", allow_missing=True)
    path = fill_stage_a(root, "A")
    sheet = pd.read_csv(path, dtype=str, keep_default_na=False)
    sheet.loc[0, "evidence_quote"] = "an invented quote"
    sheet.to_csv(path, index=False)
    with pytest.raises(ValueError, match="nothing ingested") as err:
        ingest_ledger.ingest(root, 1, "A", "a", path, now=T0)
    assert "an invented quote" not in str(err.value)
    assert not bp.paths(root, 1, "A")["stage_a"].exists()
    ingest_ledger.ingest(root, 1, "A", "a", fill_stage_a(root, "A"), now=T0)
    with pytest.raises(FileExistsError):
        ingest_ledger.ingest(root, 1, "A", "a", fill_stage_a(root, "A"), now=T0)


def test_ingestion_requires_every_case_exactly_once(root):
    bp.prepare_pass1(root, "A", T0)
    bp.render(root, 1, "A", allow_missing=True)
    path = fill_stage_a(root, "A")
    sheet = pd.read_csv(path, dtype=str, keep_default_na=False)
    sheet.iloc[1:].to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing from the sheet"):
        ingest_ledger.ingest(root, 1, "A", "a", path, now=T0)


def test_stage_b_computes_correctness_and_applies_the_rules(root):
    bp.prepare_pass1(root, "A", T0)
    bp.render(root, 1, "A", allow_missing=True)
    ingest_ledger.ingest(root, 1, "A", "a", fill_stage_a(root, "A"), now=T0)
    bp.reveal(root, 1, "A")
    frame = ingest_ledger.ingest(root, 1, "A", "b", fill_stage_b(root, "A"), now=T0)
    assert (frame.final_target_correct == False).all()  # noqa: E712 -- present reading, target 0.123
    assert (frame.old_target == 0.123).all()


def test_pass_two_waits_seven_days_and_hides_the_first_pass(root):
    bp.prepare_pass1(root, "A", T0)
    bp.render(root, 1, "A", allow_missing=True)
    ingest_ledger.ingest(root, 1, "A", "a", fill_stage_a(root, "A"), now=T0)
    bp.reveal(root, 1, "A")
    ingest_ledger.ingest(root, 1, "A", "b", fill_stage_b(root, "A"), now=T0)
    with pytest.raises(ValueError, match="nothing is due"):
        bp.prepare_pass2(root, "R1", T0 + timedelta(days=6))
    keys, _ = bp.prepare_pass2(root, "R1", T0 + timedelta(days=7))
    first = pd.read_parquet(bp.paths(root, 1, "A")["keys"])
    assert len(keys) == len(first)  # every control, and every proposed change
    assert not set(keys.case_id) & set(first.case_id)
    order_first = first.set_index(ID).loc[keys[ID]].case_id.tolist()
    assert keys.case_id.tolist() != order_first
    bp.render(root, 2, "R1", allow_missing=True)
    blob = (bp.paths(root, 2, "R1")["packet"] / "packet.html").read_text(encoding="utf-8")
    assert "radial tear" in blob and "missed_mention" not in blob and "explicit tear" not in blob


def test_pass_two_skips_pairs_already_re_reviewed(root):
    bp.prepare_pass1(root, "A", T0)
    bp.render(root, 1, "A", allow_missing=True)
    ingest_ledger.ingest(root, 1, "A", "a", fill_stage_a(root, "A"), now=T0)
    bp.reveal(root, 1, "A")
    ingest_ledger.ingest(root, 1, "A", "b", fill_stage_b(root, "A"), now=T0)
    bp.prepare_pass2(root, "R1", T0 + timedelta(days=8))
    with pytest.raises(ValueError, match="nothing is due"):
        bp.prepare_pass2(root, "R2", T0 + timedelta(days=9))


def test_report_hashes_bind_translations_to_their_report(root):
    keys, _ = bp.prepare_pass1(root, "A", T0)
    text = bp.reports(root)
    assert all(h == translate.report_hash(text[u]) for u, h in zip(keys[ID], keys.report_sha256))
