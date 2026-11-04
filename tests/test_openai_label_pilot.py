import copy
import json

import pandas as pd
import pytest

from src.labels.openai_extract import FIELDS, schema, request_body, parse_response, validate
from src.validation.proxy import ID, LABELS
from tools import openai_label_pilot as pilot


def assertions(report="The ACL is intact."):
    result = {name: {key: "unmentioned" if key == "report_states" else "" for key in FIELDS} for name in LABELS}
    result["ACL"] = dict(report_states="negated", anatomical_location="ACL", certainty="definite",
                         severity_text="", severity_norm="not stated", evidence_quote=report)
    return result


def response(report="The ACL is intact."):
    return {"status": "completed", "model": "gpt-6-sol", "usage": {"input_tokens": 3000, "output_tokens": 1800},
            "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(assertions(report))}]}]}


def test_request_keeps_silence_distinct_and_does_not_include_baseline_labels():
    body = request_body("The ACL is intact.", "Research rubric")
    assert body["store"] is False and "tools" not in body
    assert body["model"] == "gpt-6-sol"
    assert body["text"]["format"]["strict"] is True
    assert set(schema()["required"]) == set(LABELS)
    assert "unmentioned" in schema()["properties"]["ACL"]["properties"]["report_states"]["enum"]
    assert set(parse_response(response(), "The ACL is intact.")) == set(LABELS)
    assert "StudyInstanceUID" not in json.dumps(body)


@pytest.mark.parametrize("defect", ["missing", "extra", "quote", "silence", "anatomy", "certainty"])
def test_semantic_validation_rejects_unsupported_or_malformed_outputs(defect):
    result = assertions()
    if defect == "missing":
        del result["MCL"]
    elif defect == "extra":
        result["PCL"] = result["ACL"]
    elif defect == "quote":
        result["ACL"]["evidence_quote"] = "ACL torn"
    elif defect == "silence":
        result["MCL"]["evidence_quote"] = "The ACL is intact."
    elif defect == "anatomy":
        result["ACL"].update(report_states="present", anatomical_location="")
    else:
        result["ACL"]["certainty"] = ""
    with pytest.raises(ValueError):
        validate(result, "The ACL is intact.")


@pytest.mark.parametrize("bad", [
    {"status": "incomplete", "output": []},
    {"status": "completed", "output": [{"type": "message", "content": [{"type": "refusal"}]}]},
    {"status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": '{"ACL":{},"ACL":{}}'}]}]},
])
def test_incomplete_refused_or_duplicate_json_is_not_accepted(bad):
    with pytest.raises(ValueError):
        parse_response(bad, "The ACL is intact.")


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    settings = json.loads((pilot.ROOT / "configs/openai_label_pilot.json").read_text())
    settings["pilot_studies"] = 2
    monkeypatch.setattr(pilot, "ROOT", tmp_path)
    monkeypatch.setattr(pilot, "SOURCES", [])
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/report_review_rubric.md").write_text("Research fixture rubric")
    (tmp_path / "config.json").write_text(json.dumps(settings))
    uids = ["a", "b", "selection", "outer", "gold"]
    reports = pd.DataFrame({ID: uids, "Report": [f"The ACL is intact. Case {i}." for i in range(5)],
                            **{name: [None, None, None, None, 1] for name in LABELS}})
    reports.to_csv(tmp_path / "reports.csv", index=False)
    pd.DataFrame({ID: uids, "component": uids, "role": ["train", "train", "selection", "outer", "excluded"]}).to_csv(tmp_path / "roles.csv", index=False)
    pd.DataFrame({ID: uids, **{name: [0.] * 5 for label in LABELS for name in [label, label + "_weight"]}}).to_parquet(tmp_path / "labels.parquet")
    out = tmp_path / "pilot"
    pilot.prepare(out, tmp_path / "config.json", tmp_path / "reports.csv", tmp_path / "roles.csv", tmp_path / "labels.parquet")
    monkeypatch.setenv("OPENAI_API_KEY", "test-fixture-never-sent")
    return out


def test_preparation_excludes_selection_outer_gold_and_is_blind(prepared):
    manifest = json.loads((prepared / "pilot.json").read_text())
    assert {row[ID] for row in manifest["studies"]} == {"a", "b"}
    assert manifest["api_calls_made"] == 0
    review = [json.loads(line) for line in (prepared / "blind_review.jsonl").read_text().splitlines()]
    assert len(review) == 24 and all(row["report_states"] == "" for row in review)
    assert all("target" not in row and "weight" not in row for row in review)


def test_budget_gate_sends_nothing_and_resume_never_double_bills(prepared):
    calls = []

    def send(body, key):
        calls.append(body)
        return response(body["input"][0]["content"][0]["text"])

    assert pilot.run(prepared, .00001, send)["status"] == "budget_limit"
    assert not calls
    result = pilot.run(prepared, 1., send)
    assert len(calls) == 2 and result["accepted"] == 2
    assert result["charged_usd_upper"] == pytest.approx(2 * (.0075 + .018))
    assert pilot.run(prepared, 1., send)["status"] == "extraction_completed"
    assert len(calls) == 2


def test_failed_call_keeps_reservation_and_is_not_silently_retried(prepared):
    calls = []

    def send(body, key):
        calls.append(body)
        raise RuntimeError("simulated timeout")

    with pytest.raises(RuntimeError, match="timeout"):
        pilot.run(prepared, 1., send)
    record = json.loads(next((prepared / "attempts").glob("*.json")).read_text())
    assert record["status"] == "needs_review" and record["charged_usd_upper"] == record["reserved_usd"]
    with pytest.raises(RuntimeError, match="earlier request"):
        pilot.run(prepared, 1., send)
    assert len(calls) == 1


def test_changed_reports_or_request_provenance_block_execution(prepared):
    source = prepared.parent / "reports.csv"
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="input bytes changed"):
        pilot.run(prepared, 1., lambda *a: pytest.fail("Must not contact API"))
