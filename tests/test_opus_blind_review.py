import json

import pytest

from src.labels import claude_extract as sonnet
from src.labels.opus_review import MODEL, request_body, parse_response
from tools import opus_blind_review as review
from test_claude_label_pilot import claude_prepared, response as sonnet_response
from test_openai_label_pilot import prepared

REAL_ROOT = review.ROOT


def response(report="The ACL is intact."):
    value = sonnet_response(report)
    value["model"] = MODEL
    return value


@pytest.fixture
def opus_prepared(claude_prepared, monkeypatch):
    def send(body, key):
        result = sonnet_response(body["messages"][0]["content"][0]["text"])
        result["candidate_canary"] = "DO_NOT_SEND_SONNET_ANSWERS"
        return result

    review.base.run(claude_prepared, 10, send)
    monkeypatch.setattr(review, "ROOT", claude_prepared.parent)
    monkeypatch.setattr(review, "SOURCES", [])
    config = claude_prepared.parent / "opus_config.json"
    config.write_bytes((REAL_ROOT / "configs/opus_blind_review.json").read_bytes())
    out = claude_prepared.parent / "opus"
    review.prepare(out, claude_prepared, config)
    return out


def test_blind_request_only_changes_reviewer_model_and_effort():
    report, rubric = "The ACL is intact.", "Fixed rubric"
    body = request_body(report, rubric)
    assert body["model"] == MODEL and body["output_config"]["effort"] == "medium"
    body["model"] = sonnet.MODEL
    body["output_config"]["effort"] = "low"
    assert body == sonnet.request_body(report, rubric)


def test_parser_rejects_sonnet_as_opus_and_keeps_quotation_checks():
    with pytest.raises(ValueError, match="model"):
        parse_response(sonnet_response(), "The ACL is intact.")
    with pytest.raises(ValueError, match="substring"):
        parse_response(response("Invented evidence"), "The ACL is intact.")
    assert parse_response(response(), "The ACL is\nintact.")["ACL"]["evidence_quote"] == "The ACL is\nintact."


def test_budget_includes_sonnet_and_review_never_sends_previous_answers(opus_prepared):
    calls = []

    def send(body, key):
        serialized = json.dumps(body)
        assert "DO_NOT_SEND_SONNET_ANSWERS" not in serialized
        assert "StudyInstanceUID" not in serialized
        assert "baseline_fused_target" not in serialized
        calls.append(body)
        return response(body["messages"][0]["content"][0]["text"])

    result = review.run(opus_prepared, .01, send)
    assert result["status"] == "shared_budget_limit" and not calls
    assert result["total_accounted_cost_usd_upper"] == pytest.approx(.048)
    result = review.run(opus_prepared, 10, send)
    assert result["accepted"] == 2 and result["reports_reviewed"] == 2
    assert result["total_accounted_cost_usd_upper"] == pytest.approx(.048 + .096)
    review.run(opus_prepared, 10, lambda *a: pytest.fail("No repeat paid call"))
    assert len(calls) == 2


def test_old_cost_record_change_blocks_before_network(opus_prepared):
    manifest = json.loads((opus_prepared / "pilot.json").read_text())
    old = review.ROOT / manifest["prior_cost_records"][0]["path"]
    old.write_text(old.read_text() + "\n")
    with pytest.raises(ValueError, match="Prior cost record changed"):
        review.run(opus_prepared, 10, lambda *a: pytest.fail("No API call"))


def test_invalid_review_is_charged_and_excluded_without_retry(opus_prepared):
    result = review.run(opus_prepared, 10, lambda *a: response("Invented evidence"))
    assert result["accepted"] == 0 and result["rejected"] == 2
    assert result["total_accounted_cost_usd_upper"] == pytest.approx(.144)
    assert review.run(opus_prepared, 10, lambda *a: pytest.fail("No retry"))["rejected"] == 2


def test_transport_uncertainty_retains_reservation_and_stops(opus_prepared):
    def fail(*args):
        raise RuntimeError("simulated timeout")

    with pytest.raises(RuntimeError, match="timeout"):
        review.run(opus_prepared, 10, fail)
    record = json.loads(next((opus_prepared / "attempts").glob("*.json")).read_text())
    assert record["charged_usd_upper"] == record["reserved_usd"]
    with pytest.raises(RuntimeError, match="earlier request"):
        review.run(opus_prepared, 10, lambda *a: pytest.fail("No retry"))


def test_changed_report_bytes_block_before_network(opus_prepared):
    path = opus_prepared.parent / "reports.csv"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="input bytes"):
        review.run(opus_prepared, 10, lambda *a: pytest.fail("No API call"))
