import json

import pytest

from tools import compare_opus_blind_review as comparison
from tools import opus_blind_review as review
from test_opus_blind_review import opus_prepared, response
from test_claude_label_pilot import claude_prepared
from test_openai_label_pilot import prepared


def responder(change=None, rejected=False, duplicate_id=False):
    count = 0

    def send(body, key):
        nonlocal count
        count += 1
        value = response("Invented evidence" if rejected else body["messages"][0]["content"][0]["text"])
        value["id"] = "response" if duplicate_id else f"response-{count}"
        if change:
            payload = json.loads(value["content"][0]["text"])
            change(payload)
            value["content"][0]["text"] = json.dumps(payload)
        return value

    return send


def test_unfinished_comparison_stays_sealed(opus_prepared):
    with pytest.raises(ValueError, match="sealed"):
        comparison.compare(opus_prepared)
    assert not (opus_prepared / "all_comparisons.jsonl").exists()


def test_exact_agreement_and_shared_cost(opus_prepared):
    review.run(opus_prepared, 10, responder())
    result = comparison.compare(opus_prepared)
    assert result["reports_accepted"] == 2 and result["accepted_finding_assertions"] == 24
    assert result["agreement"]["exact_state_agreements"] == 24
    assert result["agreement"]["unavailable_slots"] == 0
    assert result["shared_total_accounted_cost_usd_upper"] == pytest.approx(.144)
    assert result["human_reference_accuracy_established"] is False


def test_rejections_are_missing_not_agreement(opus_prepared):
    review.run(opus_prepared, 10, responder(rejected=True))
    result = comparison.compare(opus_prepared)
    assert result["reports_rejected"] == 2
    assert result["agreement"]["comparable_slots"] == 0
    assert result["agreement"]["unavailable_slots"] == 24
    assert result["agreement"]["exact_state_agreement_fraction"] is None
    assert result["review_queue_rows"] == 24
    assert result["opus_response_token_cost_usd_upper"] == pytest.approx(.096)


def test_state_disagreement_not_hidden_by_silent_findings(opus_prepared):
    def change(payload):
        acl = next(a for a in payload["findings"] if a["finding"] == "ACL")
        acl["report_states"] = "uncertain"

    review.run(opus_prepared, 10, responder(change=change))
    result = comparison.compare(opus_prepared)
    assert result["agreement"]["state_disagreements"] == 2
    assert result["per_finding"]["ACL"]["state_disagreements"] == 2
    assert result["per_finding"]["ACL"]["exact_state_agreement_fraction"] == 0
    queue = comparison.read_jsonl(opus_prepared / "adjudication_queue.jsonl")
    assert len(queue) == 2 and all("state_disagreement" in r["review_reasons"] for r in queue)


def test_duplicate_api_response_is_not_counted_twice(opus_prepared):
    review.run(opus_prepared, 10, responder(duplicate_id=True))
    with pytest.raises(ValueError, match="Duplicate provider response"):
        comparison.compare(opus_prepared)


def test_positive_subset_denominator_and_severity_are_explicit():
    def item(state, severity="minimal"):
        return {"report_states": state, "severity_norm": severity, "certainty": "definite"}

    rows = [{"sonnet": item("present"), "opus": item("present", "moderate")},
            {"sonnet": item("present"), "opus": item("uncertain")},
            {"sonnet": item("negated"), "opus": item("negated")},
            {"sonnet": None, "opus": item("present")}]
    result = comparison.agreement(rows)
    assert result["exact_state_agreement_fraction"] == pytest.approx(2 / 3)
    assert result["either_model_present"] == 2 and result["both_models_present"] == 1
    assert result["both_present_over_either_present"] == .5
    assert result["severity_disagreements_when_both_present"] == 1
