import json
import io
import urllib.error

import pytest

from src.labels.claude_extract import MODEL, request_body, parse_response, original_quote, normalization_audit
from src.validation.proxy import ID
from tools import claude_label_pilot as pilot
from test_openai_label_pilot import prepared, assertions

REAL_ROOT = pilot.ROOT


def response(report="The ACL is intact."):
    return {"type": "message", "model": MODEL, "stop_reason": "end_turn",
            "usage": {"input_tokens": 3000, "output_tokens": 1800,
                      "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
            "content": [{"type": "text", "text": json.dumps({"findings": [
                {"finding": name, **item} for name, item in assertions(report).items()]})}]}


@pytest.fixture
def claude_prepared(prepared, monkeypatch):
    settings = json.loads((REAL_ROOT / "configs/claude_label_pilot.json").read_text())
    monkeypatch.setattr(pilot, "ROOT", prepared.parent)
    monkeypatch.setattr(pilot, "SOURCES", [])
    config = prepared.parent / "claude_config.json"
    config.write_text(json.dumps(settings))
    out = prepared.parent / "claude"
    pilot.prepare(out, prepared, config)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fixture-never-sent")
    return out


def test_request_uses_claude_schema_and_same_rubric():
    body = request_body("The ACL is intact.", "Frozen research rubric")
    assert body["model"] == MODEL and body["max_tokens"] == 6000
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"]["effort"] == "low"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert "Frozen research rubric" in body["system"]
    assert not {"input", "store", "tools", "cache_control", "metadata"} & set(body)
    assert "StudyInstanceUID" not in json.dumps(body)
    assert parse_response(response(), "The ACL is intact.") == assertions()


@pytest.mark.parametrize("stop", ["max_tokens", "refusal", "tool_use", "pause_turn", None])
def test_nonfinal_responses_are_rejected(stop):
    result = response()
    result["stop_reason"] = stop
    with pytest.raises(ValueError):
        parse_response(result, "The ACL is intact.")


def test_wrong_model_and_invented_evidence_are_rejected():
    result = response()
    result["model"] = "another-model"
    with pytest.raises(ValueError, match="model"):
        parse_response(result, "The ACL is intact.")
    with pytest.raises(ValueError, match="substring"):
        parse_response(response("ACL torn"), "The ACL is intact.")


def test_line_wrapped_evidence_is_restored_and_audited_without_new_words():
    report = "The ACL is\nintact."
    raw = response("The ACL is intact.")
    parsed = parse_response(raw, report)
    assert parsed["ACL"]["evidence_quote"] == report
    changes = normalization_audit(raw, parsed)
    assert len(changes) == 1 and changes[0]["raw_quote"] == "The ACL is intact."
    assert changes[0]["report_quote"] == report
    assert "\\n" not in raw["content"][0]["text"]


@pytest.mark.parametrize("quote,report", [
    ("ACL intact.", "ACL\nis torn."),
    ("ACL intact.", "ACL\nintact. ACL\nintact."),
    ("ACL intact.", "acl\nintact."),
    ("ACL intact.", "ACL\nintact!"),
])
def test_alignment_rejects_changed_words_case_punctuation_or_ambiguous_occurrences(quote, report):
    with pytest.raises(ValueError):
        original_quote(quote, report)


@pytest.mark.parametrize("defect", ["duplicate", "missing", "extra"])
def test_compact_schema_still_requires_exact_finding_coverage(defect):
    result = response()
    data = json.loads(result["content"][0]["text"])
    if defect == "duplicate":
        data["findings"].append(data["findings"][0])
    elif defect == "missing":
        data["findings"].pop()
    else:
        data["findings"].append({"finding": "PCL", **assertions()["ACL"]})
    result["content"][0]["text"] = json.dumps(data)
    with pytest.raises(ValueError):
        parse_response(result, "The ACL is intact.")


def test_cache_usage_is_accounted_without_underpricing():
    settings = json.loads((REAL_ROOT / "configs/claude_label_pilot.json").read_text())
    result = response()
    result["usage"].update(cache_creation_input_tokens=2000, cache_read_input_tokens=4000)
    assert pilot.usage_cost(result, settings) == pytest.approx(.014 + .008 + .018)
    result["usage"]["output_tokens"] = -1
    with pytest.raises(ValueError):
        pilot.usage_cost(result, settings)


@pytest.mark.parametrize("filename", [".env", "claude.env"])
def test_dotenv_is_literal_and_environment_takes_precedence(tmp_path, monkeypatch, filename):
    monkeypatch.setattr(pilot, "ROOT", tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    path = tmp_path / filename
    path.write_text("# ignored\nOTHER=ignored\nexport ANTHROPIC_API_KEY='$(not-executed)'\n")
    assert pilot.local_setting("ANTHROPIC_API_KEY") == "$(not-executed)"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "environment-value")
    assert pilot.local_setting("ANTHROPIC_API_KEY") == "environment-value"
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    path.write_text("ANTHROPIC_API_KEY=first\nANTHROPIC_API_KEY=second\n")
    with pytest.raises(ValueError, match="Duplicate"):
        pilot.local_setting("ANTHROPIC_API_KEY")


def test_switch_preserves_selection_and_blind_review(claude_prepared):
    manifest = json.loads((claude_prepared / "pilot.json").read_text())
    original = json.loads((claude_prepared.parent / "pilot/pilot.json").read_text())
    assert manifest["inputs"] == original["inputs"]
    assert [r[ID] for r in manifest["studies"]] == [r[ID] for r in original["studies"]]
    assert [r["report_sha256"] for r in manifest["studies"]] == [r["report_sha256"] for r in original["studies"]]
    assert manifest["studies"][0]["request_sha256"] != original["studies"][0]["request_sha256"]
    assert (claude_prepared / "blind_review.jsonl").read_bytes() == (claude_prepared.parent / "pilot/blind_review.jsonl").read_bytes()


def test_budget_gate_and_resume(claude_prepared):
    calls = []

    def send(body, key):
        calls.append(body)
        return response(body["messages"][0]["content"][0]["text"])

    assert pilot.run(claude_prepared, .00001, send)["status"] == "budget_limit"
    assert not calls
    with pytest.raises(ValueError, match="approved"):
        pilot.run(claude_prepared, 10.01, send)
    result = pilot.run(claude_prepared, 10., send)
    assert len(calls) == 2 and result["accepted"] == 2
    assert result["charged_usd_upper"] == pytest.approx(2 * (.006 + .018))
    assert pilot.run(claude_prepared, 10., send)["status"] == "extraction_completed"
    assert len(calls) == 2


def test_previous_failed_request_cost_is_carried_into_budget(claude_prepared):
    path = claude_prepared.parent / "previous_attempt.json"
    path.write_text(json.dumps({"charged_usd_upper": 9.99, "status": "http_error"}))
    manifest_path = claude_prepared / "pilot.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["prior_cost_records"] = [{"path": path.name, "sha256": pilot.sha(path),
                                      "charged_usd_upper": 9.99}]
    manifest_path.write_text(json.dumps(manifest))
    result = pilot.run(claude_prepared, 10., lambda *a: pytest.fail("No budget remains for a request"))
    assert result["status"] == "budget_limit" and result["charged_usd_upper"] == 9.99
    path.write_text(json.dumps({"charged_usd_upper": 0}))
    with pytest.raises(ValueError, match="Prior cost record changed"):
        pilot.run(claude_prepared, 10., lambda *a: pytest.fail("Prior reservation was altered"))


def test_timeout_keeps_reservation_and_blocks_paid_retry(claude_prepared):
    def send(*args):
        raise RuntimeError("simulated timeout")

    with pytest.raises(RuntimeError, match="timeout"):
        pilot.run(claude_prepared, 10., send)
    record = json.loads(next((claude_prepared / "attempts").glob("*.json")).read_text())
    assert record["status"] == "needs_review"
    assert record["charged_usd_upper"] == record["reserved_usd"]
    with pytest.raises(RuntimeError, match="earlier request"):
        pilot.run(claude_prepared, 10., lambda *a: pytest.fail("No paid retry"))


def test_invalid_evidence_still_records_paid_usage(claude_prepared):
    result = pilot.run(claude_prepared, 10., lambda *a: response("Invented evidence"))
    assert result["status"] == "pilot_completed_with_rejections"
    assert result["accepted"] == 0 and result["rejected"] == 2
    record = json.loads(next((claude_prepared / "attempts").glob("*.json")).read_text())
    assert record["status"] == "rejected" and "assertions" not in record
    assert record["charged_usd_upper"] == pytest.approx(.024)
    assert "response" in record
    resumed = pilot.run(claude_prepared, 10., lambda *a: pytest.fail("No retry of rejected response"))
    assert resumed["rejected"] == 2


def test_changed_inputs_and_prior_provider_spend_block(claude_prepared):
    prior = claude_prepared.parent / "pilot/attempts"
    prior.mkdir()
    (prior / "prior.json").write_text("{}")
    with pytest.raises(ValueError, match="shared budget"):
        pilot.run(claude_prepared, 10., lambda *a: pytest.fail("No API call"))
    reports = claude_prepared.parent / "reports.csv"
    reports.write_text(reports.read_text() + "\n")
    with pytest.raises(ValueError, match="input bytes"):
        pilot.run(claude_prepared, 10., lambda *a: pytest.fail("No API call"))


def test_missing_key_blocks_before_attempt(claude_prepared, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        pilot.run(claude_prepared, 10., lambda *a: pytest.fail("No API call"))
    assert not (claude_prepared / "attempts").exists()


def test_transport_sends_required_workspace_header_without_exposing_key(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, "ROOT", tmp_path)
    monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID", raising=False)
    (tmp_path / "claude.env").write_text("ANTHROPIC_WORKSPACE_ID=wrkspc_fixture\n")

    def urlopen(request, timeout):
        assert request.full_url == "https://api.anthropic.com/v1/messages"
        headers = {k.lower(): v for k, v in request.header_items()}
        assert headers["x-api-key"] == "fixture-key"
        assert headers["anthropic-workspace-id"] == "wrkspc_fixture"
        assert headers["anthropic-version"] == "2023-06-01"
        assert "fixture-key" not in request.data.decode()
        result = io.BytesIO(json.dumps(response()).encode())
        result.headers = {"request-id": "req_fixture"}
        return result

    monkeypatch.setattr(pilot.urllib.request, "urlopen", urlopen)
    result = pilot.post(request_body("The ACL is intact.", "Rubric"), "fixture-key")
    assert result["_request_id"] == "req_fixture"


def test_access_check_reports_missing_workspace_without_paid_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, "ROOT", tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fixture-key")
    monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID", raising=False)
    calls = []

    def urlopen(request, timeout):
        calls.append(request.full_url)
        assert request.get_method() == "GET"
        error = {"error": {"type": "invalid_request_error", "message": "workspace required fixture-key"}}
        raise urllib.error.HTTPError(request.full_url, 400 if len(calls) == 1 else 403,
                                     "fixture", {}, io.BytesIO(json.dumps(error).encode()))

    monkeypatch.setattr(pilot.urllib.request, "urlopen", urlopen)
    result = pilot.check_access()
    assert result["workspace_lookup_http_status"] == 403
    assert result["report_inference_calls"] == 0
    assert "fixture-key" not in json.dumps(result)
    assert len(calls) == 2
