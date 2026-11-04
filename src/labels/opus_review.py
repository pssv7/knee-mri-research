"""Blinded Opus annotation using the same report-only rubric and output schema."""
import json

from src.labels import claude_extract as sonnet
from src.labels.openai_extract import validate, _unique_object

MODEL = "claude-opus-5-5"


def request_body(report, rubric):
    body = sonnet.request_body(report, rubric)
    body["model"] = MODEL
    body["output_config"]["effort"] = "medium"
    return body


def parse_response(response, report):
    if response.get("model") != MODEL:
        raise ValueError("Response model differs from the frozen Opus reviewer")
    if response.get("type") != "message" or response.get("stop_reason") != "end_turn":
        raise ValueError("Opus review did not complete normally")
    content = response.get("content", [])
    if any(item.get("type") not in {"text", "thinking", "redacted_thinking"} for item in content):
        raise ValueError("Unexpected Opus response content")
    text = "".join(item["text"] for item in content if item.get("type") == "text")
    result = json.loads(text, object_pairs_hook=_unique_object)
    if not isinstance(result, dict) or set(result) != {"findings"} or not isinstance(result["findings"], list):
        raise ValueError("Expected a findings array")
    assertions = {}
    for item in result["findings"]:
        if not isinstance(item, dict) or not isinstance(item.get("finding"), str):
            raise ValueError("Invalid finding item")
        name = item["finding"]
        if name in assertions:
            raise ValueError("Duplicate finding")
        assertions[name] = {key: value for key, value in item.items() if key != "finding"}
        if "evidence_quote" in assertions[name]:
            assertions[name]["evidence_quote"] = sonnet.original_quote(assertions[name]["evidence_quote"], report)
    return validate(assertions, report)
