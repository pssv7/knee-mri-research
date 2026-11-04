"""Claude Messages adapter for the same evidence-checked report rubric."""
import json
import re

from src.labels.openai_extract import instructions, schema, validate, _unique_object

MODEL = "claude-sonnet-5-5"


def output_schema():
    # A single repeated item avoids Anthropic's compiled-grammar complexity limit
    # for twelve copies of the same six-field object. Validate coverage locally.
    original = schema()
    item = original["properties"]["ACL"]
    item["properties"] = {"finding": {"type": "string", "enum": original["required"]},
                          **item["properties"]}
    item["required"] = ["finding", *item["required"]]
    return {"type": "object", "properties": {"findings": {"type": "array", "items": item}},
            "required": ["findings"], "additionalProperties": False}


def request_body(report, rubric):
    if not isinstance(report, str) or not report.strip():
        raise ValueError("Empty report")
    return {
        "model": MODEL,
        "system": instructions(rubric) + "\nReturn findings as an array of twelve objects, "
                  "one per required finding. Put each finding's name in its finding field.",
        "messages": [{"role": "user", "content": [{"type": "text", "text": report}]}],
        "max_tokens": 6000,
        "thinking": {"type": "adaptive"},
        "inference_geo": "global",
        "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": output_schema()}},
    }


def original_quote(quote, report):
    """Restore only whitespace, and only when alignment to the report is unique."""
    if not isinstance(quote, str) or not quote.strip() or quote in report:
        return quote
    spans = list(re.finditer(r"\s+|\S", report))
    collapsed = "".join(" " if match.group().isspace() else match.group() for match in spans)
    target = " ".join(quote.split())
    start = collapsed.find(target)
    if start < 0 or collapsed.find(target, start + 1) >= 0:
        raise ValueError("Evidence is not an exact report substring or unique whitespace-only match")
    return report[spans[start].start():spans[start + len(target) - 1].end()]


def normalization_audit(response, assertions):
    text = "".join(item["text"] for item in response["content"] if item.get("type") == "text")
    raw = json.loads(text, object_pairs_hook=_unique_object)["findings"]
    return [{"finding": item["finding"], "raw_quote": item["evidence_quote"],
             "report_quote": assertions[item["finding"]]["evidence_quote"],
             "method": "unique whitespace-only alignment; original report characters restored"}
            for item in raw if item["evidence_quote"] != assertions[item["finding"]]["evidence_quote"]]


def parse_response(response, report):
    if response.get("model") != MODEL:
        raise ValueError("Response model differs from the frozen Claude model")
    if response.get("type") != "message" or response.get("stop_reason") != "end_turn":
        raise ValueError("Claude response did not complete normally; no labels accepted")
    content = response.get("content", [])
    if any(item.get("type") not in {"text", "thinking", "redacted_thinking"} for item in content):
        raise ValueError("Unexpected Claude response content")
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
            assertions[name]["evidence_quote"] = original_quote(assertions[name]["evidence_quote"], report)
    return validate(assertions, report)
