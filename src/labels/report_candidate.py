"""Sonnet candidate with pilot-derived application checks; original code stays frozen."""
from src.labels import claude_extract as base

MODEL = base.MODEL


def request_body(report, rubric, clarifications):
    body = base.request_body(report, rubric)
    body["system"] += "\n\nAPPLICATION CHECKS:\n" + clarifications
    return body


def parse_response(response, report):
    result = base.parse_response(response, report)
    for finding, item in result.items():
        if item["report_states"] in {"present", "negated"} and item["certainty"] != "definite":
            raise ValueError("Definite-state/certainty inconsistency: " + finding)
    return result
