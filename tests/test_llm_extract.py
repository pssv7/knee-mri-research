"""Tests for LLM prompt construction and response parsing.

Parsing runs on 13,000 generations with no human in the loop, so the failure
mode that matters is a malformed response being read as a confident answer.
Every ambiguous path must resolve to None.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config  # noqa: E402
from src.labels import llm_extract as lx  # noqa: E402

GOOD_RESPONSE = """{
  "ACL": {"present": "yes", "severity": "severe", "certainty": "definite",
          "evidence": "complete ACL rupture"},
  "MCL": {"present": "no", "severity": "none", "certainty": "definite",
          "evidence": "MCL intact"},
  "Medial Meniscus": {"present": "unclear", "severity": "none",
                      "certainty": "definite", "evidence": ""},
  "Lateral Meniscus": {"present": "yes", "severity": "minimal",
                       "certainty": "possible", "evidence": "possible small tear"},
  "Medial OA": {"present": "unclear", "severity": "none", "certainty": "definite", "evidence": ""},
  "Lateral OA": {"present": "unclear", "severity": "none", "certainty": "definite", "evidence": ""},
  "PF OA": {"present": "unclear", "severity": "none", "certainty": "definite", "evidence": ""},
  "Effusion": {"present": "yes", "severity": "moderate", "certainty": "definite",
               "evidence": "joint effusion"},
  "Synovitis": {"present": "unclear", "severity": "none", "certainty": "definite", "evidence": ""},
  "Baker's": {"present": "unclear", "severity": "none", "certainty": "definite", "evidence": ""},
  "Contusion": {"present": "unclear", "severity": "none", "certainty": "definite", "evidence": ""},
  "Fracture": {"present": "no", "severity": "none", "certainty": "definite",
               "evidence": "no fracture"}
}"""


class TestPrompt:
    def test_contains_all_findings(self):
        prompt = lx.build_prompt("Normal knee.")
        for name in config.LABELS:
            assert name in prompt

    def test_definitions_cover_every_label(self):
        assert set(lx.FINDING_DEFINITIONS) == set(config.LABELS)

    def test_report_is_embedded(self):
        prompt = lx.build_prompt("Rotura del menisco interno.")
        assert "Rotura del menisco interno." in prompt

    def test_distinguishes_unclear_from_no(self):
        prompt = lx.build_prompt("x")
        assert "unclear" in prompt and "silent" in prompt


class TestParsing:
    def test_parses_well_formed_response(self):
        result = lx.parse_response(GOOD_RESPONSE)
        assert result["ACL"]["present"] == 1
        assert result["MCL"]["present"] == 0
        assert result["Medial Meniscus"]["present"] is None
        assert result["ACL"]["evidence"] == "complete ACL rupture"

    def test_severity_reflects_hedging(self):
        result = lx.parse_response(GOOD_RESPONSE)
        # minimal x possible must score below moderate x definite.
        assert result["Lateral Meniscus"]["severity"] < result["Effusion"]["severity"]

    def test_negative_scores_zero(self):
        assert lx.parse_response(GOOD_RESPONSE)["MCL"]["severity"] == 0.0

    def test_tolerates_surrounding_prose(self):
        wrapped = "Here is the JSON you asked for:\n```json\n" + GOOD_RESPONSE + "\n```\nDone."
        assert lx.parse_response(wrapped)["ACL"]["present"] == 1

    def test_repairs_trailing_commas(self):
        broken = '{"ACL": {"present": "yes", "severity": "severe", "certainty": "definite", "evidence": "t",},}'
        assert lx.parse_response(broken)["ACL"]["present"] == 1

    def test_braces_inside_strings_do_not_confuse_scanner(self):
        tricky = '{"ACL": {"present": "no", "severity": "none", "certainty": "definite", "evidence": "text with } brace"}}'
        assert lx.parse_response(tricky)["ACL"]["present"] == 0

    def test_garbage_is_all_none(self):
        for text in ["", None, "I cannot help with that.", "{{{", "null"]:
            result = lx.parse_response(text)
            assert all(v["present"] is None for v in result.values()), text

    def test_missing_finding_is_none_not_negative(self):
        # A truncated generation must never be read as twelve confident zeros.
        partial = '{"ACL": {"present": "yes", "severity": "severe", "certainty": "definite", "evidence": "x"}}'
        result = lx.parse_response(partial)
        assert result["ACL"]["present"] == 1
        assert result["Fracture"]["present"] is None

    def test_unknown_presence_value_is_none(self):
        odd = '{"ACL": {"present": "maybe", "severity": "severe", "certainty": "definite", "evidence": "x"}}'
        assert lx.parse_response(odd)["ACL"]["present"] is None

    def test_always_returns_every_finding(self):
        assert set(lx.parse_response("garbage")) == set(config.LABELS)


class TestAggregation:
    def _sample(self, acl_present, severity=1.0):
        base = {name: {"present": None, "severity": None, "evidence": None}
                for name in config.LABELS}
        base["ACL"] = {"present": acl_present, "severity": severity,
                       "evidence": "e"}
        return base

    def test_majority_vote(self):
        samples = [self._sample(1), self._sample(1), self._sample(0)]
        result = lx.aggregate_samples(samples)
        assert result["ACL"]["present"] == 1

    def test_confidence_is_vote_fraction(self):
        samples = [self._sample(1), self._sample(1), self._sample(0)]
        assert abs(lx.aggregate_samples(samples)["ACL"]["confidence"] - 2 / 3) < 1e-9

    def test_unanimous_is_full_confidence(self):
        samples = [self._sample(1)] * 3
        assert lx.aggregate_samples(samples)["ACL"]["confidence"] == 1.0

    def test_severity_averages_over_winning_votes_only(self):
        samples = [self._sample(1, 1.0), self._sample(1, 0.5), self._sample(0, 0.0)]
        assert abs(lx.aggregate_samples(samples)["ACL"]["severity"] - 0.75) < 1e-9

    def test_empty_samples_are_none(self):
        result = lx.aggregate_samples([])
        assert all(v["present"] is None for v in result.values())

    def test_unclear_can_win(self):
        samples = [self._sample(None), self._sample(None), self._sample(1)]
        assert lx.aggregate_samples(samples)["ACL"]["present"] is None
