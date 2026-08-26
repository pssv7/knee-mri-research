"""The error taxonomy in tools/inspect_extraction_errors.py.

The split between UNREACHABLE and MISSED is the basis of a claim in
PROJECT.md -- that most missed positives were never written down, so they
are a ceiling rather than an extraction defect. Pin the semantics so that
claim cannot drift silently.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from inspect_extraction_errors import anatomy_hits, classify, sentences  # noqa: E402


class TestErrorTaxonomy:
    def test_a_positive_the_report_never_mentions_is_unreachable(self):
        assert classify(None, 1, in_text=False) == "UNREACHABLE"

    def test_a_positive_the_report_does_mention_is_a_real_miss(self):
        assert classify(None, 1, in_text=True) == "MISSED (in text)"

    def test_asserted_absence_against_a_positive_is_its_own_class(self):
        # Kept separate from silence: these are usually the report itself
        # contradicting the gold ("Lateral meniskus normal." against a
        # positive label), not a failure to read.
        assert classify(0, 1, in_text=True) == "MISSED (said no)"
        assert classify(0, 1, in_text=False) == "MISSED (said no)"

    def test_asserted_presence_against_a_negative_is_a_false_positive(self):
        assert classify(1, 0, in_text=True) == "FALSE POSITIVE"

    def test_agreement_is_not_an_error(self):
        assert classify(1, 1, in_text=True) is None
        assert classify(0, 0, in_text=True) is None
        # Silence on a true negative is correct behaviour, not a miss.
        assert classify(None, 0, in_text=False) is None


class TestReportReading:
    def test_sentences_survive_newlines_and_punctuation(self):
        text = "Lateral meniscus normal.\nACL intact; PCL intact."
        assert sentences(text) == ["Lateral meniscus normal.",
                                   "ACL intact;", "PCL intact."]

    def test_non_string_reports_do_not_raise(self):
        assert sentences(None) == []
        assert anatomy_hits(None, "ACL") == []

    def test_anatomy_hits_find_the_finding_in_any_language(self):
        # The corpus is multilingual; matching runs on normalised text.
        assert anatomy_hits("Rotura del menisco lateral.", "Lateral Meniscus")
        assert anatomy_hits("Lateral menisküs normal.", "Lateral Meniscus")

    def test_anatomy_hits_are_empty_when_the_finding_is_unmentioned(self):
        assert anatomy_hits("The ACL is intact.", "Baker's") == []
