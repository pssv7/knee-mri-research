"""Tests for multilingual finding extraction.

The extractor's contract has three states, and the distinction between 0 and
None is the part most likely to be broken by a well-meaning refactor: a
finding that is never mentioned is unknown, not absent.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config  # noqa: E402
from src.labels import calibrate, lexicon  # noqa: E402


class TestThreeStateContract:
    def test_unmentioned_is_none_not_zero(self):
        report = "The bones are unremarkable. No marrow oedema."
        assert lexicon.extract_finding(report, "Baker's")["present"] is None

    def test_negated_mention_is_zero(self):
        report = "No popliteal cyst is seen."
        assert lexicon.extract_finding(report, "Baker's")["present"] == 0

    def test_positive_mention_is_one(self):
        report = "There is a large popliteal cyst."
        assert lexicon.extract_finding(report, "Baker's")["present"] == 1

    def test_all_findings_returned(self):
        result = lexicon.extract_report("Normal knee.")
        assert set(result) == set(config.LABELS)

    def test_empty_report_is_all_none(self):
        for value in lexicon.extract_report("").values():
            assert value["present"] is None
        for value in lexicon.extract_report(None).values():
            assert value["present"] is None


class TestNegation:
    def test_negation_scoped_to_clause(self):
        # The negation belongs to the meniscus clause and must not reach the
        # effusion clause that follows it.
        report = "No meniscal tear. Joint effusion is present."
        assert lexicon.extract_finding(report, "Effusion")["present"] == 1

    def test_negation_across_languages(self):
        cases = [
            ("Sin derrame articular.", "Effusion"),
            ("Geen gewrichtsvocht.", "Effusion"),
            ("Kein Gelenkerguss.", "Effusion"),
            ("Eklem ici sivi yoktur.", "Effusion"),
            ("Bez izliva u zglobu.", "Effusion"),
            ("Без ставен излив.", "Effusion"),
            ("Χωρίς ενδαρθρική συλλογή υγρού.", "Effusion"),
        ]
        for report, finding in cases:
            assert lexicon.extract_finding(report, finding)["present"] == 0, report

    def test_contrastive_conjunction_breaks_negation(self):
        report = "No fracture but there is a joint effusion"
        assert lexicon.extract_finding(report, "Effusion")["present"] == 1
        assert lexicon.extract_finding(report, "Fracture")["present"] == 0


class TestInjuryContext:
    def test_named_intact_structure_is_not_positive(self):
        # A meniscus is always present; only a torn one is a positive label.
        report = "The medial meniscus is of normal morphology and signal."
        assert lexicon.extract_finding(report, "Medial Meniscus")["present"] != 1

    def test_tear_makes_it_positive(self):
        report = "Rotura del menisco interno."
        assert lexicon.extract_finding(report, "Medial Meniscus")["present"] == 1

    def test_compartment_specificity(self):
        report = "Tear of the lateral meniscus. Medial meniscus intact."
        assert lexicon.extract_finding(report, "Lateral Meniscus")["present"] == 1
        assert lexicon.extract_finding(report, "Medial Meniscus")["present"] != 1


class TestOsteoarthritisLocalization:
    def test_disease_stem_requires_a_compartment(self):
        # Anchoring on the compartment word alone fails: "medial" appears in
        # nearly every report. The disease stem is the anchor.
        report = "Medial meniscus and medial collateral ligament are normal."
        assert lexicon.extract_finding(report, "Medial OA")["present"] is None

    def test_localizes_to_the_named_compartment(self):
        report = "Artrosis femorotibial medial."
        assert lexicon.extract_finding(report, "Medial OA")["present"] == 1
        assert lexicon.extract_finding(report, "Lateral OA")["present"] is None

    def test_patellofemoral_localizer(self):
        report = "Chondromalacia of the patellar cartilage."
        assert lexicon.extract_finding(report, "PF OA")["present"] == 1


class TestSeverity:
    def test_unqualified_scores_higher_than_minimal(self):
        strong = lexicon.extract_finding("Large joint effusion.", "Effusion")
        weak = lexicon.extract_finding("Minimal joint effusion.", "Effusion")
        assert strong["severity"] > weak["severity"]

    def test_hedging_lowers_severity(self):
        certain = lexicon.extract_finding("Fracture of the tibial plateau.", "Fracture")
        hedged = lexicon.extract_finding("Possible fracture of the tibia.", "Fracture")
        assert hedged["severity"] < certain["severity"]

    def test_span_is_returned_for_positives(self):
        result = lexicon.extract_finding("There is a large joint effusion.", "Effusion")
        assert result["span"] and "effusion" in result["span"].lower()


class TestAucHelper:
    def test_perfect_separation(self):
        truth = np.array([0, 0, 1, 1])
        assert calibrate._auc(truth, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0

    def test_constant_scores_are_chance(self):
        # Ties must average to 0.5; without tie handling this returns 1.0 and
        # every silent finding would look perfect.
        truth = np.array([0, 0, 1, 1])
        assert calibrate._auc(truth, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5

    def test_inverted_scores(self):
        truth = np.array([0, 0, 1, 1])
        assert calibrate._auc(truth, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0

    def test_single_class_is_nan(self):
        assert np.isnan(calibrate._auc(np.array([1, 1]), np.array([0.1, 0.2])))


@pytest.mark.skipif(not config.TRAIN_CSV.exists(), reason="train.csv not downloaded")
class TestAgainstGold:
    def test_gold_split_sizes(self):
        gold = calibrate.load_gold()
        assert len(gold) == 58

    def test_macro_auc_does_not_regress(self):
        # Macro AUC moved 0.649 -> 0.708 (OA localisation) -> 0.721 (postfix
        # negation, clause-scoped injury context) -> 0.735 (localisers exempt
        # from clause scope). A drop below this means a term list or a window
        # broke.
        table = calibrate.score_extractor(lexicon.extract_finding)
        assert table["auc"].mean() > 0.72

    def test_no_finding_is_worse_than_chance(self):
        table = calibrate.score_extractor(lexicon.extract_finding)
        worst = table["auc"].min()
        assert worst > 0.55, f"worst finding AUC {worst:.3f}"

    def test_silence_prior_varies_by_finding(self):
        # The whole reason silence cannot be collapsed to zero: effusion is
        # positive well over half the time it goes unmentioned, MCL almost
        # never is.
        table = calibrate.score_extractor(lexicon.extract_finding)
        assert table.loc["Effusion", "prior_if_silent"] > 0.4
        assert table.loc["MCL", "prior_if_silent"] < 0.15
