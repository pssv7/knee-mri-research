from datetime import datetime, timedelta, timezone

import pytest

from src.labels import ledger

REPORT = "Lateral meniscus: radial tear of the posterior horn.\nMedial meniscus intact."


def a_row(**kw):
    row = {"report_states": "present", "anatomical_location": "lateral meniscus posterior horn",
           "certainty": "definite", "severity_text": "", "severity_norm": "not stated",
           "evidence_quote": "radial tear of the posterior horn", "note": ""}
    row.update(kw)
    return row


def b_row(**kw):
    row = {"extractor_correct": "no", "error_class": "missed_mention", "disposition": "correct_target",
           "proposed_target": "1.0", "proposed_weight": "", "rationale": "explicit tear"}
    row.update(kw)
    return row


def test_stage_a_accepts_a_quoted_present_reading():
    assert ledger.validate_stage_a(a_row(), REPORT) == []


def test_stage_a_accepts_rewrapped_whitespace_but_not_invented_text():
    assert ledger.validate_stage_a(a_row(evidence_quote="radial  tear of the\nposterior horn"), REPORT) == []
    errors = ledger.validate_stage_a(a_row(evidence_quote="complex tear"), REPORT)
    assert errors == ["evidence_quote is not an exact substring of the report"]


def test_stage_a_errors_never_echo_the_quote():
    secret = "a phrase that must not be printed"
    errors = ledger.validate_stage_a(a_row(evidence_quote=secret), REPORT)
    assert errors and all(secret not in e for e in errors)


@pytest.mark.parametrize("row, fragment", [
    (a_row(report_states="torn"), "report_states"),
    (a_row(evidence_quote=""), "evidence_quote is required"),
    (a_row(certainty="sure"), "certainty"),
    (a_row(anatomical_location=""), "anatomical_location"),
    (a_row(report_states="unmentioned"), "must be empty"),
])
def test_stage_a_rejections(row, fragment):
    assert any(fragment in e for e in ledger.validate_stage_a(row, REPORT))


def test_unmentioned_reading_needs_nothing():
    row = a_row(report_states="unmentioned", evidence_quote="", certainty="", severity_norm="",
                anatomical_location="")
    assert ledger.validate_stage_a(row, REPORT) == []


@pytest.mark.parametrize("state, target, expected", [
    ("present", .9, True), ("present", .2, False), ("negated", .2, True), ("negated", .9, False),
    ("present_below_threshold", .35, True), ("uncertain", .9, None), ("unmentioned", .1, None),
    ("unattributable", .6, None)])
def test_final_target_correctness_follows_rubric_section_4(state, target, expected):
    assert ledger.final_target_correct(state, target) is expected


def test_missed_positive_correction_is_accepted():
    errors, correct = ledger.validate_stage_b(a_row(), b_row(), old_target=0.2, old_weight=0.7)
    assert errors == [] and correct is False


def test_correcting_an_already_correct_final_target_is_refused():
    errors, correct = ledger.validate_stage_b(a_row(), b_row(error_class="anatomy"), old_target=0.8, old_weight=1)
    assert correct is True and any("already correct" in e for e in errors)


def test_an_intermediate_error_on_a_correct_target_is_recorded_with_keep():
    errors, _ = ledger.validate_stage_b(
        a_row(), b_row(error_class="anatomy", disposition="keep", proposed_target=""), 0.8, 1)
    assert errors == []


@pytest.mark.parametrize("state", ["uncertain", "unmentioned", "unattributable"])
def test_no_change_on_readings_without_judgement(state):
    errors, correct = ledger.validate_stage_b(a_row(report_states=state), b_row(), 0.2, 1)
    assert correct is None and any("disposition must be keep" in e for e in errors)


def test_proposed_target_must_follow_the_reading():
    errors, _ = ledger.validate_stage_b(a_row(), b_row(proposed_target="0.3"), 0.2, 1)
    assert any("side of 0.5" in e for e in errors)


def test_negation_correction_goes_down():
    errors, _ = ledger.validate_stage_b(
        a_row(report_states="negated", evidence_quote="Medial meniscus intact"),
        b_row(error_class="negation", proposed_target="0.0"), 0.9, 1)
    assert errors == []


@pytest.mark.parametrize("change", [
    {"error_class": "unresolved"}, {"error_class": "report_image_disagreement"},
    {"error_class": "fusion_weighting"}, {"rationale": ""}, {"proposed_weight": "0.5"},
    {"proposed_target": "1.7"}])
def test_target_correction_rejections(change):
    errors, _ = ledger.validate_stage_b(a_row(), b_row(**change), 0.2, 1)
    assert errors


def test_disagreeing_target_needs_a_class():
    errors, _ = ledger.validate_stage_b(a_row(), b_row(error_class="none", disposition="keep",
                                                       proposed_target=""), 0.2, 1)
    assert any("classify the error" in e for e in errors)


def test_weight_correction_rules():
    ok = b_row(error_class="fusion_weighting", disposition="correct_weight", proposed_target="",
               proposed_weight="1.0")
    assert ledger.validate_stage_b(a_row(), ok, 0.8, 0.4)[0] == []
    same = dict(ok, proposed_weight="0.4")
    assert ledger.validate_stage_b(a_row(), same, 0.8, 0.4)[0]
    wrong_class = dict(ok, error_class="anatomy")
    assert ledger.validate_stage_b(a_row(), wrong_class, 0.8, 0.4)[0]


def test_two_passes_must_agree_on_the_change():
    up = {"disposition": "correct_target", "proposed_target": "1.0", "old_weight": 1}
    also_up = {"disposition": "correct_target", "proposed_target": "0.85", "old_weight": 1}
    down = {"disposition": "correct_target", "proposed_target": "0.0", "old_weight": 1}
    keep = {"disposition": "keep", "old_weight": 1}
    assert ledger.agreed_change(up, also_up)
    assert not ledger.agreed_change(up, down)
    assert not ledger.agreed_change(up, keep)
    assert not ledger.agreed_change(keep, keep)


def test_rereview_waits_seven_days():
    first = datetime(2026, 9, 24, 12, tzinfo=timezone.utc)
    assert not ledger.rereview_due(first, first + timedelta(days=6, hours=23))
    assert ledger.rereview_due(first, first + timedelta(days=7))
