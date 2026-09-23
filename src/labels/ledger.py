"""The report-audit ledger: fields, and the rules a review row must satisfy.

A review has two stages, entered separately so the first can't be steered by
the second:

- **Stage A, blind.** What the report states, read from the report alone
  (docs/report_review_rubric.md sections 2-3). No target, weight, score or
  extractor output is visible.
- **Stage B, after the reveal.** The final target, weight and intermediate
  extractor calls are shown next to the reviewer's own stage-A answer. The
  reviewer classifies the error and proposes a disposition.

Whether the final target is correct is *computed* from stage A and the target
(rubric section 4), never typed in, so the two can't disagree. The rules
refuse the corrections the source plan forbids: a change justified only by a
soft target below 0.5, a change to a final target that is already correct,
a change without exact report evidence, and any change on an uncertain,
unmentioned or unattributable reading. Error messages name fields, never the
quoted text.
"""

from datetime import timedelta
import re

REPORT_STATES = ("present", "present_below_threshold", "negated", "uncertain",
                 "unmentioned", "unattributable")
JUDGED = {"present": True, "present_below_threshold": False, "negated": False}
CERTAINTY = ("definite", "probable", "possible")
SEVERITY = ("minimal", "moderate", "severe", "not stated")
EXTRACTOR_CORRECT = ("yes", "no", "not_assessable")
ERROR_CLASSES = ("none", "anatomy", "negation", "missed_mention", "severity_definition",
                 "fusion_weighting", "report_image_disagreement", "unresolved")
TARGET_CLASSES = ("anatomy", "negation", "missed_mention", "severity_definition")
DISPOSITIONS = ("keep", "correct_target", "correct_weight")

STAGE_A = ["case_id", "finding", "report_states", "anatomical_location", "certainty",
           "severity_text", "severity_norm", "evidence_quote", "note"]
STAGE_B = ["case_id", "finding", "extractor_correct", "error_class", "disposition",
           "proposed_target", "proposed_weight", "rationale"]

POSITIVE = 0.5
REREVIEW_DELAY = timedelta(days=7)


def _collapse(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def evidence_found(quote, report):
    """Exact substring, or exact after collapsing whitespace (spreadsheets rewrap lines)."""
    if not quote:
        return False
    return quote in report or _collapse(quote) in _collapse(report)


def final_target_correct(state, target):
    """Rubric section 4. None means the text supports no judgement."""
    if state not in JUDGED:
        return None
    return (target >= POSITIVE) == JUDGED[state]


def validate_stage_a(row, report):
    errors = []
    state = row.get("report_states", "")
    if state not in REPORT_STATES:
        return [f"report_states must be one of {REPORT_STATES}"]
    quote = row.get("evidence_quote", "")
    if state == "unmentioned":
        for field in ("evidence_quote", "certainty", "severity_norm", "anatomical_location"):
            if row.get(field, ""):
                errors.append(f"{field} must be empty when the finding is unmentioned")
        return errors
    if not quote:
        errors.append("evidence_quote is required unless the finding is unmentioned")
    elif not evidence_found(quote, report):
        errors.append("evidence_quote is not an exact substring of the report")
    if row.get("certainty", "") not in CERTAINTY:
        errors.append(f"certainty must be one of {CERTAINTY}")
    if row.get("severity_norm", "") not in SEVERITY:
        errors.append(f"severity_norm must be one of {SEVERITY}")
    if state in ("present", "present_below_threshold", "unattributable") and not row.get("anatomical_location", ""):
        errors.append("anatomical_location is required for this state")
    return errors


def _number(value, field, errors):
    try:
        number = float(value)
    except (TypeError, ValueError):
        errors.append(f"{field} must be a number")
        return None
    if not 0.0 <= number <= 1.0:
        errors.append(f"{field} must lie in [0, 1]")
        return None
    return number


def validate_stage_b(a_row, b_row, old_target, old_weight):
    """Rubric sections 4 and 6. Returns (errors, computed final_target_correct)."""
    errors = []
    state = a_row["report_states"]
    correct = final_target_correct(state, old_target)
    if b_row.get("extractor_correct", "") not in EXTRACTOR_CORRECT:
        errors.append(f"extractor_correct must be one of {EXTRACTOR_CORRECT}")
    error_class = b_row.get("error_class", "")
    disposition = b_row.get("disposition", "")
    if error_class not in ERROR_CLASSES:
        errors.append(f"error_class must be one of {ERROR_CLASSES}")
    if disposition not in DISPOSITIONS:
        errors.append(f"disposition must be one of {DISPOSITIONS}")
    if errors:
        return errors, correct
    if disposition != "keep" and not b_row.get("rationale", "").strip():
        errors.append("rationale is required for any change")

    if correct is None:
        if disposition != "keep":
            errors.append(f"a {state} reading supports no change; disposition must be keep")
        if error_class not in ("none", "unresolved", "report_image_disagreement"):
            errors.append("with no judgement from text, error_class must be none, unresolved or report_image_disagreement")
    elif correct is False and error_class == "none":
        errors.append("the final target disagrees with the reading; classify the error (or unresolved)")

    if error_class in ("unresolved", "report_image_disagreement") and disposition != "keep":
        errors.append(f"{error_class} is recorded, never acted on; disposition must be keep")

    if disposition == "correct_target":
        if correct is True:
            errors.append("the final target is already correct; an intermediate error is recorded, not corrected")
        if error_class not in TARGET_CLASSES:
            errors.append(f"a target correction needs error_class in {TARGET_CLASSES}")
        if error_class == "severity_definition" and state not in ("present", "present_below_threshold"):
            errors.append("severity_definition corrections apply to present or below-threshold readings")
        target = _number(b_row.get("proposed_target", ""), "proposed_target", errors)
        if target is not None and state in JUDGED and (target >= POSITIVE) != JUDGED[state]:
            errors.append("proposed_target must lie on the side of 0.5 the reading supports")
        if b_row.get("proposed_weight", ""):
            errors.append("a target correction keeps the weight; propose a weight change separately")
    elif disposition == "correct_weight":
        if error_class != "fusion_weighting":
            errors.append("a weight correction needs error_class fusion_weighting")
        weight = _number(b_row.get("proposed_weight", ""), "proposed_weight", errors)
        if weight is not None and (weight == 0.0 or abs(weight - old_weight) < 1e-9):
            errors.append("proposed_weight must be positive and differ from the current weight")
        if b_row.get("proposed_target", ""):
            errors.append("a weight correction keeps the target")
    elif b_row.get("proposed_target", "") or b_row.get("proposed_weight", ""):
        errors.append("keep carries no proposed values")
    return errors, correct


def change_key(b_row):
    """What a pass proposes, reduced to what two passes must agree on."""
    disposition = b_row["disposition"]
    if disposition == "correct_target":
        return ("correct_target", float(b_row["proposed_target"]) >= POSITIVE)
    if disposition == "correct_weight":
        return ("correct_weight", float(b_row["proposed_weight"]) > float(b_row["old_weight"]))
    return ("keep",)


def agreed_change(first_b, second_b):
    """A change stands only if both passes propose it independently (plan: two-pass rule)."""
    key = change_key(first_b)
    return key != ("keep",) and key == change_key(second_b)


def rereview_due(first_ingested, now):
    return now - first_ingested >= REREVIEW_DELAY
