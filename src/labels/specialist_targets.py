"""Training targets for the Experiment 11 knee specialists.

The specialist targets are intentionally stricter than the general-purpose
proxy labels.  An absent report mention is unknown, so it has zero loss
weight.  Only a definite assertion supervises the image model.  A second,
independently produced proxy score is used as corroboration: agreement raises
confidence while disagreement lowers it, but the report assertion remains the
main source of the target value.

The output keeps the project's twelve-column interface.  Labels outside the
requested specialist group are neutral and masked, which lets the existing
dataset and loss code train a three-head specialist without a parallel data
path.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from src import config


SYNOVITIS_GROUP = ("Synovitis", "Effusion", "Baker's")
PFOA_GROUP = ("PF OA", "Medial OA", "Lateral OA")

# Confidence is attached to the *evidence*, not to class prevalence.  These
# values are deliberately fixed before image evaluation.
AGREEMENT_WEIGHT = 1.0
DISAGREEMENT_WEIGHT = 0.4
SINGLE_SOURCE_WEIGHT = 0.7

# Definite positive assertions retain an ordinal signal.  Ungraded positives
# stay strongly positive without being treated as equivalent to severe disease.
SEVERITY_TARGETS = {
    "minimal": 0.20,
    "mild": 0.40,
    "moderate": 0.70,
    "severe": 1.00,
    "not stated": 0.85,
}

_MINIMAL_TEXT = re.compile(
    r"\b(minimal|minimo|minima|trace|tiny|punctate|millimetric)\b")
_MILD_TEXT = re.compile(r"\b(mild|slight|low[ -]?grade|grade 1|grade i)\b")


def _normalise_text(value: object) -> str:
    text = "" if value is None else str(value).strip().lower()
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def severity_band(assertion: Mapping[str, object]) -> str:
    """Return the most specific supported severity band.

    The annotation schema historically collapsed both ``minimal`` and
    ``mild`` into ``severity_norm=minimal``.  Preserve the distinction when
    the original ``severity_text`` states it; otherwise keep the conservative
    normalized band.
    """
    norm = _normalise_text(assertion.get("severity_norm"))
    text = _normalise_text(assertion.get("severity_text"))
    if norm == "minimal":
        if _MINIMAL_TEXT.search(text):
            return "minimal"
        if _MILD_TEXT.search(text):
            return "mild"
        # Annotators use the normalized minimal bucket for multilingual mild
        # terms.  The midpoint avoids pretending the source is more precise.
        return "mild"
    if norm in ("moderate", "severe"):
        return norm
    return "not stated"


def assertion_target(finding: str, assertion: Mapping[str, object]) -> tuple[float, bool]:
    """Translate one assertion to ``(target, supervised)``.

    Possible/probable assertions and missing mentions are masked.  A definite
    below-threshold PF cartilage finding remains an intermediate target rather
    than becoming a hard negative; this is the intended soft-label treatment
    for mild chondropathy.
    """
    state = str(assertion.get("report_states", ""))
    certainty = str(assertion.get("certainty", ""))
    if certainty != "definite":
        return 0.5, False
    if state == "present":
        return SEVERITY_TARGETS[severity_band(assertion)], True
    if state == "present_below_threshold":
        return (0.30 if finding == "PF OA" else 0.20), True
    if state == "negated":
        return 0.0, True
    if state in ("uncertain", "unmentioned", "unattributable"):
        return 0.5, False
    raise ValueError(f"Unknown report state for {finding}: {state!r}")


def _corroborated_target(
    evidence_target: float,
    base_target: object,
    base_weight: object,
) -> tuple[float, float, str]:
    """Fuse an assertion with the existing proxy only when it is informative."""
    if (base_target is None or base_weight is None
            or not np.isfinite(float(base_target))
            or not np.isfinite(float(base_weight))
            or float(base_weight) <= 0):
        return evidence_target, SINGLE_SOURCE_WEIGHT, "assertion_only"

    base = float(np.clip(float(base_target), 0.0, 1.0))
    agrees = (evidence_target >= 0.5) == (base >= 0.5)
    if not agrees:
        return evidence_target, DISAGREEMENT_WEIGHT, "disagree"

    # The reviewed assertion keeps 80% of the target value.  The older source
    # is corroboration rather than authority and cannot turn a severity level
    # into a hard binary target.
    return float(0.8 * evidence_target + 0.2 * base), AGREEMENT_WEIGHT, "agree"


def build_specialist_targets(
    assertions: Mapping[str, Mapping[str, Mapping[str, object]]],
    base_labels: pd.DataFrame,
    active_labels: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build masked twelve-column targets and a per-assertion audit table."""
    active = tuple(active_labels)
    if not active or len(active) != len(set(active)) or not set(active) <= set(config.LABELS):
        raise ValueError("active_labels must be unique competition labels")
    if config.ID_COL not in base_labels or base_labels[config.ID_COL].duplicated().any():
        raise ValueError("base labels need unique study identities")
    required = set(config.LABELS) | {f"{name}_weight" for name in config.LABELS}
    missing = required - set(base_labels.columns)
    if missing:
        raise ValueError("base labels are missing columns: " + ", ".join(sorted(missing)))

    base = base_labels.set_index(config.ID_COL, drop=False)
    unknown = set(assertions) - set(base.index.astype(str))
    if unknown:
        raise ValueError(f"{len(unknown)} assertion studies are absent from base labels")

    rows, audit = [], []
    for uid in sorted(assertions):
        findings = assertions[uid]
        if set(findings) != set(config.LABELS):
            raise ValueError(f"Incomplete finding inventory for {uid}")
        row = {config.ID_COL: uid}
        for name in config.LABELS:
            row[name], row[f"{name}_weight"] = 0.5, 0.0
        for name in active:
            evidence, supervised = assertion_target(name, findings[name])
            if not supervised:
                audit.append({
                    config.ID_COL: uid, "finding": name,
                    "report_state": findings[name].get("report_states", ""),
                    "severity_band": "unknown", "target": 0.5,
                    "weight": 0.0, "agreement": "masked",
                })
                continue
            target, weight, agreement = _corroborated_target(
                evidence, base.at[uid, name], base.at[uid, f"{name}_weight"])
            row[name], row[f"{name}_weight"] = target, weight
            audit.append({
                config.ID_COL: uid, "finding": name,
                "report_state": findings[name].get("report_states", ""),
                "severity_band": (severity_band(findings[name])
                                  if findings[name].get("report_states") == "present"
                                  else "below threshold" if findings[name].get("report_states") == "present_below_threshold"
                                  else "negative"),
                "target": target, "weight": weight, "agreement": agreement,
            })
        rows.append(row)

    return pd.DataFrame(rows), pd.DataFrame(audit)
