"""Prompt construction and response parsing for the LLM report extractor.

This module is deliberately free of GPU and model dependencies so it can be
unit-tested on a laptop. The notebook that runs the model embeds this file's
source directly, so the two cannot drift apart.

Contract matches src/labels/lexicon.py exactly -- `present` is 1, 0, or None,
and `severity` is a float or None -- so both extractors can be scored by the
same calibration code and fused without translation.
"""

import json
import re
from typing import Dict, List, Optional

# Findings, with the definition the model is asked to apply. Ground truth was
# adjudicated with explicit severity thresholds, so the definitions state what
# counts rather than leaving the model to guess a threshold.
FINDING_DEFINITIONS = {
    "ACL": "Anterior cruciate ligament tear, rupture, or significant injury. "
           "Not intact/normal ACL, not a mention of the PCL.",
    # Reverted to the wording that measured better. The sharper radiological
    # phrasing -- surface-reaching signal, explicit exclusions -- was reasoned
    # rather than tested, and cost 0.02 macro and 0.15 on MCL alone.
    "MCL": "Medial collateral ligament tear, rupture, or sprain.",
    "Medial Meniscus": "Tear of the medial (internal) meniscus. Degeneration "
                       "without a tear does not count.",
    "Lateral Meniscus": "Tear of the lateral (external) meniscus. Degeneration "
                        "without a tear does not count.",
    "Medial OA": "Osteoarthritis, cartilage loss, or chondral thinning in the "
                 "MEDIAL femorotibial compartment.",
    "Lateral OA": "Osteoarthritis, cartilage loss, or chondral thinning in the "
                  "LATERAL femorotibial compartment.",
    "PF OA": "Osteoarthritis, chondromalacia, or cartilage loss in the "
             "PATELLOFEMORAL compartment (patella or trochlea).",
    "Effusion": "Joint effusion or increased intra-articular fluid.",
    "Synovitis": "Synovitis or synovial thickening/enhancement.",
    "Baker's": "Baker's cyst or popliteal cyst.",
    "Contusion": "Bone contusion or bone marrow oedema (excluding oedema that "
                 "is purely degenerative subchondral change).",
    "Fracture": "Any fracture, including stress, avulsion, and insufficiency "
                "fractures.",
}

PRESENCE_VALUES = {"yes": 1, "no": 0, "unclear": None}

# Severity and certainty map onto the same 0-1 scale the lexicon produces, so
# fusion compares like with like.
SEVERITY_SCORES = {"severe": 1.0, "moderate": 0.85, "minimal": 0.35, "none": 0.0}
CERTAINTY_FACTORS = {"definite": 1.0, "probable": 0.7, "possible": 0.5}

SYSTEM_PROMPT = (
    "You are a musculoskeletal radiologist extracting structured findings from "
    "knee MRI reports. Reports may be in any language, including English, "
    "Spanish, Turkish, German, Greek, Bulgarian, Croatian, Dutch, French, or "
    "Italian. Read the report in its original language. Do not translate it. "
    "Report only what the text states. Never infer a finding that the report "
    "does not describe."
)


def build_prompt(report: str) -> str:
    """The user turn: the report plus the schema the model must fill in."""
    definitions = "\n".join(
        f'  "{name}": {definition}' for name, definition in FINDING_DEFINITIONS.items()
    )

    return f"""Extract these twelve findings from the knee MRI report below.

FINDINGS:
{definitions}

For each finding return:
  "present": "yes" if the report describes it as present,
             "no" if the report explicitly states it is absent, normal, or intact,
             "unclear" if the report does not mention it at all.
  "severity": "severe", "moderate", "minimal", or "none".
             Use "minimal" for trace, mild, or slight findings.
  "certainty": "definite", "probable", or "possible".
             Use "possible" when the report hedges (suspicious for, cannot exclude).
  "evidence": the exact phrase from the report, in its original language, that
             supports your answer. Use "" when present is "unclear".

Critical: "unclear" and "no" are different. Use "no" ONLY when the report
positively asserts the structure is normal or the finding is absent. If the
report is simply silent about a finding, use "unclear".

REPORT:
\"\"\"
{report}
\"\"\"

Respond with ONLY a JSON object, no commentary, in exactly this form:
{{"ACL": {{"present": "...", "severity": "...", "certainty": "...", "evidence": "..."}}, ...}}
Include all twelve findings."""


def _empty_result() -> Dict[str, Dict]:
    return {
        name: {"present": None, "severity": None, "evidence": None}
        for name in FINDING_DEFINITIONS
    }


def _extract_json_block(text: str) -> Optional[str]:
    """Find the outermost JSON object in a response.

    Models wrap JSON in prose or fenced blocks often enough that locating the
    braces is more reliable than requiring a clean response.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]

    return None


def _score(severity: str, certainty: str) -> float:
    base = SEVERITY_SCORES.get(str(severity).strip().lower(), 0.85)
    factor = CERTAINTY_FACTORS.get(str(certainty).strip().lower(), 1.0)
    return base * factor


def parse_response(text: str) -> Dict[str, Dict]:
    """Turn one model response into the extractor contract.

    Unparseable responses and missing findings resolve to None rather than a
    guess -- a malformed generation must not be read as a confident negative.
    """
    result = _empty_result()
    if not isinstance(text, str) or not text.strip():
        return result

    block = _extract_json_block(text)
    if block is None:
        return result

    try:
        payload = json.loads(block)
    except json.JSONDecodeError:
        # Trailing commas are the usual culprit and are cheap to repair.
        repaired = re.sub(r",\s*([}\]])", r"\1", block)
        try:
            payload = json.loads(repaired)
        except json.JSONDecodeError:
            return result

    if not isinstance(payload, dict):
        return result

    for name in FINDING_DEFINITIONS:
        entry = payload.get(name)
        if not isinstance(entry, dict):
            continue

        raw_present = str(entry.get("present", "")).strip().lower()
        if raw_present not in PRESENCE_VALUES:
            continue

        present = PRESENCE_VALUES[raw_present]
        if present is None:
            result[name] = {"present": None, "severity": None, "evidence": None}
            continue

        severity = (
            _score(entry.get("severity", "moderate"), entry.get("certainty", "definite"))
            if present == 1 else 0.0
        )
        evidence = entry.get("evidence") or None
        result[name] = {
            "present": present,
            "severity": severity,
            "evidence": str(evidence)[:300] if evidence else None,
        }

    return result


def aggregate_samples(samples: List[Dict[str, Dict]]) -> Dict[str, Dict]:
    """Combine self-consistency samples into one result per finding.

    Majority vote decides presence; the vote fraction becomes the confidence
    that fusion uses to weight the sample. Findings where the model disagrees
    with itself are exactly the ones that should carry less weight.
    """
    if not samples:
        return {
            name: {"present": None, "severity": None, "evidence": None,
                   "confidence": 0.0}
            for name in FINDING_DEFINITIONS
        }

    aggregated = {}
    for name in FINDING_DEFINITIONS:
        votes = [s[name]["present"] for s in samples if name in s]
        counts = {
            1: votes.count(1),
            0: votes.count(0),
            None: votes.count(None),
        }
        winner = max(counts, key=lambda key: counts[key])
        confidence = counts[winner] / len(votes) if votes else 0.0

        matching = [
            s[name] for s in samples
            if name in s and s[name]["present"] == winner
        ]
        severities = [m["severity"] for m in matching if m["severity"] is not None]
        evidences = [m["evidence"] for m in matching if m["evidence"]]

        aggregated[name] = {
            "present": winner,
            "severity": (sum(severities) / len(severities)) if severities else None,
            "evidence": evidences[0] if evidences else None,
            "confidence": confidence,
        }

    return aggregated


def extract_finding_from_record(record: Dict, finding: str) -> Dict:
    """Adapter so calibrate.score_extractor can score cached LLM output."""
    entry = record.get(finding, {})
    return {
        "present": entry.get("present"),
        "severity": entry.get("severity"),
        "span": entry.get("evidence"),
    }


# ---------------------------------------------------------------------------
# Probability extraction
# ---------------------------------------------------------------------------
# The categorical contract above collapses 58 gold studies onto ~10-20 distinct
# target values, and 53% of all label slots share a single one. AUC is a
# ranking metric: a target that cannot order the studies inside a tie block
# gives the model nothing to learn there, and caps the achievable score.
#
# Asking for a probability instead produces a continuous score per study, which
# breaks the ties -- including for unmentioned findings, where the model can
# still reason from context ("severe tricompartmental OA" makes an unmentioned
# effusion more likely than a report describing an otherwise normal knee).

PROBABILITY_PROMPT = """Read the knee MRI report and estimate, for each of the
twelve findings, the probability that a musculoskeletal radiologist reviewing
the images would mark it present.

FINDINGS:
{definitions}

Give a probability from 0 to 100 for each.

How to think about it:
- Explicitly described as present: high, 85-99, scaled by how definite and how
  severe the description is. A "large joint effusion" is higher than a "trace
  effusion", because the reference standard applies severity thresholds.
- Explicitly described as absent, normal, or intact: low, 1-10.
- Hedged ("possible", "suspicious for", "cannot exclude"): middle, 30-60.
- Not mentioned at all: this is NOT the same as absent. Reports state positives
  and omit much else, and what gets omitted varies by finding and by
  radiologist. Estimate from the rest of the report and from how routinely that
  finding is reported. A report describing severe degenerative change makes an
  unmentioned effusion more likely; a detailed normal study makes it less so.
  Most unmentioned findings land between 5 and 35.

Do not round to 0, 50, or 100. Distinguishing two studies that both fail to
mention a finding is exactly what this is for.

REPORT:
\"\"\"
{report}
\"\"\"

Respond with ONLY a JSON object mapping each finding to a number:
{{"ACL": 3, "MCL": 12, "Medial Meniscus": 88, ...}}
Include all twelve findings."""


# ---------------------------------------------------------------------------
# Prompt variants (Phase C — extraction v3)
# ---------------------------------------------------------------------------
# Each candidate changes ONE idea against the baseline, so a gold-sweep delta
# is attributable to it. Variants are built by anchored substitution on the
# baseline text, and the anchors are asserted below: rewording the baseline
# without updating the variants fails at import instead of silently shipping
# a variant that no longer differs from it.
#
# Nothing here ships without a gold measurement — two reasoned prompt changes
# have already measured worse (see PROJECT.md). The weak findings these aim
# at, from the fused labels: Synovitis 0.792, Lateral OA and PF OA next.

_SILENCE_ANCHOR = "Most unmentioned findings land between 5 and 35."

_CORRELATES_BLOCK = _SILENCE_ANCHOR + """
  Co-occurrence is your best evidence for silent findings, and these
  associations are strong in knee MRI:
  - Synovitis is rarely dictated but usually accompanies a joint effusion or
    active osteoarthritis. An effusion or advanced OA with no comment on the
    synovium deserves a clearly raised estimate, not a floor value.
  - Osteoarthritis travels across compartments: described medial OA raises
    the probability of unmentioned patellofemoral and lateral OA.
  - Bone contusion accompanies acute injury (an ACL rupture, a recent
    trauma); a purely chronic degenerative report lowers it.
  - Effusion is raised by any acute internal derangement or active OA.
  A correlate never overrides an explicit statement: a structure described
  as normal stays low regardless of the rest of the report."""

_ROUNDING_ANCHOR = "Do not round to 0, 50, or 100."

_GRANULAR_BLOCK = """Use precise integers — 7, 23, 41, 88 — never multiples
of 5 or 10, and avoid giving two findings the same number unless the report
truly says the same about both. The scores are used to RANK studies, so ties
are wasted answers. Do not round to 0, 50, or 100."""

_RESPOND_ANCHOR = "Respond with ONLY a JSON object mapping each finding to a number:"

_REASON_FIRST_BLOCK = """First write two or three sentences (plain prose, no
braces) sorting the findings: which the report describes as present, which it
calls absent or normal, which it hedges on, and which it never mentions.
Then output the JSON object mapping each finding to a number:"""

_VARIANT_EDITS = {
    # One idea each: named co-occurrence priors for silent findings;
    # pressure against tie blocks; a short sorting step before the numbers.
    "correlates": (_SILENCE_ANCHOR, _CORRELATES_BLOCK),
    "granular": (_ROUNDING_ANCHOR, _GRANULAR_BLOCK),
    "reason_first": (_RESPOND_ANCHOR, _REASON_FIRST_BLOCK),
}

PROBABILITY_PROMPT_VARIANTS = {"baseline": PROBABILITY_PROMPT}
for _name, (_anchor, _replacement) in _VARIANT_EDITS.items():
    assert _anchor in PROBABILITY_PROMPT, f"anchor for {_name!r} lost"
    PROBABILITY_PROMPT_VARIANTS[_name] = PROBABILITY_PROMPT.replace(
        _anchor, _replacement)
    assert PROBABILITY_PROMPT_VARIANTS[_name] != PROBABILITY_PROMPT


def build_probability_prompt(report: str, variant: str = "baseline") -> str:
    """The probability-valued variant of the extraction prompt."""
    definitions = "\n".join(
        f'  "{name}": {definition}' for name, definition in FINDING_DEFINITIONS.items()
    )
    template = PROBABILITY_PROMPT_VARIANTS[variant]
    return template.format(definitions=definitions, report=report)


def parse_probabilities(text: str) -> Dict[str, Optional[float]]:
    """Parse a probability response into {finding: 0..1} or None per finding.

    A missing or unparseable finding stays None rather than defaulting: a
    fabricated 0.5 would sit in the middle of the ranking and is worse than an
    honest gap the caller can fill deliberately.
    """
    result = {name: None for name in FINDING_DEFINITIONS}
    if not isinstance(text, str) or not text.strip():
        return result

    block = _extract_json_block(text)
    if block is None:
        return result

    try:
        payload = json.loads(block)
    except json.JSONDecodeError:
        try:
            payload = json.loads(re.sub(r",\s*([}\]])", r"\1", block))
        except json.JSONDecodeError:
            return result

    if not isinstance(payload, dict):
        return result

    for name in FINDING_DEFINITIONS:
        value = payload.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            continue
        try:
            number = float(str(value).strip().rstrip("%"))
        except ValueError:
            continue
        if not 0.0 <= number <= 100.0:
            continue
        result[name] = number / 100.0

    return result


def aggregate_probabilities(samples: List[Dict[str, Optional[float]]]) -> Dict[str, Dict]:
    """Average self-consistency samples, keeping their spread as a weight.

    Agreement across samples is evidence the estimate is stable; disagreement
    is a reason to weight that target down during training.
    """
    aggregated = {}
    for name in FINDING_DEFINITIONS:
        values = [s[name] for s in samples if s.get(name) is not None]
        if not values:
            aggregated[name] = {"probability": None, "confidence": 0.0}
            continue

        mean = sum(values) / len(values)
        if len(values) > 1:
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            spread = variance ** 0.5
        else:
            spread = 0.0
        aggregated[name] = {
            "probability": mean,
            # Full weight at perfect agreement, falling as the samples diverge.
            "confidence": max(0.0, 1.0 - 2.0 * spread),
        }
    return aggregated
