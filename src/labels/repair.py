"""Proposal rules for the PF OA label repair (plan step 7).

The audit's go decision rests on PF OA, and every PF OA error it confirmed was
a missed positive of one of two kinds:

- **missed mention:** the report states patellofemoral cartilage disease and
  the final target sits near zero, because no extractor recognised the
  wording;
- **severity threshold:** the extractor saw the finding but scored it mild,
  and fusion left the target just under 0.5, while the signed rubric (D2)
  counts chondromalacia from grade II.

This module only *proposes*. It never changes a label: the plan requires
two agreeing reviews for every applied change, so a proposal becomes a
correction only through the review ledger.

The rule reads one clause at a time. A clause proposes PF OA when it names the
patellofemoral compartment (patella, trochlea, retropatellar, ...) and
cartilage disease there, is not negated, is not hedged, and does not grade
the disease below the D2 threshold. Its vocabulary extends the lexicon's
PF OA lists with spellings the lexicon misses (German Chondromalazie,
Croatian hondromalacija, Turkish kondromalazi, Spanish rotula) and with
generic cartilage-lesion words (fissure, defect, fibrillation, thinning).
Negation and hedge detection reuse the lexicon's own, so the two readers
agree on what "not" means.

Pure functions over strings; nothing is printed.
"""

import re

from src.labels import lexicon

PF_PLACES = lexicon.PF_LOCALIZERS + [
    "rotula", "rotul", "patel", "casic", "cashic", "trohle", "trohlej", "trochle",
    "retropatel", "diz kapag", "kniescheibe", "knieschijf", "femoropatel",
]

DISEASE = lexicon.OA_STEMS + [
    "chondromalaz", "hondromalac", "hondromalaz", "kondromalaz", "kondromalas",
    "chondropath", "hondropat", "kondropat", "chondrosis", "condrosis",
    "cartilag", "knorpel", "kraakbeen", "hrskav", "kikirdak", "cartilagin", "хрущял", "χονδρ",
]
LESION = ["fissur", "fisur", "defect", "defekt", "defecto", "ulcer", "fibrillat", "fibrilac",
          "thinn", "adelgaz", "verdunn", "incel", "stanjen", "irregular", "irregul",
          "loss", "perdida", "verlust", "kayb", "gubit", "denud", "erosi", "chondral lesion",
          "lesion", "lezij", "lasion", "изтънен", "λεπτυν"]
# Words that only mean disease when a lesion word is also present ("cartilage").
NEUTRAL_TISSUE = ["cartilag", "knorpel", "kraakbeen", "hrskav", "kikirdak", "cartilagin", "хрущял", "χονδρ"]

GRADE_ONE = re.compile(
    r"\b(grade|grado|grad|stupanj|stupnja|derece|evre|степен|βαθμ\w*|gr\.?)\s*(i|1)\b(?!\s*[-–/]\s*(ii|2))"
    r"|\b(i|1)\.?\s*(grade|grado|grad|stupanj|stupnja|derece|evre|степен)\b")
SOFT_ONLY = re.compile(r"\b(softening|signal change|signal alteration|erweichung|reblandecimiento|"
                       r"mekšanj|mekšanje|yumuşama|омекотяване)\w*")

_PLACE_RE = lexicon._stem_pattern(PF_PLACES)
_DISEASE_RE = lexicon._stem_pattern([s for s in DISEASE if s not in NEUTRAL_TISSUE])
_TISSUE_RE = lexicon._stem_pattern(NEUTRAL_TISSUE)
_LESION_RE = lexicon._stem_pattern(LESION)


def _clauses(text):
    start = 0
    for match in lexicon._CLAUSE_SPLIT.finditer(text):
        yield start, match.start()
        start = match.end()
    yield start, len(text)


def read_pf_oa(report):
    """Clause-level reading of PF OA in one report.

    Returns (state, clause_start, clause_end) for the first asserting clause,
    where state is 'present' or 'below_threshold'; or ('negated'|'hedged'|None, ...)
    when no clause asserts it. Offsets index the normalised text.
    """
    text = lexicon._norm(report)
    fallback = (None, None, None)
    for start, end in _clauses(text):
        clause = text[start:end]
        if not _PLACE_RE.search(clause):
            continue
        disease = _DISEASE_RE.search(clause)
        tissue = _TISSUE_RE.search(clause)
        lesion = _LESION_RE.search(clause) if tissue else None
        # With a neutral tissue word ("cartilage"), the lesion word carries the
        # assertion, so negation is checked where the lesion is named.
        hit = disease or lesion
        if not hit:
            if tissue and SOFT_ONLY.search(clause):
                fallback = ("below_threshold", start, end)
            continue
        position = start + hit.start()
        if lexicon._is_negated(text, position, hit.end() - hit.start()):
            fallback = ("negated", start, end) if fallback[0] is None else fallback
            continue
        if lexicon._UNCERTAIN_RE.search(clause):
            fallback = ("hedged", start, end) if fallback[0] in (None, "negated") else fallback
            continue
        if GRADE_ONE.search(clause) or (SOFT_ONLY.search(clause) and not lesion and not disease):
            fallback = ("below_threshold", start, end)
            continue
        return ("present", start, end)
    return fallback


def propose(report, target, threshold=0.5):
    """A proposal for one study's PF OA label, or None.

    Proposes raising the target only when the rule reads PF OA as present and
    the current final target is below 0.5. The mechanism is named by where the
    target sits: near zero means a missed mention, just under 0.5 means the
    severity threshold. It never proposes lowering a target: the audit found
    no PF OA false positives among the random controls.
    """
    state, start, end = read_pf_oa(report)
    if state != "present" or target >= threshold:
        return None
    return {"mechanism": "missed_mention" if target < 0.2 else "severity_definition",
            "proposed_target": 1.0, "clause": (start, end)}
