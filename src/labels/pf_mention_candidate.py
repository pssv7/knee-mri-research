"""Conservative report-review proposals only; no training-label mutation.

This freezes a narrower version of the existing PF rule. It abstains on grade
language rather than adopting the participant's grade-II threshold, and on
clauses mentioning other compartments. Its validity requires fresh text review.
"""
import re

from src.labels import lexicon, repair

VERSION = 'pf_explicit_mention_v2'
GRADE = re.compile(r'\b(?:grad\w*|grado\w*|stup\w*|derece\w*|evre\w*|степен\w*|βαθμ\w*)\b')
OTHER_ANATOMY = re.compile(r'\b(?:menisc\w*|menisk\w*|tibio\w*|femorotib\w*|condyl\w*|kondil\w*|kondyl\w*)\b')


def propose(report, target):
    if not 0 <= float(target) < .2:
        return None
    old = repair.propose(report, target)
    if old is None or old['mechanism'] != 'missed_mention':
        return None
    text = lexicon._norm(report)
    start, end = old['clause']
    clause = text[start:end]
    # Roman-numeral punctuation may split a grade across clause boundaries.
    if GRADE.search(text) or OTHER_ANATOMY.search(clause):
        return None
    # Hedge words elsewhere in the same report can qualify the assertion.
    # This deliberately sacrifices coverage until multilingual scope is reviewed.
    if lexicon._UNCERTAIN_RE.search(text):
        return None
    return {'finding': 'PF OA', 'mechanism': 'explicit_mention_review_required',
            'rule_version': VERSION, 'normalized_evidence': clause,
            'suggested_target': 1.0, 'action': 'review_only'}
