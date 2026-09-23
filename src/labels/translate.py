"""Translation aid for the report review: prompt, self-check and output checks.

Runs inside `notebooks/22_translate_packets_colab.ipynb` on the participant's
Colab GPU with an open-weight model. No report text goes to a hosted service.
The notebook embeds this file verbatim, so the two can't drift apart.

A translation is an aid, never evidence: the ledger's quote is always taken
from the original report. The dangerous failure is a fluent translation that
drops a negation, a hedge or a side, so before any report is translated the
model must pass `SELF_CHECKS`. These are synthetic sentences written for this
check, not report text, and each one names the English terms a faithful
translation must keep.
"""

import hashlib
import re

LANGUAGE_NAMES = {"el": "Greek", "de": "German", "tr": "Turkish", "es": "Spanish",
                  "hr": "Croatian", "bg": "Bulgarian", "sr": "Serbian", "ru": "Russian",
                  "it": "Italian", "fr": "French", "nl": "Dutch", "pt": "Portuguese",
                  "cyrillic?": "a Cyrillic-script language", "unknown": "the source language"}

SYSTEM = ("You translate knee MRI reports into English for a reviewer who will check every "
          "statement against the original. Translate faithfully and completely. Keep every "
          "negation (no, not, without, intact), every hedge (possible, suspected, probable, "
          "cannot be excluded), every side and compartment (left, right, medial, lateral, "
          "internal, external, anterior, posterior), every grade and number, and every "
          "anatomical term. Do not add, omit, summarise, correct or interpret anything. "
          "Output only the English translation.")

# (language, synthetic sentence, groups of acceptable English terms; every group must match)
SELF_CHECKS = [
    ("de", "Kein Riss des lateralen Meniskus.",
     [("no ", "without", "not "), ("lateral",), ("tear", "rupture")]),
    ("es", "Posible rotura del menisco lateral.",
     [("possible", "possibly", "suspected", "may "), ("lateral",), ("tear", "rupture")]),
    ("el", "Ρήξη του έξω μηνίσκου.",
     [("tear", "rupture"), ("lateral", "external", "outer")]),
    ("tr", "Medial menisküs arka boynuzunda yırtık izlenmemiştir.",
     [("no ", "not ", "without"), ("medial",), ("tear", "rupture"), ("posterior",)]),
    ("hr", "Edem koštane srži lateralnog femoralnog kondila.",
     [("marrow",), ("edema", "oedema"), ("lateral",), ("condyle",)]),
    ("bg", "Хондромалация на пателата II степен.",
     [("chondromalacia",), ("patella",), ("ii", "2", "second")]),
    ("it", "Non si può escludere una contusione ossea.",
     [("cannot be excluded", "cannot exclude", "can't be excluded", "can not be excluded",
       "not be ruled out", "cannot be ruled out", "possible"), ("bone",),
      ("contusion", "bruise")]),
]


def messages(report, language):
    source = LANGUAGE_NAMES.get(language, "the source language")
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Translate this report from {source} into English.\n\n{report}"}]


def clean(text):
    text = (text or "").strip()
    return re.sub(r"^(english translation|translation)\s*:\s*", "", text, flags=re.I).strip()


def missing_terms(translation, groups):
    """The groups a translation fails to cover; empty when it keeps everything."""
    lowered = " " + clean(translation).lower() + " "
    return [group for group in groups if not any(term in lowered for term in group)]


def self_check(translate_batch):
    """Run the synthetic checks through `translate_batch(list of (text, language)) -> texts`.

    Returns one failure per check that lost a required term, or an empty list.
    """
    outputs = translate_batch([(text, language) for language, text, _ in SELF_CHECKS])
    failures = []
    for (language, text, groups), output in zip(SELF_CHECKS, outputs):
        lost = missing_terms(output, groups)
        if lost:
            failures.append({"language": language, "source": text, "translation": clean(output),
                             "missing": [" / ".join(g).strip() for g in lost]})
    return failures


def check_results(requests, results):
    """One non-empty, untruncated translation per request, by case id.

    `results` rows carry case_id, translation and finish_reason. Returns
    problems by case id; never echoes text.
    """
    wanted = {r["case_id"] for r in requests}
    got = {}
    problems = []
    for row in results:
        if row["case_id"] in got:
            problems.append(f"{row['case_id']}: duplicate translation")
        got[row["case_id"]] = row
    for case_id in sorted(wanted - set(got)):
        problems.append(f"{case_id}: missing translation")
    for case_id in sorted(set(got) - wanted):
        problems.append(f"{case_id}: translation for an unknown case")
    for case_id in sorted(wanted & set(got)):
        row = got[case_id]
        if not clean(row.get("translation")):
            problems.append(f"{case_id}: empty translation")
        if row.get("finish_reason") == "length":
            problems.append(f"{case_id}: truncated at the token limit")
    return problems


def report_hash(report):
    return hashlib.sha256(str(report).encode()).hexdigest()
