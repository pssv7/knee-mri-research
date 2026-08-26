"""Print the categorical extractor's disagreements with the gold 58, to read.

Every other tool here reduces extraction quality to a number. This one does
the opposite: it finds the studies where the extractor and the radiologist
disagree and prints them with enough context to judge *why* — the model's
own evidence span, the sentences a rule-based reader flags for that finding,
and what the other model said about the same study.

    python3 tools/inspect_extraction_errors.py                 # all findings
    python3 tools/inspect_extraction_errors.py Synovitis       # one finding
    python3 tools/inspect_extraction_errors.py --all-cases     # no per-finding cap

Reports are multilingual and are printed in their original language, which
is how they were read; do not translate before judging a span.

Four error classes, kept separate because they have different causes and
only three of them are anyone's fault:

  UNREACHABLE      gold says present, and no sentence in the report mentions
                   the finding's anatomy at all. The radiologist marked it
                   from the images; the text never discusses it. No prompt
                   and no model can extract what was not written -- this is
                   a ceiling, not a defect.
  MISSED (in text) gold says present, the anatomy *is* discussed, and the
                   extractor still stayed silent. A vocabulary or phrasing
                   gap, and the fixable half of the misses.
  MISSED (said no) asserted absence against a positive gold -- usually
                   negation scope.
  FALSE POSITIVE   asserted presence against a negative gold -- usually a
                   mention of the anatomy in a different clause.

The reachability split is measured with the lexicon's own anatomy patterns,
so it is a *lower bound* on coverage: a report describing a finding in
words the lexicon does not know counts as unreachable when it was merely
unusual. Read the printed sentences before trusting the label.

Within each class, cases are ordered by the extractor's own confidence, so
the ones it was *sure* about come first. Those are the diagnostic ones: a
0.33-confidence miss is the sampler disagreeing with itself, while a
1.0-confidence miss is a systematic misreading.
"""

import pathlib
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src import config                                    # noqa: E402
from src.labels import calibrate, eval_fusion, lexicon, mode_fusion  # noqa: E402

SHIPPED = config.META_DIR / "llm_labels_gold.parquet"          # the 14B
COMPARISON = config.META_DIR / "gold_sweep" / "llm_labels_gold__32b.parquet"
MAX_CASES = 6           # per finding, unless --all-cases
SNIPPET_CHARS = 220


def sentences(report: str):
    """Split a report into readable fragments without losing the original."""
    if not isinstance(report, str):
        return []
    parts = re.split(r"(?<=[.;:])\s+|\n+", report)
    return [p.strip() for p in parts if p and p.strip()]


def anatomy_hits(report: str, finding: str, limit: int = 3):
    """Sentences a rule-based reader would flag for this finding.

    Uses the lexicon's own compiled anatomy pattern, so a MISSED case where
    this finds nothing means the vocabulary is absent from the report,
    while one where it finds plenty means the extractor read it and decided
    against it.
    """
    pattern = lexicon._COMPILED[finding]["anatomy"]
    hits = [s for s in sentences(report) if pattern.search(lexicon._norm(s))]
    return hits[:limit]


def call_of(frame: pd.DataFrame, row: int, finding: str):
    present = frame[finding].iloc[row]
    severity = frame.get(f"{finding}_severity", pd.Series(dtype=float))
    confidence = frame.get(f"{finding}_confidence", pd.Series(dtype=float))
    evidence = frame.get(f"{finding}_evidence", pd.Series(dtype=object))
    return {
        "present": None if pd.isna(present) else int(present),
        "severity": None if len(severity) == 0 or pd.isna(severity.iloc[row])
                    else float(severity.iloc[row]),
        "confidence": None if len(confidence) == 0 or pd.isna(confidence.iloc[row])
                      else float(confidence.iloc[row]),
        "evidence": None if len(evidence) == 0 or pd.isna(evidence.iloc[row])
                    else str(evidence.iloc[row]),
    }


def label_call(call):
    if call["present"] is None:
        return "silent"
    verdict = "YES" if call["present"] == 1 else "no"
    bits = [verdict]
    if call["severity"] is not None:
        bits.append(f"sev {call['severity']:.2f}")
    if call["confidence"] is not None:
        bits.append(f"conf {call['confidence']:.2f}")
    return " ".join(bits)


def classify(present, truth, in_text):
    if truth == 1 and present is None:
        return "MISSED (in text)" if in_text else "UNREACHABLE"
    if truth == 1 and present == 0:
        return "MISSED (said no)"
    if truth == 0 and present == 1:
        return "FALSE POSITIVE"
    return None


def summarise(gold, frame):
    scores = eval_fusion._score_frame_aucs(
        gold, mode_fusion.from_categorical(frame))
    rows = []
    for finding in config.LABELS:
        truth = gold[finding].to_numpy().astype(int)
        present = [call_of(frame, i, finding)["present"] for i in range(len(gold))]
        in_text = [bool(anatomy_hits(r, finding, limit=1))
                   for r in gold["Report"]]
        errors = [classify(p, t, m)
                  for p, t, m in zip(present, truth, in_text)]
        covered = [m for m, t in zip(in_text, truth) if t == 1]
        rows.append({
            "finding": finding,
            "n+": int(truth.sum()),
            "coverage": float(np.mean(covered)) if covered else float("nan"),
            "unreachable": errors.count("UNREACHABLE"),
            "missed_text": errors.count("MISSED (in text)"),
            "missed_no": errors.count("MISSED (said no)"),
            "false_pos": errors.count("FALSE POSITIVE"),
            "auc": float(scores[finding]),
        })
    return pd.DataFrame(rows).sort_values("auc")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    show_all = "--all-cases" in sys.argv

    gold = calibrate.load_gold()
    frame = eval_fusion._align(pd.read_parquet(SHIPPED), gold)
    other = (eval_fusion._align(pd.read_parquet(COMPARISON), gold)
             if COMPARISON.exists() else None)

    table = summarise(gold, frame)
    print("14B categorical extractor vs the gold 58, worst AUC first")
    print("'in report' = share of gold positives whose report mentions the "
          "anatomy at all")
    print(f"\n{'finding':18s} {'n+':>3s} {'AUC':>6s} {'in report':>10s} | "
          f"{'unreach':>7s} {'miss-txt':>8s} {'miss-no':>7s} {'falsepos':>8s}")
    print("-" * 80)
    for _, r in table.iterrows():
        print(f"{r['finding']:18s} {r['n+']:3d} {r['auc']:6.3f} "
              f"{r['coverage']:9.0%} | {r['unreachable']:7d} "
              f"{r['missed_text']:8d} {r['missed_no']:7d} {r['false_pos']:8d}")
    print("-" * 80)
    total = table[["unreachable", "missed_text", "missed_no", "false_pos"]].sum()
    print(f"{'total':18s} {'':3s} {'':6s} {'':10s} | {total['unreachable']:7d} "
          f"{total['missed_text']:8d} {total['missed_no']:7d} "
          f"{total['false_pos']:8d}")
    reachable_misses = total["missed_text"] + total["missed_no"]
    print(f"\nOf {total['unreachable'] + reachable_misses} missed positives, "
          f"{total['unreachable']} were never written down "
          f"({total['unreachable'] / (total['unreachable'] + reachable_misses):.0%}) "
          f"-- a ceiling, not an extraction defect.")

    findings = args if args else list(table["finding"])
    unknown = [f for f in findings if f not in config.LABELS]
    if unknown:
        raise SystemExit(f"unknown finding(s): {unknown}\n{config.LABELS}")

    for finding in findings:
        truth = gold[finding].to_numpy().astype(int)
        cases = []
        for i in range(len(gold)):
            call = call_of(frame, i, finding)
            kind = classify(call["present"], truth[i],
                            bool(anatomy_hits(gold["Report"].iloc[i], finding, 1)))
            if kind:
                cases.append((kind, call["confidence"] or 0.0, i, call))
        if not cases:
            continue

        order = {"MISSED (said no)": 0, "MISSED (in text)": 1,
                 "FALSE POSITIVE": 2, "UNREACHABLE": 3}
        cases.sort(key=lambda c: (order[c[0]], -c[1]))
        shown = cases if show_all else cases[:MAX_CASES]

        auc = float(table.set_index("finding").loc[finding, "auc"])
        print(f"\n{'=' * 78}\n{finding}  (AUC {auc:.3f}, {len(cases)} "
              f"disagreements of {len(gold)})\n{'=' * 78}")

        for kind, _, i, call in shown:
            uid = str(gold[config.ID_COL].iloc[i])
            report = gold["Report"].iloc[i]
            print(f"\n-- {kind}   study {uid[:16]}...   "
                  f"gold={truth[i]}   14B: {label_call(call)}")
            if other is not None:
                print(f"   32B: {label_call(call_of(other, i, finding))}")
            if call["evidence"]:
                print(f"   evidence cited: {call['evidence'][:SNIPPET_CHARS]}")
            hits = anatomy_hits(report, finding)
            if hits:
                for h in hits:
                    print(f"   report says:    {h[:SNIPPET_CHARS]}")
            else:
                print("   report says:    (no sentence mentions this anatomy "
                      "-- the finding may be genuinely unstated)")

        if len(cases) > len(shown):
            print(f"\n   ... {len(cases) - len(shown)} more; "
                  f"rerun with --all-cases or `{finding}` to see them")


if __name__ == "__main__":
    main()
