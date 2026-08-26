"""Score a categorical re-extraction against the shipped 14B, fused.

Phase C lever 3: the categorical half of fusion — the half that carries
Medial Meniscus, ACL, PF OA, MCL, Lateral Meniscus and Baker's — is still
from the 14B. Notebook 03 section 6b runs the categorical prompt on the gold
58 with the bigger model and writes `llm_labels_gold__32b.parquet`; drop it
into `data/meta/gold_sweep/` and run:

    python3 tools/score_categorical_upgrade.py

Both categorical frames are scored through the identical full fusion (the
shipped probability frame, score_average, lexicon blend), so the only moving
part is the categorical extractor. Gate, same as every label change: fused
macro > shipped + 0.005 with no finding below 0.75. A PASS earns the ~1 h
categorical corpus re-run; a fail closes Phase C entirely.
"""

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src import config                                    # noqa: E402
from src.labels import calibrate, eval_fusion, fuse, mode_fusion  # noqa: E402

CANDIDATE = config.META_DIR / "gold_sweep" / "llm_labels_gold__32b.parquet"
STRATEGY = "score_average"
GATE_MARGIN = 0.005
GATE_FLOOR = 0.75

# Where the categorical mode carries fusion (measured 2026-08-26: the
# probability mode adds +0.00..+0.01 here). This is the ground a categorical
# upgrade must actually move.
CATEGORICAL_GROUND = ["Medial Meniscus", "ACL", "PF OA", "MCL",
                      "Lateral Meniscus", "Baker's"]


def fused_aucs(gold, cat_gold, prob_gold, lex) -> pd.Series:
    scores = mode_fusion.combine(cat_gold, prob_gold, STRATEGY)
    return eval_fusion._score_frame_aucs(gold, fuse.fuse_scores(lex, scores))


def main():
    if not CANDIDATE.exists():
        raise SystemExit(
            f"{CANDIDATE} not found. Run notebook 03 section 6b on Colab "
            "and drop the parquet there first.")

    gold = calibrate.load_gold()
    lex = fuse.lexicon_frame(gold["Report"], gold[config.ID_COL])
    prob = eval_fusion._align(
        pd.read_parquet(config.META_DIR / "llm_probs_gold.parquet"), gold)
    cat_14b = eval_fusion._align(
        pd.read_parquet(config.META_DIR / "llm_labels_gold.parquet"), gold)
    cat_32b = eval_fusion._align(pd.read_parquet(CANDIDATE), gold)

    silent_14b = {n: int(cat_14b[n].isna().sum()) for n in config.LABELS}
    silent_32b = {n: int(cat_32b[n].isna().sum()) for n in config.LABELS}

    columns = {
        "cat14b only": eval_fusion._score_frame_aucs(
            gold, mode_fusion.from_categorical(cat_14b)),
        "cat32b only": eval_fusion._score_frame_aucs(
            gold, mode_fusion.from_categorical(cat_32b)),
        "fused 14b": fused_aucs(gold, cat_14b, prob, lex),
        "fused 32b": fused_aucs(gold, cat_32b, prob, lex),
    }
    reference = float(np.nanmean(columns["fused 14b"].to_numpy()))
    candidate = float(np.nanmean(columns["fused 32b"].to_numpy()))

    names = list(columns)
    print(f"{'finding':18s} " + "  ".join(f"{n:>11s}" for n in names)
          + f"  {'silent 14b>32b':>14s}")
    print("-" * (20 + 13 * len(names) + 16))
    for finding in config.LABELS:
        marker = " *" if finding in CATEGORICAL_GROUND else ""
        print(f"{finding + marker:18s} " + "  ".join(
            f"{columns[n][finding]:11.3f}" for n in names)
            + f"  {silent_14b[finding]:>6d} > {silent_32b[finding]:<5d}")
    print("-" * (20 + 13 * len(names) + 16))
    print(f"{'macro':18s} " + "  ".join(
        f"{np.nanmean(columns[n].to_numpy()):11.3f}" for n in names))
    print("\n* categorical ground -- the six findings the probability mode "
          "cannot improve; the upgrade must win here or it wins nowhere.")

    print("categorical-ground fused delta: " + "  ".join(
        f"{name} "
        f"{float(columns['fused 32b'][name] - columns['fused 14b'][name]):+.3f}"
        for name in CATEGORICAL_GROUND))

    floor = float(columns["fused 32b"].min())
    weakest = columns["fused 32b"].idxmin()
    clears = (candidate > reference + GATE_MARGIN) and floor >= GATE_FLOOR
    print(f"\ngate: fused macro > {reference:.3f} + {GATE_MARGIN} "
          f"and every finding >= {GATE_FLOOR}")
    detail = f"macro {candidate:.3f} ({candidate - reference:+.3f})"
    if floor < GATE_FLOOR:
        detail += f"; {weakest} {floor:.3f} < {GATE_FLOOR}"
    print(f"  cat32b {'PASSES' if clears else 'fails'}  ({detail})")

    if clears:
        print("\nPASS: the categorical corpus re-run (~1 h, notebook 03 with "
              "section 6b's model) is earned. Retrain gates on gold as usual.")
    else:
        print("\nFail: Phase C closes -- every extraction lever is now "
              "measured. Record it and put the budget on Phase D.")


if __name__ == "__main__":
    main()
