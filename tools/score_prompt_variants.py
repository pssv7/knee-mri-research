"""Score the gold-sweep dumps from notebook 03 against the 0.891 gate.

Runs entirely locally in seconds. Notebook 03's section 6 writes one
`llm_probs_gold__<variant>_n<samples>.parquet` per sweep configuration; drop
them into `data/meta/gold_sweep/` and run:

    python3 tools/score_prompt_variants.py

For each dump this computes the number that actually gates a corpus re-run:
the fused gold macro — probability scores combined with the categorical
extraction under the frozen `score_average` strategy, then blended with the
lexicon by `fuse.fuse_scores` — the figure comparable to the shipped labels'
0.891. The raw probability macro is printed as context only.

Gate (docs/plan.md, Phase C): fused macro must beat the shipped labels by
more than 0.005 with no finding below 0.75. The shipped dump
(`data/meta/llm_probs_gold.parquet`) is always scored as the reference row,
so the gate threshold is measured, not remembered.
"""

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src import config                                    # noqa: E402
from src.labels import calibrate, eval_fusion, fuse, mode_fusion  # noqa: E402

SWEEP_DIR = config.META_DIR / "gold_sweep"
STRATEGY = "score_average"      # frozen 2026-08-24; see docs/worklog.md
GATE_MARGIN = 0.005
GATE_FLOOR = 0.75


def fused_aucs(gold: pd.DataFrame, cat_gold: pd.DataFrame,
               prob_gold: pd.DataFrame) -> pd.Series:
    """Per-finding AUC of the full fusion, the shape the gate needs."""
    lex_gold = fuse.lexicon_frame(gold["Report"], gold[config.ID_COL])
    scores = mode_fusion.combine(cat_gold, prob_gold, STRATEGY)
    fused = fuse.fuse_scores(lex_gold, scores)
    return eval_fusion._score_frame_aucs(gold, fused)


def main():
    gold = calibrate.load_gold()
    cat_gold = eval_fusion._align(
        pd.read_parquet(config.META_DIR / "llm_labels_gold.parquet"), gold)

    candidates = sorted(SWEEP_DIR.glob("llm_probs_gold__*.parquet"))
    shipped = config.META_DIR / "llm_probs_gold.parquet"
    if not candidates:
        raise SystemExit(
            f"no dumps in {SWEEP_DIR}. Run notebook 03 to the gold sweep on "
            "Colab and drop the llm_probs_gold__*.parquet files there first.")

    rows = {}
    for path in [shipped] + candidates:
        name = ("shipped" if path == shipped
                else path.stem.replace("llm_probs_gold__", ""))
        prob_gold = eval_fusion._align(pd.read_parquet(path), gold)
        raw = eval_fusion._score_frame_aucs(
            gold, eval_fusion.probability_scores(prob_gold))
        rows[name] = {"raw": raw, "fused": fused_aucs(gold, cat_gold, prob_gold)}

    reference = float(np.nanmean(rows["shipped"]["fused"].to_numpy()))

    names = list(rows)
    print(f"{'finding':18s} " + "  ".join(f"{n[:12]:>12s}" for n in names))
    print("-" * (20 + 14 * len(names)))
    for finding in config.LABELS:
        print(f"{finding:18s} " + "  ".join(
            f"{rows[n]['fused'][finding]:12.3f}" for n in names))
    print("-" * (20 + 14 * len(names)))
    print(f"{'fused macro':18s} " + "  ".join(
        f"{np.nanmean(rows[n]['fused'].to_numpy()):12.3f}" for n in names))
    print(f"{'raw macro':18s} " + "  ".join(
        f"{np.nanmean(rows[n]['raw'].to_numpy()):12.3f}" for n in names))

    print(f"\ngate: fused macro > {reference:.3f} + {GATE_MARGIN} "
          f"and every finding >= {GATE_FLOOR}")
    for name in names:
        if name == "shipped":
            continue
        fused = rows[name]["fused"]
        macro = float(np.nanmean(fused.to_numpy()))
        weakest = fused.idxmin()
        clears_macro = macro > reference + GATE_MARGIN
        clears_floor = float(fused.min()) >= GATE_FLOOR
        verdict = "PASSES" if clears_macro and clears_floor else "fails"
        detail = []
        if not clears_macro:
            detail.append(f"macro {macro:.3f} vs {reference + GATE_MARGIN:.3f}")
        if not clears_floor:
            detail.append(f"{weakest} {fused.min():.3f} < {GATE_FLOOR}")
        print(f"  {name:24s} {verdict}"
              + (f"  ({'; '.join(detail)})" if detail else f"  macro {macro:.3f}"))

    print("\nA passing variant earns the corpus run (notebook 03 section 7, "
          "set PROMPT_VARIANT/N_SAMPLES_CORPUS); nothing passing means stop "
          "and record it.")


if __name__ == "__main__":
    main()
