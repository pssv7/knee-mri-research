"""Phase C lever 4: average several prompt phrasings and score the fusion.

The gold sweep leaves one probability frame per configuration in
`data/meta/gold_sweep/`, which is everything an ensemble needs -- so this
lever costs seconds locally instead of a Colab session:

    python3 tools/score_prompt_ensembles.py

An ensemble is the per-study mean of the member frames' probabilities, with
their confidences averaged the same way; the result is fused exactly as a
single frame is (score_average with the categorical mode, then the lexicon
blend) and scored against the shipped 0.891.

**The combinations are fixed here and defined by structure, never by score.**
Sweeping every subset of five frames and reporting the best would be the
per-finding-fitting overfit this project has already paid for twice (0.860
in-sample, 0.812 leave-one-out); 58 studies cannot support selection over
many candidates. A winner here is a hypothesis to confirm, and the delete-8
jackknife below is the confirmation.
"""

import itertools
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src import config                                    # noqa: E402
from src.labels import calibrate, eval_fusion, fuse, mode_fusion  # noqa: E402

SWEEP_DIR = config.META_DIR / "gold_sweep"
STRATEGY = "score_average"

# Fixed before looking at any ensemble score. Each is a structural question:
# does averaging the same prompt help (more samples by another route), do the
# three rewordings help each other, does a uniform sampling budget matter,
# does everything together beat any part?
ENSEMBLES = {
    "baseline_pair": ["baseline_n3", "baseline_n5"],
    "phrasings3": ["correlates_n5", "granular_n5", "reason_first_n5"],
    "all_n5": ["baseline_n5", "correlates_n5", "granular_n5",
               "reason_first_n5"],
    "all5": ["baseline_n3", "baseline_n5", "correlates_n5", "granular_n5",
             "reason_first_n5"],
}

GATE_MARGIN = 0.005
GATE_FLOOR = 0.75


def load_frames(gold: pd.DataFrame) -> dict:
    frames = {}
    for path in sorted(SWEEP_DIR.glob("llm_probs_gold__*.parquet")):
        name = path.stem.replace("llm_probs_gold__", "")
        frames[name] = eval_fusion._align(pd.read_parquet(path), gold)
    return frames


def average_frames(members: list) -> pd.DataFrame:
    """Per-study mean of probabilities and of confidences.

    NaN-aware: a member that failed to parse a finding abstains from that
    cell rather than dragging the mean toward zero.
    """
    first = members[0]
    output = {config.ID_COL: first[config.ID_COL].to_numpy()}
    for name in config.LABELS:
        output[name] = np.nanmean(
            np.vstack([f[name].to_numpy(dtype=float) for f in members]), axis=0)
        conf_col = f"{name}_confidence"
        if conf_col in first.columns:
            output[conf_col] = np.nanmean(
                np.vstack([f[conf_col].to_numpy(dtype=float)
                           for f in members]), axis=0)
    return pd.DataFrame(output)


def fused_aucs(gold, cat_gold, prob_gold, lex=None) -> pd.Series:
    # The lexicon frame depends only on the gold subset, so the jackknife
    # passes one in rather than re-extracting every report per candidate.
    if lex is None:
        lex = fuse.lexicon_frame(gold["Report"], gold[config.ID_COL])
    scores = mode_fusion.combine(cat_gold, prob_gold, STRATEGY)
    return eval_fusion._score_frame_aucs(gold, fuse.fuse_scores(lex, scores))


def jackknife(gold, cat_gold, candidates: dict, draws=200, delete=8,
              seed=0) -> pd.Series:
    """How often each candidate wins on a resample of the 58.

    A candidate that wins on the full set but not on most resamples is a
    coin-flip dressed as a result -- the same check that retired the fitted
    fusion weights.
    """
    rng = np.random.default_rng(seed)
    wins = {name: 0 for name in candidates}
    n = len(gold)
    for _ in range(draws):
        keep = np.ones(n, dtype=bool)
        keep[rng.choice(n, size=delete, replace=False)] = False
        gold_k = gold.iloc[keep].reset_index(drop=True)
        cat_k = eval_fusion._align(cat_gold, gold_k)
        lex_k = fuse.lexicon_frame(gold_k["Report"], gold_k[config.ID_COL])
        macros = {}
        for name, frame in candidates.items():
            macros[name] = float(np.nanmean(
                fused_aucs(gold_k, cat_k, eval_fusion._align(frame, gold_k),
                           lex=lex_k).to_numpy()))
        wins[max(macros, key=macros.get)] += 1
    return pd.Series({k: v / draws for k, v in wins.items()})


def main():
    gold = calibrate.load_gold()
    cat_gold = eval_fusion._align(
        pd.read_parquet(config.META_DIR / "llm_labels_gold.parquet"), gold)
    frames = load_frames(gold)
    if not frames:
        raise SystemExit(f"no dumps in {SWEEP_DIR}")

    shipped = eval_fusion._align(
        pd.read_parquet(config.META_DIR / "llm_probs_gold.parquet"), gold)
    candidates = {"shipped": shipped}
    for name, members in ENSEMBLES.items():
        missing = [m for m in members if m not in frames]
        if missing:
            print(f"skipping {name}: missing {missing}")
            continue
        candidates[name] = average_frames([frames[m] for m in members])

    scored = {name: fused_aucs(gold, cat_gold, frame)
              for name, frame in candidates.items()}
    reference = float(np.nanmean(scored["shipped"].to_numpy()))

    names = list(scored)
    print(f"{'finding':18s} " + "  ".join(f"{n[:13]:>13s}" for n in names))
    print("-" * (20 + 15 * len(names)))
    for finding in config.LABELS:
        print(f"{finding:18s} " + "  ".join(
            f"{scored[n][finding]:13.3f}" for n in names))
    print("-" * (20 + 15 * len(names)))
    print(f"{'fused macro':18s} " + "  ".join(
        f"{np.nanmean(scored[n].to_numpy()):13.3f}" for n in names))

    print(f"\ngate: fused macro > {reference:.3f} + {GATE_MARGIN} "
          f"and every finding >= {GATE_FLOOR}")
    passing = []
    for name in names:
        if name == "shipped":
            continue
        series = scored[name]
        macro = float(np.nanmean(series.to_numpy()))
        clears = macro > reference + GATE_MARGIN and float(series.min()) >= GATE_FLOOR
        if clears:
            passing.append(name)
        note = f"macro {macro:+.3f} vs shipped"
        if float(series.min()) < GATE_FLOOR:
            note += f"; {series.idxmin()} {series.min():.3f} < {GATE_FLOOR}"
        print(f"  {name:18s} {'PASSES' if clears else 'fails':7s} "
              f"({macro:.3f}, {macro - reference:+.3f})")

    print("\ndelete-8 jackknife win rate (200 draws):")
    print(jackknife(gold, cat_gold, candidates).sort_values(
        ascending=False).to_string(float_format=lambda v: f"{v:.2f}"))

    if not passing:
        print("\nNothing clears the gate. Lever 4 closes with levers 1-2; "
              "the categorical mode (still 14B) is the remaining lever.")


if __name__ == "__main__":
    main()
