"""Compare mode-fusion strategies against the gold 58, honestly.

Runs entirely locally in seconds once `llm_probs_gold.parquet` exists, so
strategy questions cost a keystroke instead of a Colab reload. Three views:

1. Per-finding and macro AUC of each strategy's combined scores, next to the
   categorical-only and probability-only baselines and the per-finding oracle
   (reported for reference, never selectable).
2. The downstream number: strategy -> lexicon blend via `fuse.fuse_scores` ->
   gold macro. This is the figure comparable to the 0.854 the current labels
   score.
3. Robustness. The parameter-free strategies are honest as-is. `global_blend`
   gets a leave-one-out over the 58 plus a plateau report, since its alpha is
   chosen from data. A delete-8 jackknife then checks whether the apparent
   winner survives resampling; if no strategy wins most resamples, they are
   tied and the simplest one should be frozen.
"""

import numpy as np
import pandas as pd

from src import config
from src.labels import calibrate, fuse, mode_fusion


def _score_frame_aucs(gold: pd.DataFrame, frame: pd.DataFrame) -> pd.Series:
    """Per-finding AUC of a score frame aligned to the gold order."""
    aucs = {}
    for name in config.LABELS:
        truth = gold[name].to_numpy().astype(int)
        aucs[name] = calibrate._auc(truth, frame[name].to_numpy(dtype=float))
    return pd.Series(aucs)


def _align(frame: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    return (frame.set_index(config.ID_COL)
            .loc[gold[config.ID_COL]].reset_index())


def probability_scores(prob_frame: pd.DataFrame) -> pd.DataFrame:
    """Probability-only baseline in score-frame shape (NaN scores -> 0)."""
    output = {config.ID_COL: prob_frame[config.ID_COL].to_numpy()}
    for name in config.LABELS:
        prob = prob_frame[name].to_numpy(dtype=float)
        output[name] = np.nan_to_num(prob, nan=0.0)
    return pd.DataFrame(output)


def strategy_table(gold: pd.DataFrame, cat_gold: pd.DataFrame,
                   prob_gold: pd.DataFrame) -> pd.DataFrame:
    """Per-finding AUC for both baselines, every strategy, and the oracle."""
    columns = {
        "categorical": _score_frame_aucs(
            gold, mode_fusion.from_categorical(cat_gold)),
        "probability": _score_frame_aucs(gold, probability_scores(prob_gold)),
    }
    for strategy in mode_fusion.STRATEGIES:
        combined = mode_fusion.combine(cat_gold, prob_gold, strategy)
        columns[strategy] = _score_frame_aucs(gold, combined)

    table = pd.DataFrame(columns)
    # Oracle = per-finding max of the two modes. Reported to show the ceiling;
    # selecting it per finding is the known 58-study overfit.
    table["oracle"] = table[["categorical", "probability"]].max(axis=1)
    return table


def downstream_macro(gold: pd.DataFrame, cat_gold: pd.DataFrame,
                     prob_gold: pd.DataFrame, strategy: str) -> float:
    """Gold macro after the full lexicon blend -- comparable to 0.854."""
    lex_gold = fuse.lexicon_frame(gold["Report"], gold[config.ID_COL])
    scores = mode_fusion.combine(cat_gold, prob_gold, strategy)
    fused = fuse.fuse_scores(lex_gold, scores)
    return float(np.nanmean(_score_frame_aucs(gold, fused).to_numpy()))


def downstream_macro_categorical(gold: pd.DataFrame,
                                 cat_gold: pd.DataFrame) -> float:
    lex_gold = fuse.lexicon_frame(gold["Report"], gold[config.ID_COL])
    fused = fuse.fuse_scores(lex_gold, mode_fusion.from_categorical(cat_gold))
    return float(np.nanmean(_score_frame_aucs(gold, fused).to_numpy()))


def blend_plateau(gold: pd.DataFrame, cat_gold: pd.DataFrame,
                  prob_gold: pd.DataFrame,
                  alphas=np.arange(0.0, 1.01, 0.05)) -> pd.DataFrame:
    """Macro AUC of `global_blend` across the alpha grid."""
    rows = []
    original = mode_fusion.GLOBAL_BLEND_ALPHA
    try:
        for alpha in alphas:
            mode_fusion.GLOBAL_BLEND_ALPHA = float(alpha)
            combined = mode_fusion.combine(cat_gold, prob_gold, "global_blend")
            macro = float(np.nanmean(
                _score_frame_aucs(gold, combined).to_numpy()))
            rows.append({"alpha": float(alpha), "macro": macro})
    finally:
        mode_fusion.GLOBAL_BLEND_ALPHA = original
    return pd.DataFrame(rows)


def blend_loo_macro(gold: pd.DataFrame, cat_gold: pd.DataFrame,
                    prob_gold: pd.DataFrame,
                    alphas=np.arange(0.0, 1.01, 0.05)) -> float:
    """Leave-one-out estimate for `global_blend`.

    For each held-out study, alpha is chosen on the remaining 57 and the
    held-out study is scored with that alpha's combined frame. The held-out
    scores are then pooled per finding and one AUC computed -- the same
    protocol that exposed the fitted fusion weights (0.860 in-sample vs
    0.812 LOO).
    """
    n = len(gold)
    pooled = {name: np.empty(n) for name in config.LABELS}
    original = mode_fusion.GLOBAL_BLEND_ALPHA
    try:
        for held in range(n):
            keep = np.arange(n) != held
            gold_k = gold.iloc[keep].reset_index(drop=True)
            cat_k = _align(cat_gold, gold_k)
            prob_k = _align(prob_gold, gold_k)

            best_alpha, best_macro = None, -np.inf
            for alpha in alphas:
                mode_fusion.GLOBAL_BLEND_ALPHA = float(alpha)
                combined = mode_fusion.combine(cat_k, prob_k, "global_blend")
                macro = float(np.nanmean(
                    _score_frame_aucs(gold_k, combined).to_numpy()))
                if macro > best_macro:
                    best_alpha, best_macro = float(alpha), macro

            mode_fusion.GLOBAL_BLEND_ALPHA = best_alpha
            combined = mode_fusion.combine(cat_gold, prob_gold, "global_blend")
            row = combined.set_index(config.ID_COL).loc[
                gold[config.ID_COL].iloc[held]]
            for name in config.LABELS:
                pooled[name][held] = row[name]
    finally:
        mode_fusion.GLOBAL_BLEND_ALPHA = original

    aucs = [calibrate._auc(gold[name].to_numpy().astype(int), pooled[name])
            for name in config.LABELS]
    return float(np.nanmean(aucs))


def jackknife(gold: pd.DataFrame, cat_gold: pd.DataFrame,
              prob_gold: pd.DataFrame, drop: int = 8,
              n_resamples: int = 200, seed: int = 0) -> pd.DataFrame:
    """Which strategy wins when 8 random gold studies are deleted?

    A winner that only wins on the full 58 is a coin flip wearing a crown.
    """
    rng = np.random.default_rng(seed)
    n = len(gold)
    wins = {s: 0 for s in mode_fusion.STRATEGIES}
    for _ in range(n_resamples):
        keep = np.sort(rng.choice(n, size=n - drop, replace=False))
        gold_k = gold.iloc[keep].reset_index(drop=True)
        cat_k = _align(cat_gold, gold_k)
        prob_k = _align(prob_gold, gold_k)

        best, best_macro = None, -np.inf
        for strategy in mode_fusion.STRATEGIES:
            combined = mode_fusion.combine(cat_k, prob_k, strategy)
            macro = float(np.nanmean(
                _score_frame_aucs(gold_k, combined).to_numpy()))
            if macro > best_macro:
                best, best_macro = strategy, macro
        wins[best] += 1

    return pd.DataFrame({
        "strategy": list(wins),
        "win_rate": [wins[s] / n_resamples for s in wins],
    }).set_index("strategy")


def main():
    prob_gold_path = config.META_DIR / "llm_probs_gold.parquet"
    if not prob_gold_path.exists():
        raise SystemExit(
            f"{prob_gold_path} not found. Run notebook 03 to the gold "
            "checkpoint on Colab and download the dump first."
        )

    gold = calibrate.load_gold()
    cat_gold = _align(
        pd.read_parquet(config.META_DIR / "llm_labels_gold.parquet"), gold)
    prob_gold = _align(pd.read_parquet(prob_gold_path), gold)

    table = strategy_table(gold, cat_gold, prob_gold)
    with pd.option_context("display.width", 140, "display.precision", 3):
        print(table)
    print("\nmacro:")
    for column in table.columns:
        print(f"  {column:22s} {table[column].mean():.3f}")

    print("\ndownstream (with lexicon blend, vs 0.854 baseline):")
    print(f"  {'categorical-only':22s} "
          f"{downstream_macro_categorical(gold, cat_gold):.3f}")
    for strategy in mode_fusion.STRATEGIES:
        print(f"  {strategy:22s} "
              f"{downstream_macro(gold, cat_gold, prob_gold, strategy):.3f}")

    plateau = blend_plateau(gold, cat_gold, prob_gold)
    print("\nglobal_blend alpha grid:")
    print(plateau.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"global_blend LOO macro: "
          f"{blend_loo_macro(gold, cat_gold, prob_gold):.3f}")

    print("\ndelete-8 jackknife win rates:")
    print(jackknife(gold, cat_gold, prob_gold).to_string(
        float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
