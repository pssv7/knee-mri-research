"""Combine the rule-based and LLM extractors into soft training targets.

Produces one row per study with, for each finding, a target in [0, 1] and a
sample weight. Neither is a hard label: the extractors disagree often enough,
and are wrong often enough, that training on thresholded output would teach the
model to reproduce extraction errors.

**Nothing here is fitted on the 58 gold studies.** An earlier version derived
per-finding blend weights from gold AUCs and per-finding silence priors from
gold positive rates -- 24 numbers estimated from 58 samples. It scored macro AUC
0.860 in-sample and 0.812 under leave-one-out, which is worse than the LLM
extractor alone at 0.838. The apparent gain was the fit, not the fusion.

What replaced it is deliberately parameter-free:

**A fixed blend.** Targets are 0.10 x lexicon + 0.90 x LLM for every finding.
The ratio sits on a plateau, not an argmax: with the dual-mode LLM scores,
0.00 through 0.15 all land within 0.002 on gold (0.889-0.891) and the weight
falls away steadily above that -- the 0.30 chosen for the categorical-only
LLM cost ~0.010 against the stronger extractor. A small nonzero weight keeps
the lexicon's independent evidence without letting it drag the ranking.

**Silence scores zero, and is handled by the weight instead.** Measuring
P(positive | both silent) on gold and using it as the target was worse than
using zero, on every variant tested. The concern that motivated it -- that a
silent finding is not a confident negative -- is real, and is addressed by
giving those studies a low sample weight rather than a fabricated target.

**Disagreement lowers the weight rather than picking a winner.** A study where
the extractors conflict is genuinely ambiguous and should not pull as hard as
one where they concur.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src import config
from src.labels import calibrate, lexicon

# Blend ratio. Chosen from a plateau, not an argmax -- see module docstring.
LEXICON_WEIGHT = 0.10
LLM_WEIGHT = 1.0 - LEXICON_WEIGHT

# Sample weights by how much the two extractors corroborate each other.
WEIGHT_AGREE = 1.0        # both informative, same call
WEIGHT_DISAGREE = 0.4     # both informative, opposite calls
WEIGHT_SINGLE = 0.7       # one informative, one silent
WEIGHT_SILENT = 0.25      # neither informative; target is zero but barely counts


def _informative(present) -> bool:
    """False when an extractor said nothing about this finding."""
    if present is None:
        return False
    if isinstance(present, float) and np.isnan(present):
        return False
    return True


def _score(present, severity) -> float:
    """One extractor's contribution to the target, in [0, 1]."""
    if not _informative(present):
        return 0.0
    if int(present) == 1:
        return float(severity) if severity == severity and severity is not None else 1.0
    return 0.0


def lexicon_frame(reports, ids) -> pd.DataFrame:
    """Run the rule-based extractor over a set of reports."""
    records = []
    for report in reports:
        extracted = lexicon.extract_report(report)
        row = {}
        for name in config.LABELS:
            row[name] = extracted[name]["present"]
            row[f"{name}_severity"] = extracted[name]["severity"]
        records.append(row)

    frame = pd.DataFrame(records)
    frame.insert(0, config.ID_COL, list(ids))
    return frame


def fuse(lex: pd.DataFrame, llm: pd.DataFrame) -> pd.DataFrame:
    """Blend two extractor frames into soft targets and sample weights."""
    merged = lex.merge(llm, on=config.ID_COL, suffixes=("_lex", "_llm"))
    if len(merged) != len(lex):
        raise ValueError(f"merge lost rows: {len(lex)} -> {len(merged)}")

    output = {config.ID_COL: merged[config.ID_COL].to_numpy()}

    for name in config.LABELS:
        lex_present = merged[f"{name}_lex"].to_numpy(dtype=object)
        llm_present = merged[f"{name}_llm"].to_numpy(dtype=object)
        lex_sev = merged[f"{name}_severity_lex"].to_numpy()
        llm_sev = merged[f"{name}_severity_llm"].to_numpy()

        targets = np.empty(len(merged))
        sample_weights = np.empty(len(merged))

        for i in range(len(merged)):
            lex_score = _score(lex_present[i], lex_sev[i])
            llm_score = _score(llm_present[i], llm_sev[i])
            targets[i] = LEXICON_WEIGHT * lex_score + LLM_WEIGHT * llm_score

            lex_ok = _informative(lex_present[i])
            llm_ok = _informative(llm_present[i])
            if lex_ok and llm_ok:
                agree = (lex_score > 0) == (llm_score > 0)
                sample_weights[i] = WEIGHT_AGREE if agree else WEIGHT_DISAGREE
            elif lex_ok or llm_ok:
                sample_weights[i] = WEIGHT_SINGLE
            else:
                sample_weights[i] = WEIGHT_SILENT

        output[name] = np.clip(targets, 0.0, 1.0)
        output[f"{name}_weight"] = sample_weights

    return pd.DataFrame(output)


def fuse_scores(lex: pd.DataFrame, llm_scores: pd.DataFrame) -> pd.DataFrame:
    """Blend the lexicon frame with an already-combined LLM score frame.

    `llm_scores` is the output of `mode_fusion.combine` (or
    `mode_fusion.from_categorical`): per finding a score in [0, 1], an
    `_informative` flag, and a `_call` boolean. The weight semantics mirror
    `fuse`: agreement compares the lexicon's call with the LLM's call.
    """
    merged = lex.merge(llm_scores, on=config.ID_COL, suffixes=("_lex", ""))
    if len(merged) != len(lex):
        raise ValueError(f"merge lost rows: {len(lex)} -> {len(merged)}")

    output = {config.ID_COL: merged[config.ID_COL].to_numpy()}

    for name in config.LABELS:
        lex_present = merged[f"{name}_lex"].to_numpy(dtype=object)
        lex_sev = merged[f"{name}_severity"].to_numpy()
        llm_score = merged[name].to_numpy(dtype=float)
        llm_ok = merged[f"{name}_informative"].to_numpy(dtype=bool)
        llm_call = merged[f"{name}_call"].to_numpy(dtype=bool)

        targets = np.empty(len(merged))
        sample_weights = np.empty(len(merged))

        for i in range(len(merged)):
            lex_score = _score(lex_present[i], lex_sev[i])
            targets[i] = LEXICON_WEIGHT * lex_score + LLM_WEIGHT * llm_score[i]

            lex_ok = _informative(lex_present[i])
            if lex_ok and llm_ok[i]:
                agree = (lex_score > 0) == llm_call[i]
                sample_weights[i] = WEIGHT_AGREE if agree else WEIGHT_DISAGREE
            elif lex_ok or llm_ok[i]:
                sample_weights[i] = WEIGHT_SINGLE
            else:
                sample_weights[i] = WEIGHT_SILENT

        output[name] = np.clip(targets, 0.0, 1.0)
        output[f"{name}_weight"] = sample_weights

    return pd.DataFrame(output)


def extractor_aucs(gold: pd.DataFrame, lex_gold: pd.DataFrame,
                   llm_gold: pd.DataFrame) -> pd.DataFrame:
    """Diagnostic only: how each extractor scores on gold.

    These numbers are reported, never fed back into the targets. Letting them
    set blend weights is exactly the overfit this module was rewritten to avoid.
    """
    rows = []
    for name in config.LABELS:
        truth = gold[name].to_numpy().astype(int)
        lex_scores = np.array([
            _score(p, s) for p, s in
            zip(lex_gold[name], lex_gold[f"{name}_severity"])])
        llm_scores = np.array([
            _score(p, s) for p, s in
            zip(llm_gold[name], llm_gold[f"{name}_severity"])])
        rows.append({
            "finding": name,
            "lexicon_auc": calibrate._auc(truth, lex_scores),
            "llm_auc": calibrate._auc(truth, llm_scores),
        })
    return pd.DataFrame(rows).set_index("finding")


def build(train_csv=None, llm_mined=None, llm_gold=None,
          probs_mined=None, probs_gold=None,
          out_path: Optional[str] = None):
    """Full pipeline: extract, fuse, and score against gold.

    When the probability-mode parquets exist alongside the categorical ones,
    the two modes are combined by `mode_fusion` before the lexicon blend.
    Otherwise this is the original categorical-only fusion, so the pipeline
    still runs on a machine that never downloaded the probability dumps.
    """
    from src.labels import mode_fusion

    gold = calibrate.load_gold(train_csv)
    unlabelled = calibrate.load_unlabelled(train_csv)

    mined_frame = pd.read_parquet(
        llm_mined or (config.META_DIR / "llm_labels.parquet"))
    gold_frame = pd.read_parquet(
        llm_gold or (config.META_DIR / "llm_labels_gold.parquet"))

    # Align to CSV order so positional comparisons stay valid.
    gold_frame = (gold_frame.set_index(config.ID_COL)
                  .loc[gold[config.ID_COL]].reset_index())
    mined_frame = (mined_frame.set_index(config.ID_COL)
                   .loc[unlabelled[config.ID_COL]].reset_index())

    probs_mined_path = probs_mined or (config.META_DIR / "llm_probs.parquet")
    probs_gold_path = probs_gold or (config.META_DIR / "llm_probs_gold.parquet")
    use_probs = (Path(probs_mined_path).exists()
                 and Path(probs_gold_path).exists())

    if use_probs:
        probs_mined_frame = pd.read_parquet(probs_mined_path)
        probs_gold_frame = pd.read_parquet(probs_gold_path)
        scores_mined = mode_fusion.combine(mined_frame, probs_mined_frame)
        scores_gold = mode_fusion.combine(gold_frame, probs_gold_frame)
    else:
        scores_mined = mode_fusion.from_categorical(mined_frame)
        scores_gold = mode_fusion.from_categorical(gold_frame)

    lex_gold = lexicon_frame(gold["Report"], gold[config.ID_COL])
    lex_mined = lexicon_frame(unlabelled["Report"], unlabelled[config.ID_COL])

    fused = fuse_scores(lex_mined, scores_mined)
    fused_gold = fuse_scores(lex_gold, scores_gold)
    aucs = extractor_aucs(gold, lex_gold, gold_frame)

    if out_path:
        fused.to_parquet(out_path, index=False)

    return fused, fused_gold, aucs


def report(gold: pd.DataFrame, fused_gold: pd.DataFrame,
           aucs: pd.DataFrame, fused: pd.DataFrame) -> str:
    """Per-finding summary of the fused targets.

    The fused AUC column is an honest estimate: no parameter in this module was
    estimated from the gold studies, so in-sample and held-out are the same.
    """
    lines = [
        f"{'finding':18s} {'lexAUC':>7s} {'llmAUC':>7s} {'fusedAUC':>9s} "
        f"{'delta':>7s} {'mean w':>7s} {'silent':>7s}",
        "-" * 72,
    ]
    fused_aucs = []
    for name in config.LABELS:
        truth = gold[name].to_numpy().astype(int)
        auc = calibrate._auc(truth, fused_gold[name].to_numpy())
        fused_aucs.append(auc)
        best_single = max(aucs.loc[name, "lexicon_auc"], aucs.loc[name, "llm_auc"])
        silent = float((fused[f"{name}_weight"] == WEIGHT_SILENT).mean())
        lines.append(
            f"{name:18s} {aucs.loc[name, 'lexicon_auc']:7.3f} "
            f"{aucs.loc[name, 'llm_auc']:7.3f} {auc:9.3f} "
            f"{auc - best_single:+7.3f} "
            f"{fused[f'{name}_weight'].mean():7.2f} {silent:6.0%}"
        )
    lines.append("-" * 72)
    lines.append(
        f"macro: fused {np.nanmean(fused_aucs):.3f}   "
        f"lexicon {aucs['lexicon_auc'].mean():.3f}   "
        f"LLM {aucs['llm_auc'].mean():.3f}"
    )
    return "\n".join(lines)


def main():
    out = config.META_DIR / "train_labels.parquet"
    fused, fused_gold, aucs = build(out_path=out)
    gold = calibrate.load_gold()
    print(report(gold, fused_gold, aucs, fused))
    print(f"\nwrote {out}  {fused.shape}")


if __name__ == "__main__":
    main()
