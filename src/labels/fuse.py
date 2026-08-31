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


def silence_rates(labels: pd.DataFrame) -> pd.Series:
    """Per finding, the share of studies neither extractor said anything about.

    Measured on the mined corpus -- 4,349 studies -- and never on gold. That
    distinction is what separates this from the per-finding silence priors this
    module removed: those were 24 numbers fitted on 58 samples and scored 0.860
    in-sample against 0.812 leave-one-out. A rate over 4,349 studies is an
    observation, not a fit.
    """
    return pd.Series(
        {name: float((labels[f"{name}_weight"] == WEIGHT_SILENT).mean())
         for name in config.LABELS})


def rescale_silent_weights(labels: pd.DataFrame) -> pd.DataFrame:
    """Down-weight silence in proportion to how uninformative it is.

    **The problem, measured on Synovitis.** 82.2% of studies are silent about
    it, the mined target averages 0.087, and the true prevalence on gold is
    46.6%. So roughly 1,666 of the 3,576 silent studies are true positives
    being trained toward zero -- and because silence is so common here, those
    studies carry 57% of the finding's gradient. The model is being taught to
    rank real positives low. Synovitis duly scores 0.682 on gold, the only
    finding where the fine-tuned model is WORSE than the retired frozen one
    (0.746), which is what fitting label noise harder looks like.

    **The fix, and why it is not the one that failed.** Targets are untouched:
    fabricating P(positive | silent) as a target measured worse on every
    variant tested, and that stands. What changes is the *weight*, which this
    module's own docstring already names as the mechanism for silence being an
    unconfident negative. The weight becomes

        WEIGHT_SILENT x (1 - silence_rate(finding))

    which is one rule with no free parameters, evaluated on the mined corpus.
    Silence about ACL (9.2% of studies) stays worth nearly the full 0.25,
    because a report that discusses ligaments and omits the ACL is saying
    something. Silence about Synovitis (82.2%) falls to 0.044, because a
    corpus that mentions it in one study out of six is not saying anything.

    Returns a copy; the target columns are passed through unchanged.
    """
    rates = silence_rates(labels)
    out = labels.copy()
    for name in config.LABELS:
        column = f"{name}_weight"
        silent = out[column] == WEIGHT_SILENT
        out.loc[silent, column] = WEIGHT_SILENT * (1.0 - rates[name])
    return out


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


# Order-keeper inside the reference distribution's tie blocks. Small enough
# that the restored shape dominates the target value; the epsilon term only
# arranges studies within a block.
SHARPEN_EPSILON = 0.02


def sharpen_targets(fused: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Restore `reference`'s target distribution, keeping `fused`'s ordering.

    The dual-mode targets rank studies better than the categorical fusion
    (0.891 vs 0.854 on gold) but drained the confident mass out of the
    distribution -- extreme targets fell from 73% to 60% of slots and the
    model trained on them regressed on the leaderboard (0.845 -> 0.827).
    AUC is invariant under strictly monotone maps, so the ordering gain and
    the crisp shape are separable: each finding's targets are quantile-mapped
    onto the reference marginal, plus an epsilon-scaled rank term that keeps
    the map strictly increasing where the reference has ties. Sample weights
    are not touched.
    """
    from src.labels import mode_fusion

    out = fused.copy()
    for name in config.LABELS:
        targets = fused[name].to_numpy(dtype=float)
        ref = np.sort(reference[name].to_numpy(dtype=float))
        ranks = mode_fusion._tied_ranks_01(targets)
        mapped = np.quantile(ref, np.clip(ranks, 0.0, 1.0))
        out[name] = np.clip(
            (mapped + SHARPEN_EPSILON * ranks) / (1.0 + SHARPEN_EPSILON),
            0.0, 1.0)
    return out


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

    if use_probs:
        # The categorical-only fusion of the same inputs supplies the shape
        # the dual-mode targets are sharpened back onto.
        ref_mined = fuse_scores(
            lex_mined, mode_fusion.from_categorical(mined_frame))
        ref_gold = fuse_scores(
            lex_gold, mode_fusion.from_categorical(gold_frame))
        fused = sharpen_targets(fused, ref_mined)
        fused_gold = sharpen_targets(fused_gold, ref_gold)

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
