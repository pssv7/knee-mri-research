# Plan — current cycle and the road to 0.94

State document: `PROJECT.md` (read it first; it holds what exists and what
failed). This file holds only what happens next and the gates between steps.

Position as of 2026-08-25. **Best public LB 0.891** (ConvNeXt-T, 24 slices,
5 series), up from 0.845 at the start of the ladder — which is now finished.

| configuration | honest gold | LB |
|---|---|---|
| 8 slices, categorical labels | 0.825* | 0.845 |
| 8 slices, dual-mode soft | 0.816 | 0.827 |
| 8 slices, dual-mode sharpened | 0.827 | 0.843 |
| 8 slices, + silence weights | 0.817 | not submitted |
| 16 slices, 3 series | 0.840 | 0.869 |
| 16 slices, 5 series (fold 1) | 0.861 | 0.889 |
| **24 slices, 5 series** (fold 1) | **0.865** | **0.891** |

\* inflated by the gold-selection leak, since fixed.

**Phase A is closed.** Four label configurations spanning 0.854 to 0.891 on
gold moved the leaderboard 0.018 between them; one capacity change moved it
0.024 on its own. Labels are not the binding constraint at this scale. The
label work still underpins the current best — the 16-slice run trains on the
dual-mode sharpened targets — but it is no longer the first lever.

---

## Phase B — capacity ladder

Active. One change at a time, each measured as a fold-0 run against the
current best before buying five folds of it. Five folds multiply a config;
they do not rescue one, so the ladder is climbed first and the ~7 GPU-h
five-fold run is spent once on whatever config stops improving.

**The per-finding evidence points at coverage, not raw capacity.** From the
sharpened run, the tuned model beats frozen DINOv2 enormously on some
findings (MCL +0.217, Fracture +0.118, Contusion +0.090) yet *loses* on
Lateral OA (0.795 vs 0.899) and PF OA (0.725 vs 0.772), and its weakest
findings are the focal ones (Lateral Meniscus 0.725, PF OA 0.725, Synovitis
0.730, Medial Meniscus 0.750). Losing to frozen natural-image features
suggests the relevant anatomy is not reaching the network rather than that
the network is too small — a meniscal tear can occupy one or two slices, and
8 of 24 evenly-spaced slices can miss it outright.

1. **`SLICES_PER_SERIES` 8 → 16 — DONE. Gold 0.840, LB 0.869 (was 0.845).**
   The largest gain on the project, and confirmation that capacity rather
   than labels was the binding constraint. `ACCUM_STEPS` is derived
   (`8 // (BATCH_STUDIES × DEVICES)`) so the effective batch stays at eight
   studies however many GPUs are visible. Notebook 08 moved to Lightning with
   `ddp_notebook` for Kaggle's T4 ×2 at the same time; notebook 10's
   `TUNED_SLICES` was raised to 16 to match, which is mandatory — the
   aggregation masks over the slice axis, so a train/infer mismatch scores
   worse in silence.
2. **`max_series` 3 → 5 — DONE. Gold 0.861, LB 0.889 (was 0.869).** `BATCH_STUDIES` drops
   to 1 so the per-step image count falls to 80 (1 x 5 x 16), under the 96 the
   16-slice run held. Ran on fold 1 rather than 0, but the LB gain (+0.020)
   tracked the gold gain (+0.021) closely enough to retire that confound.
   Runtime scaled 1.29 h → 1.95 h at submission, near the 5/3 ratio, which
   confirmed the extra series were actually being read.

3. **`SLICES_PER_SERIES` 16 → 24 — implemented, ready to run.** The whole
   cache, and the last rung on this axis without rebuilding it. It changes
   something the intermediate values could not: `stack_2p5d` gives each slice
   its neighbours *in the subsampled stack*, so at 16 of 24 the 2.5D context
   is stitched from anatomy one or two real slices away; at 24 every gap is 1
   and the context is genuinely adjacent. 120 images/step (up from 80) is the
   most this shape has asked of a T4 — if it OOMs, drop `MAX_SERIES` to 4
   rather than the slice count. Notebook 10's `TUNED_SLICES` follows to 24.
   Run on **fold 1** to keep it a single-variable change against the current
   best. Expect ~16 min/epoch, ~3.3 h. **Gate: LB must beat 0.889.**
4. Backbone ConvNeXt-T → ConvNeXt-S.
5. **Five folds** on whatever config the ladder settles at: four more sessions
   editing only `FOLD`, ~7 GPU-h, historically +0.005–0.010.
6. Resolution last — 224 → 384 requires rebuilding the 29 GB cache;
   expensive, only if everything else plateaus.

## Phase C — extraction v3 (the lever that reaches toward 0.94)

Even perfect fusion of the current extractors caps near 0.89, and the cluster
sits at 0.94+. The remaining label headroom is extraction quality itself.
All iteration happens at the gold checkpoint (58 reports, minutes) before any
corpus run — two prompt changes reasoned from domain knowledge both measured
worse, so nothing ships without a gold measurement.

Candidate levers, in order of expected value per hour:

1. **Prompt iteration on the probability prompt**, measured at the checkpoint.
   Findings to target: Synovitis (labels 0.792, weakest) and Lateral/PF OA.
2. **More self-consistency samples** (`N_SAMPLES` 3 → 5–7) — pure compute,
   no new failure modes.
3. **Categorical re-extraction with the 32B.** The categorical corpus is
   still from the 14B; the fusion currently mixes 14B-categorical with
   32B-probability. Requires restoring the categorical cells in notebook 03.
4. **Prompt ensembling** — two or three phrasings averaged, if single-prompt
   iteration plateaus.

Gate for a corpus re-run: gold macro of the fused labels must beat 0.891 by
more than ~0.005 with no finding dropping below 0.75.

## Phase C2 — auxiliary objectives (implemented 2026-08-25, ready to run)

Two training-time-only losses in notebook 08, attacking what the pipeline
structurally ignored. Both are dropped at export — the submission state dict
and notebook 10 are untouched — and both run at the reference geometry
(fold 1, 16 slices, 5 series) against gold 0.861.

1. **Report-embedding regression** (`LAMBDA_REPORT`, default 0.2). The
   pipeline distils each report to 12 numbers and discards the text; this
   keeps a multilingual sentence embedding per report
   (`src/labels/report_embed.py` → `report_embeddings.parquet`, uploaded
   with `knee-labels`) and regresses it from the pooled study embedding
   (cosine distance). Rationale: hands the network the report's full summary
   as extra supervision, and the target is far harder to memorise than 12
   soft labels — also a regulariser against the overfitting the 24-slice run
   showed. This is the CheXzero-family recipe and the best guess at what the
   0.94 cluster does.
2. **Pairwise ranking loss** (`LAMBDA_RANK`, default 0.2). The metric is
   macro AUC — a ranking statistic — while BCE optimises calibration.
   RankNet term against a cross-batch queue of detached recent predictions
   (one study per rank leaves no in-batch pairs); pairs only count when soft
   targets differ by ≥ 0.25, so extraction noise is not what gets ranked.

**Sequencing: the first run is a screen with both on.** If gold improves,
ablate one at a time (set a lambda to 0.0) to attribute it; if it lands
within noise, the two single-lambda runs are still cheap (~1.8 h each). Gate
for adopting a config: gold must beat 0.861 meaningfully (≥ ~0.01) at the
selected epoch, on the same fold-1 split. Watch `report_loss` and
`rank_loss` in the epoch table's CSV alongside the macros.

## Phase D — endgame (October)

- Deadlines: entry/team merger 2026-10-15, final submission 2026-10-22.
- Keep two final submissions distinct, and use the second slot for the
  Efficiency Prize rather than for a second AUC config: 16 slices scored 0.889
  at penalty 0.240, 24 slices scored 0.891 at 0.321. The tracks pull opposite
  ways and two slots mean no compromise config is needed.
- Efficiency Prize (separate $18k track). Measured, superseding the earlier
  DICOM-bound guess: three submission runtimes (16x3 1.29 h, 16x5 1.95 h,
  24x5 2.89 h) fit ~0.30 h of shared setup plus ~0.0206 h per slice-series
  unit. **The loop is bound by GPU forward passes, not per-study decode** --
  a faster reader would recover ~18 min at most. The cost to attack is slices
  x series x folds.
- **Break-even on that track: an extra hour of runtime (penalty 0.111) needs
  +0.050 AUC**, since `Benchmark - maxAUC` ~= -0.45 makes an AUC point worth
  ~2.22. No capacity change this project has made clears that bar, so the
  efficiency entry should be the leanest config that still beats the
  benchmark comfortably.
- Fold ensembling is capped by the 9 h limit, not by quota: five folds at 24
  slices is ~13.3 h (killed), at 16 slices ~8.6 h (24 min slack). Two folds at
  24 slices (~5.5 h) is the practical ceiling.

## Standing budget

Kaggle: 25 GPU-h/week, one session at a time. A fold is ~1.5 h, a submission
run ~1 h. Colab Pro (A100/L4) is reserved for extraction; a gold checkpoint
pass costs minutes, a corpus pass ~1 h.
