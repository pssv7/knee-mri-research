# Plan — current cycle and the road to 0.94

State document: `PROJECT.md` (read it first; it holds what exists and what
failed). This file holds only what happens next and the gates between steps.

Position as of 2026-08-24: training labels score 0.891 on gold (was 0.854).
Fold 0 retrained on them scored **LB 0.827 against the old fold-0's 0.845** —
the gate caught it before folds 1–4 were spent. Diagnosis: the dual-mode
targets rank better but drained the confident mass out of the distribution
(extreme targets 73% → 60% of slots; Medial Meniscus mid-zone 35% → 63%) while
mean sample weight rose 0.63 → 0.78 — more mushy targets, counted harder,
with weaker gradients. The fix is `fuse.sharpen_targets`: a strictly monotone
quantile map back onto the categorical fusion's shape, which preserves the
0.891 gold AUC exactly (verified) while restoring 75% extreme targets.

---

## Phase A — five-fold cycle (restarted on sharpened labels)

1. Upload the sharpened `train_labels.parquet` as a new `knee-labels`
   version; rerun notebook 08 with `FOLD = 0` (~1.5 h).
2. **Gate**: fold-0 submission LB must beat 0.845 (old labels) — 0.827
   (soft labels) merely beaten is not enough to justify five folds.
   - **LB > 0.845** → folds 1–4, five-fold submission.
   - **LB ≈ 0.83–0.845** → the ordering gain isn't reaching the model;
     inspect the epoch logs for a selection-signal problem (val binarised at
     0.5 against sharp targets is cleaner truth, but check where gold peaked
     vs where val selected) before touching capacity.
   - **LB < 0.83** → something else regressed; diff the two runs' epoch logs
     before any further training.
3. Folds 1–4: four sessions, editing only `FOLD`; same shards, same labels
   version. Upload all five `convnext_fold*.pt` to `knee-convnxt` (replace,
   don't accumulate — notebook 10 globs and averages everything it finds).
   Save & Run All on notebook 10, submit. Five folds historically buy
   +0.005–0.010.

## Phase B — capacity ladder (only if the model trails its labels badly)

Trigger: fold-0 LB ≤ 0.845, or the five-fold LB lands more than ~0.03 below
the labels' 0.891. One change at a time, each measured as a fold-0 run
against the Phase A fold-0 baseline before buying five folds of it:

1. `SLICES_PER_SERIES` 8 → 16 (~2x epoch time).
2. `max_series` 3 → 5 (most studies have ~5.5 series).
3. Backbone ConvNeXt-T → ConvNeXt-S.
4. Resolution last — 224 → 384 requires rebuilding the 29 GB cache; expensive,
   only if everything else plateaus.

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

## Phase D — endgame (October)

- Deadlines: entry/team merger 2026-10-15, final submission 2026-10-22.
- Keep two final submissions distinct: best-LB and best-honest-gold configs.
- Efficiency Prize (separate $18k track): current tuned-only run projects
  ~2.2 h → penalty 0.240. If pursued, the cost to attack is per-study DICOM
  decode; a cached/parallel reader could halve it. Only worth effort after
  the main-track score settles.

## Standing budget

Kaggle: 25 GPU-h/week, one session at a time. A fold is ~1.5 h, a submission
run ~1 h. Colab Pro (A100/L4) is reserved for extraction; a gold checkpoint
pass costs minutes, a corpus pass ~1 h.
