# Plan — current cycle and the road to 0.94

State document: `PROJECT.md` (read it first; it holds what exists and what
failed). This file holds only what happens next and the gates between steps.

Position as of 2026-08-24. Three fold-0 submissions now exist:

| labels | gold (labels) | gold (model) | LB |
|---|---|---|---|
| categorical, leaked selection | 0.854 | 0.825 | 0.845 |
| dual-mode soft | 0.891 | 0.816 | 0.827 |
| dual-mode sharpened | 0.891 | 0.827 | 0.843 |

**Labels improved 0.037 on gold; the leaderboard moved −0.002.** Sharpening
recovered most of the soft-label regression. Selection is exonerated (val
peaked epoch 10, gold's own peak was epoch 7, a 0.002 difference). The
remaining cause was found in the *weights*, not the targets: the probability
extractor answers for every finding, so `WEIGHT_SILENT` never fired and 34.5%
of slots — the both-silent third of the corpus — carried 2.8x their intended
training mass. Fixed; weights are now bit-for-bit identical to the old
pipeline while targets keep the dual-mode ordering and sharpened shape.

---

## Phase A — one clean fold-0 (the capacity hypothesis needs a fair test)

The model has never trained on the new labels with correct weighting, so
Phase B's trigger cannot be evaluated yet. This run is a single-variable
change from the 0.843 run: **only the sample weights differ.**

1. Upload the rebuilt `train_labels.parquet` as a new `knee-labels` version;
   rerun notebook 08 with `FOLD = 0` (~1.5 h); submit fold-0 alone.
2. **Gate**:
   - **LB > 0.845** → the label gain is finally reaching the model. Run
     folds 1–4 and submit the five-fold ensemble (+0.005–0.010 historically).
   - **LB ≈ 0.843–0.845** → labels are not the binding constraint at this
     capacity. The capacity hypothesis is then cleanly convicted → Phase B.
   - **LB < 0.83** → the weight change hurt; that would mean silence carries
     more signal than assumed, which is itself informative. Revert and go to
     Phase B.
3. Do **not** run folds 1–4 on a config that merely ties 0.845. Five folds
   multiply a config; they do not rescue one. Bank the current checkpoint as
   the fallback and spend quota on finding a better config first.

## Phase B — capacity ladder

Trigger: the Phase A clean fold-0 fails to beat 0.845. One change at a time,
each measured as a fold-0 run against the 0.843 baseline before buying five
folds of it.

**The per-finding evidence points at coverage, not raw capacity.** From the
sharpened run, the tuned model beats frozen DINOv2 enormously on some
findings (MCL +0.217, Fracture +0.118, Contusion +0.090) yet *loses* on
Lateral OA (0.795 vs 0.899) and PF OA (0.725 vs 0.772), and its weakest
findings are the focal ones (Lateral Meniscus 0.725, PF OA 0.725, Synovitis
0.730, Medial Meniscus 0.750). Losing to frozen natural-image features
suggests the relevant anatomy is not reaching the network rather than that
the network is too small — a meniscal tear can occupy one or two slices, and
8 of 24 evenly-spaced slices can miss it outright.

1. **`SLICES_PER_SERIES` 8 → 16 — DONE, honest gold 0.817 → 0.840.** The
   largest single gain on the project; submission pending. Everything below
   this line is now the live ladder. `BATCH_STUDIES`
   drops to 2 so the per-step image count is unchanged (2 × 3 × 16 = 96), and
   `ACCUM_STEPS` is now derived (`8 // (BATCH_STUDIES × DEVICES)`) so the
   effective batch stays at eight studies on one GPU or two. Notebook 08 also
   moved to Lightning with `ddp_notebook` to use Kaggle's T4 ×2; notebook 10's
   `TUNED_SLICES` was raised to 16 to match, which is mandatory — the
   aggregation masks over the slice axis, so a train/infer mismatch scores
   worse in silence.
2. `max_series` 3 → 5 (most studies have ~5.5). Cheaper than slices and the
   likelier fix for PF OA specifically, since the patella may simply be
   absent from the three series currently chosen.
3. Backbone ConvNeXt-T → ConvNeXt-S.
4. Resolution last — 224 → 384 requires rebuilding the 29 GB cache;
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
