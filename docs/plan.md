# Plan — current cycle and the road to 0.94

State document: `PROJECT.md` (read it first; it holds what exists and what
failed). This file holds only what happens next and the gates between steps.

Position as of 2026-08-24. **Best public LB 0.869** (ConvNeXt-T, 16 slices,
dual-mode sharpened labels), up from 0.845.

| configuration | honest gold | LB |
|---|---|---|
| 8 slices, categorical labels | 0.825* | 0.845 |
| 8 slices, dual-mode soft | 0.816 | 0.827 |
| 8 slices, dual-mode sharpened | 0.827 | 0.843 |
| 8 slices, + silence weights | 0.817 | not submitted |
| **16 slices, same labels** | **0.840** | **0.869** |

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
2. **`max_series` 3 → 5 — implemented, ready to run.** `BATCH_STUDIES` drops
   to 1 so the per-step image count falls to 80 (1 x 5 x 16), under the 96 the
   16-slice run held; `ACCUM_STEPS` derives to 4 on two GPUs, keeping the
   effective batch at eight studies. Notebook 10's `TUNED_MAX_SERIES` follows
   to 5 — the same mandatory coupling as `TUNED_SLICES`, and the same silent
   failure if it drifts. Expect ~10-11 min/epoch, ~2.2 h.
   **Gate: LB must beat 0.869.**
3. Backbone ConvNeXt-T → ConvNeXt-S.
4. **Five folds** on whatever config the ladder settles at: four more sessions
   editing only `FOLD`, ~7 GPU-h, historically +0.005–0.010.
5. Resolution last — 224 → 384 requires rebuilding the 29 GB cache;
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
- Efficiency Prize (separate $18k track): the 16-slice run projected 1.29 h →
  penalty 0.144, but that extrapolates from three public studies and is noise
  dominated. What it does suggest is that the loop is bound by per-study DICOM
  decode rather than GPU work, since doubling slices barely moved it. If pursued, the cost to attack is per-study DICOM
  decode; a cached/parallel reader could halve it. Only worth effort after
  the main-track score settles.

## Standing budget

Kaggle: 25 GPU-h/week, one session at a time. A fold is ~1.5 h, a submission
run ~1 h. Colab Pro (A100/L4) is reserved for extraction; a gold checkpoint
pass costs minutes, a corpus pass ~1 h.
