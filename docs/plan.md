# Plan — current cycle and the road to 0.94

State document: `PROJECT.md` (read it first; it holds what exists and what
failed). This file holds only what happens next and the gates between steps.
History and per-run numbers live in `docs/worklog.md`.

Position as of 2026-08-25. **Best public LB 0.891** (ConvNeXt-T, 24 slices,
5 series). Cluster: 0.941 at 20th, 0.952 at top.

| configuration | honest gold | LB |
|---|---|---|
| 8 slices, categorical labels | 0.825* | 0.845 |
| 8 slices, dual-mode soft | 0.816 | 0.827 |
| 8 slices, dual-mode sharpened | 0.827 | 0.843 |
| 16 slices, 3 series | 0.840 | 0.869 |
| **16 slices, 5 series** (fold 1) | **0.861** | **0.889** ← reference |
| 24 slices, 5 series (fold 1) | 0.865 | 0.891 |

\* inflated by the gold-selection leak, since fixed.

**Gold has forecast the LB three times running** (+0.023→+0.024,
+0.021→+0.020, +0.004→+0.002). A small gold delta is a small real delta, not
noise concealing a win. The gold 58's own noise floor is ~±0.02 per finding;
macro deltas ≥ ~0.01 have been reliable.

## Closed: Phase A (labels) and Phase B (capacity)

**Phase A closed 2026-08-24.** Four label configurations spanning 0.854 to
0.891 on gold moved the LB 0.018 in total; one capacity change moved it
0.024 alone. Label *delivery* is done; label *quality* continues as Phase C.

**Phase B closed 2026-08-25.** The ladder ran 0.845 → 0.869 (16 slices) →
0.889 (5 series) → 0.891 (24 slices). The last rung paid +0.002 for +57%
training and +48% inference cost, and its gold curve fell after epoch 7 —
overfitting, not headroom. What survives from Phase B as standing rules:

- **Experiments iterate on fold 1 at 16 slices × 5 series** against gold
  0.861 — the only split with a measured baseline. One change at a time.
- 24 slices is a one-line change held for a final AUC-slot run only.
- ConvNeXt-S is demoted, not dead: if ever run, it needs regularisation and
  the 16-slice geometry. Resolution (224 → 384) means rebuilding the 29 GB
  cache; last resort.
- **Five folds multiply a winning config, they do not rescue one** —
  historically +0.005–0.010, spent once at the end (see Phase D for the 9 h
  ceiling on fold count).

## Phase C2 — auxiliary objectives (ACTIVE — the next Kaggle run)

Two training-time-only losses in notebook 08, attacking what the pipeline
structurally ignored. Both are dropped at export — the submission state dict
and notebook 10 are untouched, and inference cost is unchanged, so a win
here is free on the efficiency track too.

1. **Report-embedding regression** (`LAMBDA_REPORT`, default 0.2). The
   pipeline distils each report to 12 numbers and discards the text; this
   keeps a multilingual sentence embedding per report
   (`src/labels/report_embed.py` → `report_embeddings.parquet`, uploaded
   with `knee-labels`) and regresses it from the pooled study embedding
   (cosine distance). Hands the network the report's full summary as extra
   supervision, and the target is far harder to memorise than 12 soft
   labels — also a regulariser against the overfitting the 24-slice run
   showed. CheXzero-family recipe; best guess at what the 0.94 cluster does.
2. **Pairwise ranking loss** (`LAMBDA_RANK`, default 0.2). The metric is
   macro AUC — a ranking statistic — while BCE optimises calibration.
   RankNet term against a cross-batch queue of detached recent predictions
   (one study per rank leaves no in-batch pairs); pairs only count when soft
   targets differ by ≥ 0.25, so extraction noise is not what gets ranked.

**The screen run:** fold 1, 16 × 5, both lambdas at 0.2, ~1.8 h. Check the
`aux losses: report 0.2 (dim 384)  rank 0.2` line and the absence of the
"not attached" notice before letting it run. Read three things from the
result: the gold macro at the selected epoch, the shape of the val/gold
curves in late epochs (does the peak hold instead of decaying?), and the
`report_loss` / `rank_loss` columns in `metrics.csv` (are they actually
falling?).

**Decision tree on the screen's gold vs the 0.861 reference:**

- **≥ 0.875 (clear win).** Submit it (~2 h, expect the LB to track the gold
  delta) and make this the new reference config. Then attribute: two
  ablation runs (`LAMBDA_RANK = 0.0`, then `LAMBDA_REPORT = 0.0`), keep
  whichever components carry the gain, and consider one lambda-tuning run
  on the winner (0.2 → 0.5) before Phase D multiplies it into folds.
- **0.87 ± (borderline, +0.005–0.015).** Do not submit yet. Ablate first —
  the likeliest explanation of a small net gain is one loss helping and the
  other hurting. Adopt and submit the best single-loss config only if its
  own run clears ~0.87.
- **≈ 0.861 (flat) but the aux losses trained** (their columns fell and the
  late-epoch curves decayed less). The supervision works but 0.2 is too
  timid or 16 slices is not where it pays: one run with `LAMBDA_REPORT =
  0.5`, and if the reduced-overfitting signature is real, one run at **24
  slices + report loss** — the ladder's last rung failed on overfitting,
  which is exactly what this loss treats. Two runs maximum; if still flat,
  close C2 and pivot to Phase C.
- **≈ 0.861 (flat) and the aux losses did not train** (report_loss stuck
  near 1.0, or rank_loss near log 2 throughout). Something is broken, not
  weak — check the two printed lines, the parquet actually attached, and
  the loss curves before spending anything more. Fix or close.
- **< 0.85 (hurts).** Close C2 for the current geometry: set both lambdas
  to 0.0 as the shipped default, record it in the worklog, pivot to Phase
  C. Keep the code — the report loss may still earn its place as a
  regulariser if a later phase reaches for more capacity.

Whatever the branch: **fold 1 only** until a config wins; folds 0/2/3/4
have no baseline and buy no information.

## Phase C — extraction v3 (parallel lever, Colab)

Even perfect fusion of the current extractors caps near 0.89 on gold, and
the cluster sits at 0.94+. The remaining label headroom is extraction
quality. All iteration happens at the gold checkpoint (58 reports, minutes)
before any corpus run — two prompt changes reasoned from domain knowledge
both measured worse, so nothing ships without a gold measurement. This
costs Colab minutes, not Kaggle hours, so it runs in parallel with C2.

Candidate levers, in order of expected value per hour:

1. **Prompt iteration on the probability prompt**, measured at the
   checkpoint. Findings to target: Synovitis (labels 0.792, weakest) and
   Lateral/PF OA.
2. **More self-consistency samples** (`N_SAMPLES` 3 → 5–7) — pure compute,
   no new failure modes.
3. **Categorical re-extraction with the 32B.** The categorical corpus is
   still from the 14B; the fusion currently mixes 14B-categorical with
   32B-probability. Requires restoring the categorical cells in notebook 03.
4. **Prompt ensembling** — two or three phrasings averaged, if single-prompt
   iteration plateaus.

Gate for a corpus re-run: gold macro of the fused labels must beat 0.891 by
more than ~0.005 with no finding dropping below 0.75. If new labels ship,
the C2-winning config retrains on them (fold 1 first, same gate discipline).

## Phase D — endgame (October)

- Deadlines: entry/team merger 2026-10-15, final submission 2026-10-22.
  Leave the final week for submission dry-runs.
- **Five folds** on whatever config C2/C settle: four more sessions editing
  only `FOLD`, ~7 GPU-h, historically +0.005–0.010.
- Keep the two final submissions distinct: one AUC entry, one lean
  Efficiency entry. The tracks pull opposite ways and two slots mean no
  compromise config is needed.
- Efficiency Prize (separate $18k track), measured: submission runtime fits
  ~0.30 h shared setup + ~0.0206 h per slice × series unit, GPU-bound.
  **Break-even: an extra hour of runtime (penalty 0.111) needs +0.050
  AUC** — nothing this project has measured clears that, so the efficiency
  entry is the leanest config that comfortably beats the benchmark
  (16 × 5 at penalty 0.240 is the current candidate; C2's losses cost it
  nothing).
- Fold ensembling is capped by the 9 h limit, not quota: five folds fit at
  16 slices (~8.6 h, 24 min slack); at 24 slices the ceiling is two folds
  (~5.5 h).

## Standing budget

Kaggle: 25 GPU-h/week, one session at a time. A fold is ~1.8 h at 16 × 5, a
submission run ~2 h. Colab Pro (A100/L4) is reserved for extraction; a gold
checkpoint pass costs minutes, a corpus pass ~1 h. Report embeddings build
locally on CPU in minutes.
