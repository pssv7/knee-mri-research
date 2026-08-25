# Worklog

Chronological session record. `PROJECT.md` holds current state; this holds
what happened, with numbers. Newest entry first.

---

## 2026-08-24 (e) — a missing input became a submission of NaN

The sixteen-slice submission failed its own validator with "missing
predictions". Cause was not the model: `knee-convnxt` was not attached, so
`find_inputs("convnext_fold*.pt")` returned nothing and every tuned
prediction was None.

- The guard asked whether *any* checkpoint had been found. With `BLEND = 1.0`
  the attached frozen weights satisfied it, the frozen path was then skipped
  by design, and `pd.DataFrame(None, columns=...)` produced an empty frame
  that the id insert padded into rows of NaN -- several cells downstream of
  the actual problem, and printed as `tuned folds : 0` where it was easy to
  scroll past.
- Guards now check the paths the blend actually uses, and a `blended` of None
  refuses before the frame is built. `check_submission.py` gained
  `CHECK_OMIT_TUNED=1`, which withholds the tuned checkpoints and asserts the
  notebook refuses; verified both ways. Four tests.
- The reason this reached a submission at all: notebook 08's section 7 scores
  gold *from the exported checkpoint* and would have caught an export
  problem, but the earlier reporting crash meant it never ran. Export
  verification is not optional before notebook 10.

---

## 2026-08-25 (b) — five series: **LB 0.889**, new best

+0.020 over 0.869, and it tracked the gold gain (+0.021) closely enough to
retire the fold-1 confound: the capacity effect was real and fold variance did
not meaningfully distort it.

| configuration | gold | LB |
|---|---|---|
| 8 slices, 3 series | 0.827 | 0.845 |
| 16 slices, 3 series | 0.840 | 0.869 |
| **16 slices, 5 series** | **0.861** | **0.889** |

Runtime scaled 1.29 h -> 1.95 h, near the 5/3 ratio of the change, which was
independent evidence at submission time that the extra series were actually
being read rather than silently dropped.

Two capacity rungs, +0.044 between them, with labels held constant throughout.

---

## 2026-08-25 — five series: gold 0.840 -> 0.861

**Second capacity rung lands about as hard as the first. Submission pending.**

    epoch    1     3     5     7     8    10    12
    val    0.769 0.831 0.865 0.870 0.871 0.872 0.870
    gold   0.740 0.823 0.850 0.856 0.864 0.861 0.858

- Selected epoch 10 on val 0.872; **gold there 0.861**, against 0.840 for the
  three-series run. Capacity has now moved gold 0.817 -> 0.840 -> 0.861 while
  labels held constant.
- **Confounded by fold.** This was `FOLD = 1`, not 0. Gold is measured on the
  same 58 studies regardless of fold so the comparison is meaningful, but the
  delta carries fold-to-fold variance, which has never been measured here.
  +0.021 is large enough that variance is unlikely to explain it.
- Selection cost more than usual: gold peaked at epoch 7 (0.873) while val
  selected epoch 10 (gold 0.861), a 0.012 gap against the 0.002 seen before.
  Val is flat within 0.002 from epoch 7 onward, so the choice among those
  epochs is arbitrary from val's side and gold's bounce is 58-study noise.
  Not a selection fault -- selecting on gold remains off the table.
- Val and gold now sit ~0.011 apart, against ~0.02 in the eight-slice runs.
- Converged: val plateaus from epoch 7, so twelve epochs remains right and
  more would not obviously help.
- Cost 2.2 h on two T4s at ~11 min/epoch.

---

## 2026-08-24 (f) — sixteen slices: **LB 0.869**, new best

**+0.024 over the previous best of 0.845, and the largest move the project
has made.** Gold 0.840 -> LB 0.869, an offset of +0.029 against the +0.011 to
+0.020 seen previously.

Every submission to date:

| configuration | honest gold | LB |
|---|---|---|
| frozen DINOv2-S | 0.770 | 0.787 |
| frozen + tuned, 50/50 | - | 0.839 |
| 8 slices, categorical labels | 0.825* | 0.845 |
| 8 slices, dual-mode soft | 0.816 | 0.827 |
| 8 slices, dual-mode sharpened | 0.827 | 0.843 |
| **16 slices, dual-mode sharpened** | **0.840** | **0.869** |

\* inflated by the gold-selection leak, since fixed.

The label work was not wasted -- the 16-slice run trains on the dual-mode
sharpened labels -- but on its own it moved nothing, and capacity moved
everything. Slices were the binding constraint, exactly as the elimination
argument predicted once the two delivery bugs were ruled out.

Two bugs paid for reaching this number, both now guarded: a missing input
became a submission of NaN, and the checkpoint export sat behind a reporting
cell that crashed.

---

## 2026-08-24 (d) — 16 slices: honest gold 0.817 -> 0.840

**Outcome: the capacity change is the largest single gain measured on this
project. Submission pending.**

Per-epoch, two T4s, effective batch held at eight studies:

    epoch   1     4     7     9    12
    val   0.727 0.847 0.860 0.858 0.854
    gold  0.700 0.835 0.840 0.842 0.839

- Selected epoch 7 (val peak 0.860), gold there **0.840**. Gold's own peak was
  0.842 at epoch 9, so validation-fold selection cost 0.002 -- consistent with
  every prior run and further evidence the selection signal is sound.
- Epoch 1 val 0.727 sat inside the 0.73-0.75 band predicted from the 8-slice
  runs, which was the agreed check that the DDP port had not silently broken
  something. It had not.
- Honest gold by configuration: 0.827 (8 slices, sharpened) -> 0.817 (plus
  silence weights) -> **0.840** (16 slices). The capacity hypothesis, which
  the label experiments pointed at by elimination, is confirmed on gold.
- Converged rather than truncated: val flat from epoch 7, so twelve epochs is
  right and more would not obviously help. The next capacity rung is series
  (3 of ~5.5), not epochs.
- Two bugs paid for on the way: `torch.cuda.is_available()` poisoning the
  fork, and the metrics read sitting in front of the checkpoint export.

---

## 2026-08-24 (c) — Phase B step 1: 16 slices, and Lightning for T4 x2

**Outcome: notebook 08 trains at 16 slices under Lightning DDP across both
Kaggle GPUs. Not yet run.**

- `SLICES_PER_SERIES` 8 → 16. `BATCH_STUDIES` 4 → 2 keeps the per-step image
  count at 96, and `ACCUM_STEPS` is now derived from
  `8 // (BATCH_STUDIES * DEVICES)` so the effective batch is eight studies on
  one GPU or two, which also keeps the OneCycleLR step count fixed.
- **`TUNED_SLICES` in notebook 10 raised to 16 to match.** Nothing enforces
  this: the attention pool masks over the slice axis, so a checkpoint trained
  at 16 and scored at 8 runs clean and just scores worse. Notebook 09 stays
  at 8 deliberately (its job is fitting Colab's ~5.5 GB transfer ceiling) and
  the dataset now raises when the cache supplies fewer slices than configured,
  rather than training quietly on short volumes.
- New `src/model/lightning.py` holding the dataset, LightningModule, an
  epoch reporter, and the checkpoint exporter. It lives in `src/` rather than
  a notebook cell because dataloader workers and the spawn start method
  pickle: a class defined in a cell belongs to `__main__` and does not
  survive that. (`ddp_notebook` itself forks -- corrected below.) Same rule that
  put the cache's pool worker in `src/data/cache.py`.
- Three DDP hazards handled explicitly, each of which fails silently:
  memmaps are opened lazily per process and dropped in `__getstate__` (so
  ~29 GB never gets serialised into a worker); AUC is computed over
  all-gathered predictions, since averaging per-rank AUCs is not the AUC of
  the set; and the checkpoint is exported by reading the file rather than a
  module in memory, because the process that calls `fit` is not the process
  that trained.
- **A test caught a real bug before it reached Kaggle**: the prediction buffer
  was named `self._buffers`, which collides with `nn.Module`'s reserved
  attribute and corrupts `state_dict`, `load_state_dict`, and `.to(device)`.
  Renamed. Lightning was installed locally so these tests run rather than
  skip; 16 new tests, suite at 278.
- Gold remains monitor-only: the checkpoint callback monitors `val_macro`,
  and a test asserts `monitor="gold_macro"` appears nowhere in the generator.

---

## 2026-08-24 (b) — sharpened fold-0 LB 0.843; silence weighting found retired

**Outcome: sharpening recovered +0.016 (0.827 → 0.843) but still ties the
old-label baseline 0.845. Selection exonerated. Root cause of the remaining
gap identified as a second half of the same bundled change: the probability
extractor silently retired the silence down-weighting.**

- Sharpened fold-0: gold 0.827 at the selected epoch, **LB 0.843**. Three
  fold-0 points now exist: old labels 0.845, soft dual 0.827, sharpened dual
  0.843 — labels improved 0.854 → 0.891 on gold (+0.037) and the LB moved
  −0.002. The label gain is not reaching the model.
- **Selection exonerated.** Val peaked epoch 10 (0.846, gold 0.827); gold's
  own peak was epoch 7 (0.829). Val's pick costs 0.002 against an oracle
  selecting on gold, and both curves flatten after epoch 10, so 12 epochs is
  the right length. The validation signal is working as designed.
- **Root cause found by auditing the weights, not the targets.** The
  probability mode answers for every finding, so `_informative` was true
  everywhere and `WEIGHT_SILENT` never fired: 34.5% of slots (both
  extractors silent) went from 0.25 to 0.7+, a **2.8x increase in training
  mass on the least trustworthy third of the corpus**. Mean weight 0.63 →
  0.78. This contradicted a documented design decision ("silence is weighted
  down, not imputed") and was never a deliberate change — the label swap
  moved targets *and* weights together, the bundling the playbook warns of.
- Fix: `_informative` now tracks the categorical mode alone, which is the
  only extractor with a mention signal. The probability estimate still
  orders studies inside the silent block (the tie-breaking gain is kept);
  it just no longer asserts certainty. Verified: weights are now bit-for-bit
  identical to the old pipeline (mean 0.625, 34.5% at WEIGHT_SILENT), gold
  AUC unchanged at 0.891, targets still dual-mode and 75% extreme.
- Three new tests, including the invariant that would have caught this:
  adding a probability frame must change targets and never weights. Suite
  at 262.
- **Consequence for the standing hypotheses:** the capacity hypothesis has
  *not* had a clean test — the model has never trained on the new labels
  with correct weighting. Capacity (Phase B) waits one more fold-0 run.

---

## 2026-08-24 — fold-0 LB 0.827: soft targets diagnosed, sharpening added

**Outcome: the fold-0 gate fired (LB 0.827 vs old 0.845); the regression was
traced to target-shape dilution, and `fuse.sharpen_targets` now restores the
crisp shape while keeping the 0.891 gold AUC bit-for-bit.**

- Fold-0-only submission scored **0.827** — below the old-label fold-0's
  0.845 despite labels improving 0.854 → 0.891. LB tracked honest gold in
  both cycles (0.825→0.845, 0.816→0.827), so the result is real.
- Local diff of old vs new targets found the mechanism: `score_average`
  pulled confident calls toward the middle. Extreme targets (<0.1 or >0.9)
  fell 73% → 60% of slots; Medial Meniscus mid-zone grew 35% → 63%,
  Synovitis 12% → 36%. Probability mode is never silent, so the old
  silence down-weighting vanished too: mean sample weight 0.63 → 0.78.
  Net: more mid-range targets, counted harder, with weaker BCE gradients —
  and the fine ordering that raised label AUC lives in differences too
  small to learn at n=4,349.
- Fix: `sharpen_targets` quantile-maps each finding's targets onto the
  categorical fusion's marginal, plus an epsilon rank term keeping the map
  strictly increasing. Ordering (and therefore gold AUC 0.891) is preserved
  exactly — verified per finding — while the written labels return to 75%
  extreme (old: 73%). Weights untouched (one change at a time). Four new
  tests; suite at 259.
- Capacity (Phase B) demoted as suspect: a capacity limit caps gains, it
  does not regress 0.018.
- Next: upload sharpened labels, rerun fold 0, gate at LB > 0.845.

## 2026-08-23 → 08-24 — dual-mode labels, leak fix, fold-0 retrain

**Outcome: training labels 0.854 → 0.891 on gold; fold 0 retrained honestly
at gold 0.816; fold-0-only submission run, LB score pending.**

### Label fusion (local)

- Built `src/labels/mode_fusion.py`: combines categorical and probability
  extraction modes under five strategies, each with ≤1 global parameter and
  nothing fitted per finding. Built `src/labels/eval_fusion.py`: gold scoring,
  leave-one-out for the parameterised strategy, delete-8 jackknife, plateau
  scans.
- Extended `fuse.py` with a score-frame path and a probability-aware
  `build()`; verified the no-dumps fallback reproduces the previous
  `train_labels.parquet` bit for bit. Test suite 234 → 255, all passing.
- Gold checkpoint with the 32B extractor (user, Colab): probability mode
  0.810 → **0.860** macro; MCL 0.705 → 0.839, Medial Meniscus 0.576 → 0.765.
- Strategy comparison: `score_average` 0.889, `rank_average` 0.889 (jackknife
  split 0.54/0.47 → tie → simplest wins), both above the 0.885 per-finding
  oracle. Froze `score_average`; it also keeps a probability-like scale
  (mean target 0.33 vs gold prevalence 0.35), which matters for BCE targets.
- `LEXICON_WEIGHT` 0.30 → 0.10: the old weight cost ~0.010 against the
  stronger LLM; 0.00–0.15 is a plateau (0.889–0.891).
- Full 32B corpus run (user, ~1 h): `llm_probs.parquet`, 0% missing. Rebuilt
  `train_labels.parquet`: **gold 0.891**, weakest finding Synovitis 0.792,
  all twelve above the 0.70 kill threshold, 300–975 distinct target values
  per finding (the categorical ties are gone).

### Training pipeline (generator)

- Notebook 08: **gold-selection leak fixed** — best epoch now selected on the
  validation fold's mined labels binarised at 0.5; gold prints as monitoring
  only. `FOLD` is a per-session constant so five sessions produce
  `convnext_fold{0..4}.pt`.
- Notebook 10: `BLEND = 1.0` (leaderboard's verdict: frozen 0.787, blend
  0.839, tuned 0.845); tuned path averages all fold checkpoints it finds.
- Notebook 03: extractor ladder now prefers Qwen2.5-32B-Instruct-AWQ at
  ≥22 GB VRAM, AWQ-safe dtype handling, autoawq install on the transformers
  fallback.
- Fixed `tools/check_notebook.py`, which still asserted the categorical
  schema and had been failing silently since the probability switch; it now
  drives the probability aggregation path. `check_submission.py` fakes two
  tuned folds to exercise multi-checkpoint averaging.

### Fold-0 retrain (user, Kaggle)

- First attempt caught running on **one shard** (tell: 2.3 min/epoch ≈ the
  one-shard speed; 1,490 studies, 21 gold). Stopped, attached all three
  shards, reran: 4,349 studies, 58 gold, 6.1 min/epoch.
- Result: best val macro 0.821, **gold 0.816 honest** (old run: 0.825 *with*
  the selection leak). Per finding vs frozen baseline: Fracture 0.865 vs
  0.683, Baker's 0.924, Medial OA 0.949; weak spots Synovitis 0.699 and
  Lateral OA 0.809 (both below frozen).
- Fold-0-only submission run clean: `frozen=False tuned=True weight=1.0`,
  projected 2.16 h / 1,500 studies (efficiency penalty 0.240). **Decision
  gate on the LB score**: ≥ ~0.85 → launch folds 1–4; ≤ 0.845 → 16-slice
  capacity experiment first. See `docs/plan.md`.

### Documentation

- `PROJECT.md` updated through each state change; `docs/plan.md` (forward
  plan with gates) and `docs/agent_playbook.md` (decision method for whoever
  continues) written; `*.pt` gitignored.
