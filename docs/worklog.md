# Worklog

Chronological session record. `PROJECT.md` holds current state; this holds
what happened, with numbers. Newest entry first.

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
