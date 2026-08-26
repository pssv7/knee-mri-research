# Plan — current cycle and the road to 0.94

State document: `PROJECT.md` (read it first; it holds what exists and what
failed). This file holds only what happens next and the gates between steps.
History and per-run numbers live in `docs/worklog.md`.

Position as of 2026-08-26. **Best public LB 0.891** (ConvNeXt-T, 24 slices,
5 series). Cluster: 0.941 at 20th, 0.952 at top.

| configuration | honest gold | LB |
|---|---|---|
| 8 slices, categorical labels | 0.825* | 0.845 |
| 8 slices, dual-mode soft | 0.816 | 0.827 |
| 8 slices, dual-mode sharpened | 0.827 | 0.843 |
| 16 slices, 3 series | 0.840 | 0.869 |
| **16 slices, 5 series** (fold 1) | **0.861** | **0.889** ← reference |
| 24 slices, 5 series (fold 1) | 0.865 | 0.891 |
| 16 slices, 5 series + report loss 0.5 | 0.867 | 0.880 |

\* inflated by the gold-selection leak, since fixed.

**Gold forecast the LB three times running, then missed** (+0.023→+0.024,
+0.021→+0.020, +0.004→+0.002, then +0.006→**−0.009** on the report-loss
run). The rider from the miss: gold forecasts the LB when the change is
label- or capacity-shaped; an auxiliary loss aimed at the same reports the
val labels were mined from can move gold and val without moving the LB.
The gold 58's own noise floor is ~±0.02 per finding.

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
- **ConvNeXt-S has never been run** — Tiny is the incumbent from the first
  fine-tuning run, not the winner of a comparison. It stays demoted on three
  measured grounds, recorded 2026-08-26 so this is not re-litigated:
  1. *The last capacity rung failed the same way it would.* 24 slices added
     capacity against the same 4,349 studies and overfit — val peaked at
     epoch 7, gold fell 0.865 → 0.842 — for +0.002 LB. Small adds 78% more
     parameters (27.8M → 49.5M) in that same direction, and model gold
     (0.865) sits *below* label gold (0.891): the model is label-limited,
     not capacity-starved.
  2. *It collides with the 9 h submission cap.* Measured ~1.7× forward cost.
     At 16 × 5 a fold costs 1.65 h of inference, so five folds fit in 8.5 h;
     at Small's cost that becomes ~2.8 h/fold and the cap allows **three
     folds, not five**. Choosing Small spends two folds of ensembling
     (historically +0.005–0.010 across five) on an untested backbone.
  3. *The Efficiency Prize charges for it twice* — an extra hour of runtime
     costs 0.111 and needs +0.050 AUC to break even; nothing measured here
     has ever cleared that.

  The one shape worth testing, if budget survives the fold ensemble: Small
  on fold 1 at 16 × 5 **with explicit regularisation** (`drop_path_rate`,
  which timm exposes — not the report loss, which the LB rejected), one run,
  gated on gold against 0.861. Resolution (224 → 384) is a separate lever and
  means rebuilding the 29 GB cache; last resort.
- **Five folds multiply a winning config, they do not rescue one** —
  historically +0.005–0.010, spent once at the end (see Phase D for the 9 h
  ceiling on fold count).

## Phase C2 — CLOSED 2026-08-26: the LB overruled gold

Three runs settled it, the last one against the first two. The screen (both
lambdas 0.2) came in at gold 0.854; the escalation (`LAMBDA_REPORT = 0.5`,
`LAMBDA_RANK = 0.0`) reached val 0.881 / gold 0.867, the best fold-1
numbers measured at 16 slices — and then **scored LB 0.880 against the
aux-free twin's 0.889** on the identical geometry, fold, and notebook. The
val gain was partly circular (val labels are mined from the same reports
the aux target embeds) and the gold gain was inside the 58-study floor.
See worklog 2026-08-26 for the full verification that this was the model
and not the plumbing.

**Reopened 2026-08-26 on a method, not a hunch.** Du et al.
(arXiv:1812.02224) argue the constant λ *is* the bug — an auxiliary task's
usefulness changes during training, so a fixed weight keeps pushing after it
starts to hurt, and their Figure 3(c) (fixed-λ multi-task below single-task
throughout, gating recovers it) is C2's result exactly. Implemented as
`aux_mode="cosine"`: apply the auxiliary gradient only while its cosine with
the main gradient is non-negative. **Queued behind the fold ensemble as one
fold-1 run at 16 × 5 with `LAMBDA_REPORT = 1.0`** — read `report_gate`, and
gate the decision on the LB, never on val, which is circular for this loss.
Expect no-worse-than-baseline rather than a win: the paper promises blocked
negative transfer, not positive transfer.

**Both lambdas ship at 0.0 meanwhile.** The "24 slices + report loss" follow-up
was gated on the LB confirming; the gate failed and the run is cancelled.
What survives: the code, and the measured reduced-overfitting signature
(val flat through epoch 12 where the plain 24-slice run decayed) — if a
later phase reaches for capacity again, a small report lambda is a
candidate regulariser there, gated on the LB and nothing softer. Phase C
(extraction) is now the lever.

**Standing trap until the next upload:** `knee-convnxt` holds the 0.880
report-loss checkpoint and notebook 10 ships `TUNED_SLICES = 16`. Restore a
matching checkpoint + slice-count pair before any future submission.

**Checkpoint inventory, verified 2026-08-26 by opening each file** (geometry
is *not* recoverable from a checkpoint — the architecture is identical at
every slice/series count — so identification is by Lightning's stored
`best_model_score` cross-referenced to this log):

| Desktop file | val | aux | geometry | LB |
|---|---|---|---|---|
| `lit_fold0.ckpt` | 0.8605 | none | fold 0, 16 × **3** | 0.869 |
| `lit_fold1_old.ckpt` | 0.8718 | none | fold 1, 16 × 5 | 0.889 |
| `lit_fold1_oldd.ckpt` | 0.8799 | none | fold 1, **24** × 5 | **0.891** |
| `lit_fold1.ckpt` | 0.8814 | report 0.5 | fold 1, 16 × 5 | 0.880 |
| `convnext_fold1.pt` | — | none | export of `lit_fold1_oldd` (220/220 tensors identical) | 0.891 |

Two consequences for the fold ensemble: **the only `.pt` export on hand is
the 24-slice one**, so fold 1's 16 × 5 contribution must come from
`lit_fold1_old.ckpt` (exported, or attached as the `.ckpt` — notebook 10
reads both); and **`lit_fold0.ckpt` is 16 × 3**, predating the five-series
change, so fold 0 is a retrain like 2, 3 and 4 rather than a checkpoint
already in hand.

The original screen design and its decision tree follow, kept because the
branch conditions are the record of how this was called.

## Phase C2 — auxiliary objectives (the screen, as designed)

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

Candidate levers, and where each now stands:

1. ~~**Prompt iteration on the probability prompt.**~~ **CLOSED 2026-08-26,
   measured.** Three one-idea variants, all fused below the shipped 0.891
   (correlates 0.884, granular 0.888, reason_first 0.882).
2. ~~**More self-consistency samples** (3 → 5).~~ **CLOSED 2026-08-26**:
   fused 0.885, *below* 3 samples. More compute, slightly worse labels.
3. ~~**Categorical re-extraction with the 32B.**~~ **CLOSED 2026-08-26,
   measured.** Fused 0.885 against the shipped 0.891, and Synovitis fell
   below the 0.75 floor. The 32B is *worse* at categorical extraction than
   the 14B (cat-only 0.824 vs 0.838), losing MCL and Medial Meniscus — the
   ground it had to win. Confounded by a possible prompt-revision difference
   in the 14B corpus; recorded in worklog (h), and it does not change the
   verdict.
4. ~~**Prompt ensembling.**~~ **CLOSED 2026-08-26**, tested free on the
   sweep's own dumps: the three phrasings averaged fuse to 0.893 (+0.002,
   jackknife 0.76) — a real but sub-gate gain costing 3× the corpus
   extraction. Declined on the trade.

**PHASE C IS CLOSED (2026-08-26). All four levers measured, none passed.**
The finding that outlives them: a replication of the whole sweep put the
run-to-run sd of the fused gold macro at **~0.004, with identical
configurations differing by 0.005 and the entire between-config spread only
0.007** — the 58-study gold set cannot resolve a label change smaller than
about 0.01, while the gate asks for 0.005. **The instrument is less precise
than the decision.** Two consequences for anyone reopening this:

- Never compare two extractions on a single draw each. Average over draws or
  resample the extraction; the delete-8 jackknife resamples *studies* and is
  blind to this noise entirely — it gave 0.76 confidence to a winner that
  scored 0.00 on the next draw.
- A label change worth shipping must clear ~0.01 on gold, not 0.005, and
  even then the standing history says gold gains have never moved the LB.

**Costed 2026-08-26 and declined** (`tools/score_categorical_merge.py`):
merging the 32B into the 14B to repair its misses is worth **+0.000 to
+0.002**, inside the noise — the two models share **80% of their false
positives** and the 32B recovers only 7 of 32 silent positives. The one large
ceiling, removing every false positive, is +0.043 but is worth at most
**+0.008 for any single finding** and is largely irreducible: for Effusion
the identical phrase ("geringer Gelenkerguss") carries both gold answers,
and where a threshold *is* legible (MCL grade 1 vs grade 2) severity already
ranks it correctly at AUC 0.959. Do not reopen this without a larger gold
set.

**The measurement that reframed Phase C** (2026-08-26): how much fused gold
AUC the probability mode adds over categorical-only, per finding —

| leverage | findings |
|---|---|
| high (+0.04 to +0.11) | Fracture, Effusion, Synovitis, Contusion, Medial OA, Lateral OA |
| none (+0.00 to +0.01) | Medial Meniscus, ACL, PF OA, MCL, Lateral Meniscus, Baker's |

Half the label space is carried by the **14B** categorical extractor and is
untouchable by any amount of probability-prompt work — which is both why
levers 1/2/4 could not move the macro, and why lever 3 is the only remaining
one with room. **Standing rule from the same day: never rank a prompt
candidate on raw macro** — the raw screen ranked `reason_first` first and it
fused last.

### The label-premise caution, resolved into sequencing (2026-08-26)

Phase A closed with labels moving 0.854 → 0.891 on gold (+0.037) while the
LB moved −0.002, and C2 added a second case of a gold gain the LB refused
(+0.006 gold → −0.009 LB). **No label-quality gain this project has measured
has ever produced a leaderboard gain.** The resolution: lever 3's *gate* is
minutes of Colab, so it runs in parallel — only a PASS commits the ~5 h of
corpus + retrain + submission, and even a pass should be read against this
history when the LB entry is spent.

**The active plan is two tracks:**
- **Kaggle → Phase D fold ensemble, started now.** Notebook 08 ships
  `FOLD = 3` for the next session; sweep 0, 2, 4 after (fold 1 exists).
  16 × 5 reference config, lambdas 0.0, ~1.8 h per fold. Then one five-fold
  submission (~8.6 h, inside the cap), expecting +0.005–0.010 on 0.889.
  Before it: `knee-convnxt` must hold exactly the five `convnext_fold*.pt`
  exports — the 0.880 report-loss ckpt leaves, and no fold may be present
  as both `.pt` and `.ckpt`.
- ~~**Colab → notebook 03 §6b.**~~ Ran 2026-08-26: fused 0.885, failed.
  **Phase C is closed; the Colab track is done and the budget is Phase D's.**
- **Kaggle → the aux experiment** (fold 1, `AUX_MODE = "cosine"`,
  `LAMBDA_REPORT = 1.0`), queued behind the ensemble folds.

**The harness for levers 1–2 is built (2026-08-26).** Notebook 03's gold
checkpoint is now a five-configuration sweep — baseline at n=3 and n=5, then
three one-idea prompt variants at n=5 (`correlates`: co-occurrence priors
for silent findings, aimed straight at Synovitis/Lateral/PF OA; `granular`:
anti-tie pressure; `reason_first`: a short sorting step before the numbers;
all in `src/labels/llm_extract.py` under `PROBABILITY_PROMPT_VARIANTS`).
One Colab session, minutes on an A100. The dumps land in
`data/meta/gold_sweep/` and `python3 tools/score_prompt_variants.py` scores
the **full fusion** per variant against the shipped 0.891 — raw AUC is
printed as a screen but the fused number gates, because fusion is what
trains. Section 7's corpus cell takes `PROMPT_VARIANT` / `N_SAMPLES_CORPUS`
and runs only on a gate pass. Levers 3–4 stay parked until the sweep reads
out.

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

## Screened external ideas (2026-08-25)

A manufacturing defect-inspection pipeline (DDRNet/EfficientViT/EfficientAD
on Triton) was mined for transferable techniques. Verdict, so nobody mines
it twice:

- **Already in place here:** fp16 — training is `16-mixed`, and notebook 10
  infers under `torch.amp.autocast`. Nothing to adopt.
- **Not applicable:** the serving stack (Triton, TensorRT engines, Kafka,
  early-exit cascade) — Kaggle scores a notebook, internet off, and every
  study must be scored anyway.
- **Candidates, unmeasured — park behind C2/C:**
  - *OHEM / hard-example emphasis* (their OHEM 0.9 threshold, hard-loss
    quantile 0.999). Caution: our labels are LLM-extracted and noisy;
    hard-example mining preferentially trains on label errors. Only worth
    trying per-finding (e.g. upweighting Synovitis) after Phase C improves
    label quality, gated on gold like everything else.
  - *`torch.compile` for the efficiency entry* — their TensorRT theme's
    only notebook-shaped analogue. Could shave the 0.0206 h/unit inference
    cost; measure compile overhead against the 9 h cap before trusting it.

## Standing budget

Kaggle: 25 GPU-h/week, one session at a time. A fold is ~1.8 h at 16 × 5, a
submission run ~2 h. Colab Pro (A100/L4) is reserved for extraction; a gold
checkpoint pass costs minutes, a corpus pass ~1 h. Report embeddings build
locally on CPU in minutes.
