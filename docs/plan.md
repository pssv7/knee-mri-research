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

**Reopened on a method and closed on a measurement (2026-08-26).** Du et al.
(arXiv:1812.02224) argue the constant λ *is* the bug, so `aux_mode="cosine"`
applies the auxiliary gradient only while its cosine with the main gradient
is non-negative. Run on fold 1 at 16 × 5 with `LAMBDA_REPORT = 1.0`:
**val 0.874, gold 0.856** — third configuration, third failure against the
reference's 0.861. The paper's guarantee held (no collapse) and delivered
exactly what it promises (no gain).

**The finding worth keeping is about validation, not about λ.** Inside that
run val rose from 0.872 to 0.874 while gold fell from 0.870 to 0.856, and
selection is on val — correctly, since gold selection is a fixed leak. So a
report-derived auxiliary loss makes *correct* checkpoint selection choose a
worse model, because the selection metric is mined from the same reports the
loss trains on. That objection applies to the whole family regardless of
weighting, and is why this stays closed.

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
| `convnext_fold1.pt` (older) | — | none | export of `lit_fold1_oldd` (220/220 tensors identical) | 0.891 |
| `convnext_fold1.pt` (2026-08-26) | 0.874 | report 1.0 cosine | fold 1, 16 × 5 | not submitted, gold 0.856 |
| `lit_fold3.ckpt` | 0.8686 | none | fold 3, 16 × 5 | ensemble member, verified clean |

**Two different models are now called `convnext_fold1.pt`** — the 24-slice
export (LB 0.891) and the cosine-aux run of 2026-08-26 (gold 0.856, never
submitted). Neither is the ensemble's fold-1 member, which is the 16 × 5
reference and exists only as `lit_fold1_old.ckpt`. Notebook 10 globs that
filename and would average whichever copy is present without complaint, so
check the file, not the name.

Two consequences for the fold ensemble: **no `.pt` export of the 16 × 5
reference exists**, so fold 1's 16 × 5 contribution must come from
`lit_fold1_old.ckpt` (exported, or attached as the `.ckpt` — notebook 10
reads both); and **`lit_fold0.ckpt` is 16 × 3**, predating the five-series
change, so fold 0 is a retrain like 2 and 4 rather than a checkpoint
already in hand. Fold 3 is done and clean.

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

**The active plan, redrafted 2026-08-26 after C2 closed and fold 3 banked.**
Every text-side lever is now measured shut: prompts, samples, the 32B,
ensembling, all three aux configurations. What remains open is one
mechanical multiplier (the fold ensemble) and one untried supervision
channel (Phase E below). The redraft interleaves them so the ensemble runs
*produce* Phase E's inputs.

**Why fold 3 and the cosine run do not change the direction.** Fold 3's
val 0.869 is not a miss — it has no baseline to miss against (worklog (l)
records the misread and the retraction); it is a verified-clean ensemble
member and the ensemble is judged only by its five-fold LB. The cosine run
is a genuine close, but it closes a *family* that was already two-for-two
against the LB; nothing downstream was gated on it passing.

**The budget assumption that shaped the old ordering was wrong.** With the
final submission on 2026-10-22 there are ~8 weeks left, i.e. **~200 GPU-h**
at 25 h/week, against a plan that spends 15.5 h. Training hours are not the
scarce resource; the scarce resources are the **9 h inference cap**, the
**two final submission slots**, and ideas that could actually work. The
three-part cost argument that demoted ConvNeXt-S was written when the fold
ensemble looked like the whole budget, and points 2 and 3 of it (cap,
efficiency prize) still stand for anything that ships — but point 1, "we
cannot afford the run", no longer does for a *probe*.

- **Done 2026-08-26 (local): notebook 08 instrumented.** `net.build_backbone`
  now constructs the trunk for both notebooks from one name, handling the
  ViT-family `img_size` argument (notebook 10 had ConvNeXt hard-coded, so a
  backbone experiment would have trained one trunk and scored another);
  notebook 10 gained `TUNED_BACKBONE`, which travels with `TUNED_SLICES` and
  `TUNED_MAX_SERIES`. Section 9 writes `oof_fold{N}.parquet` after training,
  so every run from here produces Phase E's inputs for free. 346 tests.

- **Next Kaggle run: the DINOv2-S fine-tune probe, ahead of folds 0/2/4.**
  One run, fold 1, 16 × 5, `BACKBONE =
  "vit_small_patch14_reg4_dinov2.lvd142m"`, everything else identical,
  ~2.3 h. **It goes first because it is the only queued run that can
  invalidate the expensive one.** The ensemble's expected value is now
  ~level with 0.891 (above), so spending 6.9 h to finish it before knowing
  whether the ensemble should be ConvNeXt-only is buying an artefact that
  might be rebuilt. Cheap-before-expensive, applied to sequencing.

  **What the prior says, honestly.** Frozen DINOv2-S (reg4) *is* the 0.787
  baseline — the worst result the project has. That does not refute a
  fine-tune (fine-tuning bought ConvNeXt +0.058 over its frozen self, and
  `embed.py` has run this exact backbone at 224 for a year), but nothing
  here encourages it either: every architecture and capacity change measured
  on this project has failed — grid-4 (−0.023 gold), 24 slices (+0.002 LB,
  overfitting), ConvNeXt-S (demoted). ViTs are data-hungry and there are
  4,349 studies with mined labels.

  **The case for running it anyway is diversity, not capacity.** Fold 3
  settled that fold-diversity is worth ~nothing: two folds of one
  architecture sit 0.012 apart and average to near their mean. A trunk with
  a different inductive bias and a different pretraining objective
  (self-supervised LVD-142M vs supervised IN22k) is the kind of diversity
  that does pay in an ensemble. **Counter-evidence to hold onto:** the
  frozen+tuned blend landed *between* its members (0.839, between 0.787 and
  0.845), so diversity pays only if the new member is comparable in
  strength, never if it is weaker.

  **Gate, on fold 1's gold 0.861** (fold 1 is mandatory — it is the only
  split with a baseline, and fold 3 just demonstrated why):
  - **≥ 0.861** — the axis is open. Rebuild the ensemble around a mixed
    ConvNeXt/DINOv2 average and re-cost the cap before committing folds.
  - **0.851–0.861** — not a replacement, but a candidate diversity member.
    Worth two DINOv2 folds in a mixed ensemble, decided on one LB entry.
  - **< 0.85** — closed like grid-4. Back to ConvNeXt folds 0/2/4, and the
    backbone question is settled by measurement rather than re-argued.

  **Read the projected-runtime line in the same run.** DINOv2-S/14 at 224 is
  ~261 tokens against ConvNeXt-T's convolutions; if inference is ~1.3× then
  five DINOv2 folds do not fit under the 9 h cap and a mixed ensemble is the
  only shape that can ship. Measure it, do not estimate it.

- **Then folds 0/2/4** (~6.9 h), composition decided by the probe.
- **Kaggle → Phase D fold ensemble, 2 of 5 in hand.** Fold 3 done
  2026-08-26 (verified aux-free from the checkpoint's own hparams); fold 1
  is `lit_fold1_old.ckpt` (needs a `.pt` export; the same short session can
  backfill OOF dumps for folds 1 and 3 from their checkpoints). Notebook 08
  ships `FOLD = 0`; then 2, then 4. ~2.3 h each (~6.9 h), then the
  five-fold submission (~8.6 h, inside the 9 h cap) — ~15.5 h of the
  25 h/week quota.
  **Expected result: ~0.888–0.893, i.e. roughly level with the current
  0.891** — corrected 2026-08-26 (m) after fold 3 scored 0.877 solo against
  fold 1's 0.889. Ensembling lifts above the *mean* member (~0.883 across
  the two folds measured), not above the best one; the old "+0.005–0.010 on
  0.889" was anchored to the best member. **Record ~0.890 as success, not
  failure**, and do not reopen a closed lever on the strength of it.
  **The justification is the private LB, not the public one.** Two
  equivalent models landing 0.012 apart on the public set means keeping
  0.891 as "best" is partly a bet on which fold drew the friendlier split —
  selection on the one set that does not decide the competition. The
  five-fold average is variance insurance for the private split, and that
  is what the ~15.5 h buys.
  **Never read an ensemble fold's gold against fold 1's references** — a
  different split has no baseline, the notebook warns when `FOLD != 1`, and
  fold 3 confirmed it on the LB (higher gold, 0.012 lower LB).
  Judge members by the ensemble's leaderboard score.
  **No more solo-fold submissions.** Fold 3's cost the entry that bought the
  cross-split measurement; a second one would buy a duplicate of it.
  Before the submission: `knee-convnxt` must hold exactly the five
  `convnext_fold*.pt` exports — the 0.880 report-loss ckpt leaves, the
  cosine-aux file wearing the name `convnext_fold1.pt` never ships, and no
  fold may be present as both `.pt` and `.ckpt`.
- **Local → Phase E gate** (below) as soon as five OOF dumps exist. Costs
  seconds of local compute; only a pass commits GPU hours.

## Phase E — self-distillation: the one supervision channel not yet tried

**The structural case.** The model's LB (0.891) has caught up to its labels'
gold (0.891): the labels have been fully cashed. Every route to better
labels through the *text* is measured shut, and the ceiling analysis says
why: **27% of gold positives are never written down at all** (Synovitis:
63%). No extractor, at any size or prompt, can label what the report does
not contain — but the images contain it, and the model reads images. The
five folds' out-of-fold predictions are 4,349 studies labelled by a reader
that (a) saw the knee, not the report, and (b) never trained on the study
it is scoring. That is the only label source in the project that can rank
studies inside the silent blocks, which is precisely where the remaining
AUC lives (the tie problem: 369 of 696 gold slots tied; silence
down-weighted to 0.25 since day one).

**The mechanism, kept to the project's rules.** One global blend:
`targets' = (1 − w) · targets + w · OOF`, one parameter, chosen from a
plateau, nothing per-finding. Weights untouched (one change at a time —
the 2026-08-25 audit is the standing warning about moving both). Scored
through the full label-evaluation path against gold, exactly like a fusion
strategy.

**Gates, in cost order:**
1. **Local, seconds:** blended-label gold macro must clear the shipped
   0.891 by **≥ 0.01** (the post-replication bar, not the old 0.005) on a
   plateau in `w`, with no finding falling below 0.75. Read the gold-58
   OOF predictions as monitoring only — they never pick `w`.
2. **One retrain, ~2.3 h:** fold 1 at 16 × 5 on the blended labels, gold
   read against 0.861. This is a label-shaped change, the shape gold has
   historically forecast well (unlike the aux losses).
3. **The LB, one entry:** and read even a gold pass against the standing
   history — Phase A's +0.037 gold bought −0.002 LB. If the LB refuses a
   second-generation gain, Phase E closes in one submission and the
   five-fold ensemble remains the final-entry config.

**What would kill it early, cheaply:** if the OOF matrix's own gold macro
lands far below the labels' 0.891 (the fold-1 model scored 0.861 alone;
five averaged should land near 0.87–0.89), the blend can only dilute — stop
at gate 1. If it passes, the val-circularity objection from C2 does *not*
apply: OOF predictions come from images, not from the reports the val
labels were mined from.

**Known risk, recorded now:** self-distillation amplifies the teacher's
systematic errors — the 97 hedged false positives are in the OOF teacher's
training diet too. The gates above are the containment; no second
distillation round without an LB confirmation of the first.

**Parked behind Phase E, in order, only if budget survives:**
- ConvNeXt-S + `drop_path_rate` on fold 1 at 16 × 5, one run, gated on
  gold vs 0.861 (already sanctioned under Phase B's closing notes).
- Mirror TTA at inference: ~2× tuned inference cost, so it fits the AUC
  entry only at ≤ 2 folds' worth of slack — cost it against the 9 h cap
  before building it.
- `torch.compile` for the efficiency entry (screened 2026-08-25).

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
  only `FOLD`, **~9 GPU-h** for the four outstanding folds, historically
  +0.005–0.010.
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

Kaggle: 25 GPU-h/week, one session at a time. A fold is **~2.3 h** at 16 × 5, a
submission run ~2 h. Colab Pro (A100/L4) is reserved for extraction; a gold
checkpoint pass costs minutes, a corpus pass ~1 h. Report embeddings build
locally on CPU in minutes.
