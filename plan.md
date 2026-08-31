# Synovitis Recovery Plan

**Backbone:** ConvNeXt-T · 16 slices × 5 series
**Baseline:** Synovitis AUC 0.701 (fold 1, gold n=58, 27 pos / 31 neg)
**Status:** diagnosis confirmed; intervention gated on one free check

---

## 1. What is established

Every number below was recomputed from `gold_truth_58.parquet` joined to
`gold_predictions.parquet`. All twelve per-finding AUCs reproduce the project's
own table exactly, so the analysis is on the same footing as the source report.

### 1.1 The synovitis head carries no information beyond effusion status

| Score used as a synovitis classifier | AUC |
|---|---|
| Synovitis head | 0.701 |
| **Effusion head** | **0.722** |
| Gold effusion label alone | 0.698 |

Paired bootstrap, same 58 studies:
`Synovitis head − Effusion head = −0.020, 95% CI [−0.146, +0.105]`

The head can be deleted and replaced by the effusion head with no measured loss.
Knowing only whether the knee has an effusion reproduces the model's entire
performance on this finding.

### 1.2 The model over-couples relative to truth

- `r_pred` = corr(synovitis pred, effusion pred) = **0.651**
- `r_true` = corr(synovitis truth, effusion truth) = **0.403**

Gold 2×2 (Fisher OR 6.09, p = 0.0030):

```
              Effusion=0   Effusion=1
Synovitis=0       18           13
Synovitis=1        5           22
```

The findings genuinely co-occur. The model couples them roughly 60% more
tightly than reality does.

### 1.3 The head is gated by fluid

Across the 44 studies with Effusion pred < 0.70, the Synovitis score never
exceeds 0.28. All 8 studies scoring above 0.35 have Effusion above 0.72. Only
3 of 58 have Synovitis pred > Effusion pred. In logit space, Synovitis
reconstructs from Effusion alone at R² = 0.55 with a coefficient of +0.843,
and from all 11 other heads at R² = 0.76.

### 1.4 Where the error sits

All 837 discordant pairs, split by the effusion status of each member:

| positive / negative | pairs | misranked | share of total error |
|---|---|---|---|
| Eff=1 vs Eff=1 | 286 | 38.8% | **44.4%** |
| Eff=1 vs Eff=0 | 396 | 18.9% | 30.0% |
| Eff=0 vs Eff=1 | 65 | **55.4%** | 14.4% |
| Eff=0 vs Eff=0 | 90 | 31.1% | 11.2% |

Within-stratum pairs are 376 of 837 (**44.9%**) and are the entire addressable
headroom. The Eff=0-positive vs Eff=1-negative cell is misranked worse than
chance; that is the gating tax, paid on the 5 positives with no effusion.

### 1.5 Post-hoc recombination is closed

Five-fold CV, no leakage, logistic on logit-transformed heads:

- 11 other heads → Synovitis: **CV AUC 0.606**
- all 12 heads → Synovitis: **CV AUC 0.601**

Both below the 0.701 already in hand. No stacking, calibration, or head-mixing
recovers anything. Any fix must change what the trunk computes.

### 1.6 The unresolved number

**Effusion-stratified conditional AUC = 0.630, 95% CI [0.464, 0.786],
P(AUC ≤ 0.5) = 0.060.**

This is the only quantity that measures synovitis-specific skill. It is 76%
driven by the Eff=1 stratum (22 pos / 13 neg) and 24% by the Eff=0 stratum
(5 pos / 18 neg). At n=58 it cannot be resolved. It sits right at the edge of
significance, which is why the whole plan is gated on it.

---

## 2. Logic audit — three corrections

### 2.1 Cause A is not as closed as the report states

The silence-reweighting experiment was judged on **pooled** AUC (0.682 → 0.694,
declared inside noise). But pooled AUC is ~55% cross-stratum pair mass, and
cross-stratum ranking is driven by effusion coupling, which synovitis label
reweighting does not touch. The experiment was scored on a metric substantially
insensitive to the thing it was designed to change.

Back-solving that run through the pair-mass decomposition puts its conditional
AUC at roughly 0.588 → 0.614, a directional gain of about +0.026 that nobody
measured. **This is an estimate, not a measurement** — it assumes cross-stratum
errors held constant at 111, which cannot be verified without the run's saved
predictions.

**Action:** re-score the saved silence-reweighting predictions on conditional
AUC. Free if the predictions exist. Cause A should be treated as *open* until
this is done, not closed.

### 2.2 "Ship the frozen path now" was premature

The frozen DINOv2 gap is +0.045 (0.746 vs 0.701). Bootstrap SD on this gold set
is 0.069, so the gap is **0.66 SD** — not resolvable by an unpaired comparison.
It needs a paired bootstrap on matched studies.

More importantly, shipping it would be justified on pooled AUC, which §1.1 shows
is a broken metric for this finding. If the frozen path's advantage is only a
better effusion proxy, shipping it improves the reported number without
improving capability.

**Revised:** ship the frozen path for synovitis only if it wins on *both* the
paired pooled comparison and conditional AUC. Both fall out of Stage 0 anyway.

### 2.3 The 0.867 headroom is not a guaranteed ceiling

The decomposition treats within-stratum and cross-stratum error as separable,
but one model produces both. A residualised model could improve within-stratum
discrimination while degrading the gating, netting zero. Treat 0.867 as the
arithmetic bound under an unchanged gating, not as a reachable target.

### 2.4 One risk that survives the audit

Stratified ranking (Stage 1) strips away effusion — the one *reliable* cue —
and leaves the model to learn the residual from synovitis labels that are 82.2%
silent against 46.6% true prevalence. The residual task is harder and the labels
are noisier relative to the remaining signal. This is the main way Stage 1 could
fail, and §2.1's re-scoring is the cheapest early read on whether label quality
binds the conditional task.

---

## 3. Metric change (do this first, no gate)

Replace pooled AUC with **effusion-stratified conditional AUC** as the primary
metric for synovitis. Report pooled alongside it, never alone.

Rationale: pooled AUC of 0.698 is obtainable from the effusion label by itself.
It cannot detect progress on the actual task until the gain is very large. This
is why seven consecutive experiments read as null.

Exact conversion, fixed by this gold set:

| conditional AUC | → pooled AUC |
|---|---|
| 0.630 (now) | 0.701 |
| 0.70 | 0.733 |
| 0.75 | 0.755 |
| 0.80 | 0.778 |
| 0.85 | 0.800 |
| 1.00 | 0.867 |

Stratifying on **predicted** effusion rather than the gold label gives 0.617
against 0.630 — close enough that the metric is implementable during training,
not just at gold-set evaluation.

Never judge on macro AUC. A perfect synovitis fix moves it by 0.022, inside one
standard error.

---

## 4. Stage 0 — the gate

Free. Uses runs that already exist.

**0a. Conditional AUC of the frozen DINOv2 predictions** on the same 58 studies.

Decisive because the frozen path has only a small aggregation head and cannot
have fit the mined label noise. If its general-purpose features discriminate
synovitis *within* effusion strata, the signal is in the pixels.

**0b. Re-score the silence-reweighting run** on conditional AUC (§2.1).

**0c. Contrast-series inventory.** Grep `SeriesDescription` and `ProtocolName`
in `index.parquet` for post-contrast markers; check for a `ContrastBolusAgent`
tag; cross-check against repeat T1-FS series within an exam as a positive
control. Absence of a string is not absence of contrast.

**0d. Series-composition confound.** Failure sheet `worst05_317032662220`
(Synovitis pred 0.76, truth 0) carries two distinct series both labelled
`sag fluid fatsat R`. If aggregation is a mean over series, that study's fluid
signal enters at double weight. Test: does the synovitis score scale with the
count of fluid-sensitive fat-suppressed series per study? Computable from
`index.parquet` plus predictions already in hand.

### Gate decision rule

| 0a result | reading | next |
|---|---|---|
| Frozen conditional AUC clearly > 0.5 | synovitis-specific signal survives conditioning on effusion | Stage 1 |
| Frozen ≈ 0.5, pooled 0.746 | advantage is purely better effusion coupling | Stage 3 |
| Ambiguous (CI straddles 0.5) | expected at n=58; combine with the fine-tuned 0.630 as two semi-independent readings | Stage 1 at reduced confidence, and prioritise §6 |

If 0c finds enhanced series being routed as ordinary fluid-sensitive ones, that
supersedes everything below. Fix it first.

---

## 5. Stage 1 — train the conditional task directly

**Stratified pairwise ranking loss.** Sample positive/negative pairs only from
within the same effusion stratum.

This is not an analogy to the metric; conditional AUC *is* within-stratum pair
concordance, so the loss optimises exactly what is being measured.

Feasible because effusion mining is reliable (12.0% report silence) even though
synovitis mining is not (82.2%). The strata are trustworthy even where the
synovitis labels inside them are not.

**Stratify on the predicted effusion score, not the mined effusion label.**
Predicted effusion is continuous and scores AUC 0.888; mined labels carry their
own error that would blur the strata. Match pairs on similar effusion score
rather than a hard binary split.

**Residual parameterisation** (alternative or complement):

```
syn_logit = β · stopgrad(eff_logit) + g(features)
```

with `g` carrying all trainable synovitis capacity, forcing it to learn
non-effusion signal by construction.

Calibrate β empirically on out-of-fold predictions. **Do not derive it from
r_true = 0.403** — a rank correlation between two binary labels does not map
cleanly onto a logit coefficient.

**If using a decorrelation penalty, do not target zero.** True coupling is
r = 0.403, OR = 6.09. Driving the heads to independence optimises toward a wrong
target. Aim at the observed truth coupling.

**Success criterion (pre-registered):** conditional AUC 0.630 → ≥ 0.75, tested
by paired bootstrap on the same 58 studies. Expected pooled AUC ≈ 0.755.

---

## 6. Stage 2 — expand the gold set

Probably the highest-leverage item in this document, and it is not a modelling
change.

44.9% of addressable signal rests on comparisons involving 13 negatives (Eff=1
stratum) and 5 positives (Eff=0 stratum). The conditional AUC CI is 0.32 wide.
A genuine move from 0.63 to 0.80 may not clear significance on 58 studies.

**Target ~150 radiologist-labelled studies.** This resolves the §1.6 question
outright rather than gating on a proxy, and makes every subsequent experiment
interpretable.

**Interim substitute:** the ~17.8% of the 4,349 training studies whose reports
*do* mention synovitis are high-precision positives. A precision-oriented
conditional AUC proxy on that subset has far more power than n=58, at the cost
of a severity-biased positive sample.

**Also capture at labelling time:** what the reader had access to. If the gold
reader saw contrast series, the report, or clinical context, part of the target
is outside the model's input and no modelling closes the gap. This was never
recorded and it bounds everything.

---

## 7. Stage 3 — anatomical targeting (a bet, not a confirmed step)

Crops centred on Hoffa's fat pad and the suprapatellar pouch, replacing
whole-knee pooling.

Motivation is external: the best published result on this task (AUC 0.76,
roughly matching senior radiologists at 0.77) came from explicit segmentation of
the synovial capsule and infrapatellar fat pad. A ConvNeXt-T pooling 16 slices ×
5 series into one score has no mechanism to concentrate on a structure occupying
a small fraction of the volume.

**Not measured on this data.** Literature-motivated only. Second thing to try.

Two hypotheses already tested on the failure sheets and **dropped**: uniform
slice sampling wasting out-of-joint slices (0–8.6% of panels below 15% tissue),
and edge-slice structural degeneracy (edge energy flat across z0–z23). Neither
held. The sheets are rendered figures rather than preprocessed tensors, so they
are weak proxies regardless.

---

## 8. Stage 4 — if the gate fails

Descope deliberately rather than by attrition.

Report synovitis as effusion-coupled. State conditional AUC alongside pooled so
readers can see the head carries no independent information. Cite the
non-contrast limitation: effusion-synovitis is defined in the literature as
fluid distension that cannot be accurately discerned from synovitis without
contrast, and Hoffa-synovitis is an explicitly non-specific surrogate with among
the lowest reader reliability of any MOAKS item.

This is an honest negative result, not a failure.

---

## 9. Data still needed

| Item | Purpose | Blocking |
|---|---|---|
| Frozen DINOv2 predictions on the 58 (Synovitis + Effusion, same UID key) | Stage 0 gate | **yes** |
| Saved silence-reweighting predictions on the 58 | §2.1 re-score | yes for that check |
| `index.parquet` (SeriesDescription, ProtocolName, SeriesNumber, ContrastBolusAgent, plane, fat-sat) | 0c, 0d | yes for those |
| OOF predictions on the 4,349 | §6 interim proxy | no |
| Any synovitis severity grade, even a subset | severity-bias test | no |
| Gold reader's series access | ceiling determination | no |

---

## 10. Decision table

| Check | Result | Action |
|---|---|---|
| 0c contrast series | present, misrouted | fix routing first; supersedes all |
| 0c contrast series | absent | proceed; hard non-contrast ceiling applies |
| 0a frozen conditional | > 0.5 clearly | Stage 1 |
| 0a frozen conditional | ≈ 0.5 | Stage 4 descope |
| 0a frozen conditional | ambiguous | Stage 1 at low confidence + prioritise Stage 2 |
| 0b reweighting conditional | improved | Cause A reopens; combine label fix with Stage 1 |
| 0b reweighting conditional | flat | Cause A genuinely closed |
| 0d series count correlation | present | series-type-aware aggregation |
| Stage 1 conditional AUC | ≥ 0.75 | proceed, then Stage 3 |
| Stage 1 conditional AUC | < 0.70 | Stage 3 before further objective work |

---

## Standing rules

1. Judge synovitis on **conditional AUC**, via paired bootstrap on matched
   studies. Never on macro. Never on pooled alone.
2. Paired tests only. Independent Hanley–McNeil SEs (±0.07 here) hide real
   movement by carrying between-study variance that matched comparison removes.
3. Pre-register the decision rule before each run.
