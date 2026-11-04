# System architecture

How 570 GB of knee MRI and 4,349 free-text radiology reports become twelve
probabilities per study.

This document preserves the original standalone system that scored **0.906** on
the public leaderboard: a five-fold ConvNeXt-T ensemble over a 29 GB image cache,
trained against a label matrix manufactured from radiology reports. Later
experiments and ensembles are summarized in `PROJECT.md`; their generated
artifacts and campaign logs are deliberately excluded from the source archive.

---

## The problem, and why it shapes everything

Predict twelve binary findings per knee MRI study — ACL, MCL, medial and
lateral meniscus, medial, lateral and patellofemoral OA, effusion, synovitis,
Baker's cyst, contusion, fracture — scored by macro-averaged ROC AUC.

There are 4,407 training studies. **Fifty-eight of them carry radiologist
labels.** The other 4,349 carry a free-text radiology report and nothing else,
and reports exist in train but **not** in test.

That single asymmetry determines the whole design. Text cannot be a feature; it
can only be a way of manufacturing training targets. So the system has two
tributaries — one that processes images, one that manufactures a label matrix —
which meet at training and part again at inference. This is a weak-supervision
problem wearing an imaging problem's clothes, and the imaging model tracks its
labels almost mechanically: as the manufactured labels improved from 0.854 to
0.891 on the gold set, the model followed.

The 58 gold studies are the measuring instrument. They never enter training —
not as data, not as a checkpoint selector, not as a source of per-finding
weights.

---

## 1. Overview

```
       IMAGE TRIBUTARY                        LABEL TRIBUTARY

  ┌───────────────────────────┐        ┌───────────────────────────┐
  │ raw DICOM                 │        │ radiology reports         │
  │ 570 GB · 24,371 series    │        │ 4,349 studies · ~11 langs │
  └─────────────┬─────────────┘        └─────────────┬─────────────┘
                ▼                                    ▼
  ┌───────────────────────────┐        ┌───────────────────────────┐
  │ header scan               │        │ dual-mode LLM extraction  │
  │ ~820k headers             │        │ categorical + probability │
  └─────────────┬─────────────┘        └─────────────┬─────────────┘
                ▼                                    ▼
  ┌───────────────────────────┐        ┌───────────────────────────┐
  │ series volume cache       │        │ fusion with the lexicon,  │
  │ 24 × 224 × 224 uint8      │        │ then sharpening           │
  │ 29 GB · 3 shards          │        └─────────────┬─────────────┘
  └────┬─────────────┬────────┘                      ▼
       │             ▼                 ┌───────────────────────────┐
       │   ┌───────────────────┐       │ soft targets + weights    │
       │   │ side classifier   │       │ 4,349 × 12                │
       │   │ L or R per series │       └─────────────┬─────────────┘
       │   └─────────┬─────────┘                     │
       └─────────────┴──────────┬────────────────────┘
                                ▼
                  ┌───────────────────────────┐
                  │ training                  │
                  │ 5 folds · ConvNeXt-T      │
                  └─────────────┬─────────────┘
                                ▼
                  ┌───────────────────────────┐
                  │ 5 fold checkpoints        │
                  └─────────────┬─────────────┘
                                ▼
                  ┌───────────────────────────┐
                  │ ensemble inference        │
                  └─────────────┬─────────────┘
                                ▼
                         submission.csv
```

The gold 58 touch this diagram twice and never as data: they score the
extractors that build the label matrix, and they are printed during training
as monitoring only.

Everything below is a zoom into one of those boxes.

---

## 2. Ingest: DICOM to a 29 GB cache

Raw DICOM is 570 GB and exists only on the competition host. Training against
it directly is impossible, so the first stage reduces every series to a fixed
shape once, and everything afterwards reads that.

```
  ┌──────────────────────────────────────────┐
  │ DICOM series, N files                    │
  └─────────────────────┬────────────────────┘
                        ▼
  ┌──────────────────────────────────────────┐
  │ sort slices by geometry                  │
  │ ImagePositionPatient onto the slice      │
  │ normal — never InstanceNumber            │
  └─────────────────────┬────────────────────┘
                        ▼
  ┌──────────────────────────────────────────┐
  │ choose which slices to decode            │
  │ two-step: N → 24, then 24 → depth        │
  └─────────────────────┬────────────────────┘
                        ▼
  ┌──────────────────────────────────────────┐
  │ decode only the chosen files             │
  └─────────────────────┬────────────────────┘
                        ▼
  ┌──────────────────────────────────────────┐
  │ resize in-plane to 224 × 224             │
  └─────────────────────┬────────────────────┘
                        ▼
  ┌──────────────────────────────────────────┐
  │ percentile stretch, 1st–99th, to uint8   │
  └─────────────────────┬────────────────────┘
                        ▼
  ┌──────────────────────────────────────────┐
  │ series volume                            │
  │ 24 × 224 × 224 uint8 · ~1.2 MB           │
  │ stored as acquired, NOT mirrored         │
  └─────────────────────┬────────────────────┘
                        ▼
              shard by md5(study UID) % 3
```

A **series volume** is one acquisition — sagittal PD, coronal T2, and so on —
reduced to `24 x 224 x 224 uint8`, about 1.2 MB. A study holds roughly 5.5 of
them. In total: 24,371 series, ~29 GB, three shards.

Four decisions here are load-bearing.

**Slices are ordered by geometry, not by `InstanceNumber`.** Slice position
comes from projecting `ImagePositionPatient` onto the slice normal. On the
first real series tested, `InstanceNumber` ran 30 down to 1 — exactly
backwards. Sorting by it reverses the volume and raises nothing.

**Slice selection is two-step, not one interpolation.** Picking 24 of N and
then depth-of-24 lands on different files than picking depth-of-N directly, for
41 of the 44 series lengths between 16 and 60. Training and inference must
decode the same slices, so both go through the same two-step function.

**Intensities are stretched per volume, between the 1st and 99th percentiles.**
MRI has no standardised units; the same tissue on two scanners is two different
numbers. There is no global normalisation that would be correct.

**Sharding is by hash of the study UID, not by position.** Each shard then
holds a representative mix of sites and protocols, and — more importantly —
every series of a study lands in the same shard, which is what lets
study-level reasoning complete within one shard.

---

## 3. Laterality: the sub-system that has to exist

Four of the twelve findings are compartment-specific: medial versus lateral
meniscus, medial versus lateral OA. Getting the side wrong does not add noise,
it transposes those four labels.

And the side is largely missing. `ImageLaterality` is never populated,
`Laterality` is present but empty on most rows, and `SeriesDescription` is
anonymised to a placeholder. From the tags alone 14,210 of 24,371 series carry
a side; study-level propagation recovers a further 193, leaving **40.9% of
series — 9,968 of 24,371 — with no recoverable side from metadata.**

Twenty-six studies are left unresolved deliberately: their own series
contradict each other on the side, which breaks the one-knee-per-study
assumption propagation rests on.

*(Corrected 2026-09-02. This paragraph read "41.7%, 10,161 of 24,371 **after**
propagation"; 10,161 is the count **before** it. The 193-series difference is
exactly what propagation recovers. Verified twice by independent routes —
`tools/audit_orientation.py` over the header index, and the cache build's own
logs, which record 57+88+48 = 193 recovered and 10+11+5 = 26 conflicting
across the three shards.)*

The fix is a small CNN trained from scratch on the 14,210 series whose side
*is* known, predicting the rest. It reaches **98.97% study-level accuracy**.
Two details matter:

- **Slices enter as input channels**, at half resolution. For coronal and
  axial series a single slice settles it — the fibula is lateral. For sagittal
  series the left-right axis *is* the slice axis, so the cue is the direction
  of travel through the volume; a per-slice model with pooling cannot express
  that at all.
- Predictions are combined across a study by a confidence-weighted vote, since
  a study images one knee.

**Mirroring is plane-aware, and deferred to load time.** Left knees are
reflected onto the canonical right so the medial compartment always falls on
the same side of the image — but *which axis to flip depends on the plane*.
Coronal and axial series flip columns. **Sagittal series reverse slice order.**
A blanket column flip on a sagittal series does not swap medial with lateral at
all; it reflects anterior-posterior, putting the patella behind the knee.

The cache stores volumes **as acquired**, with laterality recorded beside them,
and mirroring happens when a volume is read. That is what made the sagittal
mirroring bug a code fix rather than a 29 GB rebuild.

---

## 4. Manufacturing the label matrix

The reports are **multilingual** — around eleven languages, English about 39%,
then Greek, Croatian and Serbian, Spanish, Turkish, German, Bulgarian and
Russian, Dutch, French, Italian, Portuguese. They are read in the original;
nothing is translated first. The current baseline runs a locally hosted quantised
Qwen2.5-Instruct. The host clarification supplied September 29 permits hosted
inference, subject to accessibility and minimal-cost requirements; local inference
is an implementation choice, not a blanket competition requirement.

```
                      ┌─────────────────────┐
                      │ radiology report    │
                      └──┬───────┬───────┬──┘
            ┌────────────┘       │       └────────────┐
            ▼                    ▼                    ▼
  ┌───────────────────┐ ┌──────────────────┐ ┌──────────────────┐
  │ categorical pass  │ │ probability pass │ │ rule-based       │
  │ presence,         │ │ 0–100 per        │ │ multilingual     │
  │ severity,         │ │ finding, incl.   │ │ lexicon          │
  │ certainty,        │ │ the unmentioned  │ │                  │
  │ evidence span     │ │                  │ │                  │
  └─────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘
            └─────────┬──────────┘                    │
                      ▼                               │
          ┌─────────────────────────┐                 │
          │ mode fusion             │                 │
          │ score average           │                 │
          └────────────┬────────────┘                 │
                       └───────────┬──────────────────┘
                                   ▼
                     ┌─────────────────────────┐
                     │ weighted blend          │
                     │ 0.90 LLM · 0.10 lexicon │
                     └────────────┬────────────┘
                                  ▼
                     ┌─────────────────────────┐
                     │ sharpen: quantile map   │
                     │ onto the categorical    │
                     │ marginal                │
                     └────────────┬────────────┘
                                  ▼
                     ┌─────────────────────────┐
                     │ 4,349 × 12 soft targets │
                     │ and 12 weights          │
                     └─────────────────────────┘
```

**Two extraction modes, and both are needed.** The categorical pass returns
presence, severity, certainty and the exact span of text it relied on, which is
then scored through fixed severity and certainty factors. The probability pass
returns 0-100 per finding and is explicitly asked to estimate findings the
report never mentions, from context.

The reason for the second mode is mechanical: the categorical pass ties 369 of
696 gold label slots on a single value, and **AUC cannot order anything inside
a tie block.**

**Averaging beats choosing.** On the gold set: categorical alone 0.838,
probability alone 0.860, a per-finding oracle that picks the better mode each
time 0.885, and a plain average of the two **0.889**. The oracle loses to the
average, which is the usual sign that per-finding selection is fitting noise.

**Silence is weighted down, never imputed.** A report that narrates several
positives and says nothing about the ACL is not asserting a negative — but it
is not saying nothing, either. `P(positive | unmentioned)` ranges from about
0.05 for MCL to 0.58 for effusion. Using that as a *target* measured worse than
zero on every variant tried, so silence gets a target of zero and a low weight
instead:

| both extractors | weight |
|---|---|
| informative, agree | 1.00 |
| one informative, one silent | 0.70 |
| informative, disagree | 0.40 |
| neither informative | 0.25 |

The silent weight is then scaled by how often that finding is silent across the
whole corpus. Synovitis goes unmentioned in 82.2% of reports, so its silent
weight falls to 0.044 — a corpus that mentions something once in six studies is
not making a statement by omitting it. The ACL is silent in only 9.2%, so it
keeps nearly the full 0.25: a report that discusses ligaments and skips the ACL
*is* saying something.

**Nothing in this pipeline is fitted on the gold 58.** An earlier version fitted
24 fusion weights on them and scored 0.860 in-sample against 0.812 under
leave-one-out — worse than the LLM alone. Every constant here is either a
global blend chosen from a plateau or an observation measured over the 4,349
mined studies.

---

## 5. The model

One study in, twelve logits out. The interesting part is that a study is not an
image — it is a variable-length set of series, each of which is a stack of
slices.

```
                        study
                          │
                          ▼
  ┌─────────────────────────────────────────────────┐
  │ up to 5 series × 16 of 24 slices                │
  └───────────────────────┬─────────────────────────┘
                          ▼
  ┌─────────────────────────────────────────────────┐
  │ 2.5D stack: previous / current / next slice     │
  │ as the three input channels                     │
  └───────────────────────┬─────────────────────────┘
                          ▼
  ┌─────────────────────────────────────────────────┐
  │ ConvNeXt-T backbone        768 features / slice │
  └───────────────────────┬─────────────────────────┘
                          ▼
  ┌─────────────────────────────────────────────────┐
  │ attention pool over slices           → a series │
  └───────────────────────┬─────────────────────────┘
                          ▼                              plane,
  ┌─────────────────────────────────────────────────┐    contrast,
  │ FiLM conditioning, 7-dim series metadata        │◄── predicted
  └───────────────────────┬─────────────────────────┘    side
                          ▼
  ┌─────────────────────────────────────────────────┐
  │ transformer over series                         │
  │ 2 layers · 8 heads · padding-masked             │
  └───────────────────────┬─────────────────────────┘
                          ▼
  ┌─────────────────────────────────────────────────┐
  │ attention pool over series            → a study │
  └───────────────────────┬─────────────────────────┘
                          ▼
  ┌─────────────────────────────────────────────────┐
  │ LayerNorm · dropout · linear                    │
  └───────────────────────┬─────────────────────────┘
                          ▼
                      12 logits
```

**2.5D, not 3D.** Neighbouring slices become the three input channels, so
ImageNet-pretrained ConvNeXt weights apply unchanged and each forward pass still
sees local through-plane context. Pretrained weights are explicitly permitted by
the competition, and a 3D trunk would give them up.

**Attention pooling twice, never a mean.** A meniscal tear occupies two or
three patches on two or three slices. Averaging over 16 slices and then over 5
series buries it under normal tissue — which is exactly the failure the first
model showed, scoring 0.865 on regional findings like effusion and OA against
0.695 on focal ones like meniscus and ligaments.

**Metadata enters once, through FiLM**, as a seven-dimensional vector per
series: fluid-sensitive, fat-suppressed, the plane one-hot, and the predicted
side with a flag for whether it was known. Side therefore enters the model
twice — the image is mirrored geometrically, *and* the side is a covariate.

**Masking is load-bearing.** Studies have different series counts, so batches
are padded, and the padding must be invisible in three separate places: the
attention pool's softmax, the transformer's key-padding mask, and the
normalisation between them. A padded series changing a study's prediction is a
tested-against failure, not a theoretical one.

The trunk is ConvNeXt-T at 27.8M parameters, 768 features. The competition
cluster is tight and the ceiling here is labels, not capacity: the slice and
series ladder ran 0.845 to 0.869 to 0.889 to 0.891, with the last rung buying
0.002 for 48% more inference.

---

## 6. Training and the fold ensemble

Five folds, stratified by how many findings a study has positive, **split by
study**. A study's ~5.5 series image the same knee in one session, so splitting
by series would put near-duplicates on both sides of the boundary.

- Effective batch is **1 study x 2 GPUs x 4 gradient accumulation = 8 studies**
  per optimiser step. One forward pass is 5 series x 16 slices = 80 slice
  images per rank.
- AdamW at 1e-4, OneCycle to a 3e-4 peak, 12 epochs, mixed precision. About one
  hour per fold.
- **Checkpoint selection reads the mined validation fold.** The gold 58 are
  scored every epoch and printed, and select nothing — 58 studies carry a macro
  standard error of 0.015 to 0.025, so selecting on them is choosing among noise
  while also spending the only honest measurement available.
- Every run also accumulates an EMA and an SWA average alongside the monitored
  checkpoint, so a fold ships three candidate weight sets and can be read
  without committing to a selector.

The shipping ensemble is the best-epoch family, five folds:

```
fold   selected epoch   val macro   gold macro
  0          6            0.8814      0.8683
  1          6            0.8726      0.8747
  2          8            0.8675      0.8659
  3          8            0.8736      0.8915
  4          6            0.8697      0.8753
```

Against a single fold's 0.889 on the leaderboard, the five-fold ensemble scores
**0.906**. That is +0.017, above what fold averaging was expected to be worth —
and worth reading alongside a separate measurement: two *identical*
configurations on different folds landed 0.889 and 0.877 on the public
leaderboard. Cross-split spread is about 0.012, so a single fold's score is
partly a statement about which split drew the friendlier public set. The
ensemble is insurance against that as much as it is an accuracy gain.

---

## 7. Inference

The submission is a notebook, capped at **9 hours**, with **no internet**, and
must write `submission.csv`. Every weight it uses is pre-uploaded as a dataset,
since nothing can be downloaded at run time.

```
   test.csv study list
            │
            ▼
  ┌────────────────────────────────────┐
  │ select ≤5 series, THEN decode      │──────┐ any failure
  └─────────────────┬──────────────────┘      │
                    ▼                         ▼
  ┌────────────────────────────────────┐   ┌──────────────────┐
  │ predict side, vote across series   │   │ training prior   │
  └─────────────────┬──────────────────┘   │ for this study   │
                    ▼                      └────────┬─────────┘
  ┌────────────────────────────────────┐            │
  │ mirror by plane                    │            │
  │ subsample to 16 slices             │            │
  └─────────────────┬──────────────────┘            │
                    ▼                               │
  ┌────────────────────────────────────┐            │
  │ run the 5 fold members             │            │
  └─────────────────┬──────────────────┘            │
                    ▼                               │
  ┌────────────────────────────────────┐            │
  │ probability mean across members    │◄───────────┘
  └─────────────────┬──────────────────┘
                    ▼
              submission.csv
```

Series selection happens **before any file is read** — the study is capped to
five series first, so the loop never decodes images the model will discard.
Members are combined by a plain probability mean across folds, which is what
every recorded leaderboard score including 0.906 was produced by.

**The measured cost, per study:**

```
decode   2.74 s   52% of the loop
forward  2.52 s   over 5 members, 0.50 s each
```

Projected over ~1,500 studies that is **2.18 hours** — comfortably inside the
nine-hour cap, with room for a 24-slice ensemble that was long assumed not to
fit.

The result worth stating plainly: **half the inference budget is DICOM reading,
not the model.** On a single-fold run it is 84%. A sixth fold costs 0.21 hours;
the decoding costs 1.14 hours no matter how many folds share it.

### The Efficiency Prize

A second prize track scores `AUC / (Benchmark - maxAUC) + RuntimeSeconds /
32400`, minimised — a submission using the full nine hours carries a runtime
penalty of 1.0, a thirty-minute one carries 0.056. This is why series and slice
caps exist at all, and why they are deterministic rather than random.

At the current spread it works out to **one hour of runtime being worth 0.050
AUC**. Four extra folds cost 0.83 hours and buy 0.017, which is net negative on
that metric and clearly positive on the main one. So the two final entries are
different submissions: the five-fold ensemble for the AUC track, a single fold
for the efficiency track.

---

## 8. Correctness by construction

Most of what has gone wrong in this project failed *silently* — the run
completed, the submission validated, and the score was merely worse. The
architecture reflects that.

**The notebooks are generated from the source tree.** Neither Kaggle nor Colab
can import this repo, so instead of reimplementing preprocessing in a notebook
— which is precisely how train/test skew gets introduced — a generator embeds
the real module source, and the notebook writes it to disk and imports it
normally. Training and inference therefore provably run the same volume
builder, the same mirroring, the same network. Editing a notebook directly is
the documented anti-pattern, and a test fails if any notebook has drifted from
the source.

**Guards sit where a mistake would be silent rather than loud.** Three worth
naming:

- A **duplicate-fold check.** Each fold is written both as a Lightning
  checkpoint and as an exported state dict, and they contain bit-identical
  weights. Attaching both would give that fold double weight in the ensemble,
  and nothing in the output would say so.
- A **checkpoint-family check.** Best-epoch, EMA and SWA are three views of the
  *same* run; averaging across families double-counts every fold.
- An **active-path check.** An earlier guard asked whether *any* checkpoint had
  been found. With the frozen path retired but its weights still attached, that
  test passed while the path actually in use had nothing — and produced a
  submission of NaN.

**A per-study fallback to the training prior** means one unreadable study
cannot fail a nine-hour run, and post-write assertions check the header
spelling, row count, id uniqueness, NaN and range before the file is trusted.

Behind all of it are **597 tests**, aimed at silent failures rather than
coverage: that a padded series cannot change a study's output, that sagittal
mirroring preserves the side, that nothing is fitted on the gold 58, that
ambiguity in extraction resolves to "unknown" rather than "no".

---

## 9. Key numbers

| | |
|---|---|
| Studies / with radiologist labels | 4,407 / 58 |
| Studies with a report only | 4,349 |
| Report languages | ~11, English 39% |
| Series / cache size | 24,371 / ~29 GB, 3 shards |
| Cached series volume | 24 x 224 x 224 uint8 |
| Training geometry | 16 slices x 5 series |
| Backbone | ConvNeXt-T, 27.8M params, 768 features |
| Series with no side in metadata | 40.9% (9,968 of 24,371, after propagation) |
| Side classifier accuracy | 98.97% at study level |
| Folds | 5, stratified, split by study |
| Training cost | ~1 h per fold on 2 GPUs |
| Inference cost | 2.18 h for 5 folds over ~1,500 studies |
| Metric | macro-averaged ROC AUC over 12 findings |
| Public leaderboard | **0.906** |
