# Validation and standalone-model reset — 2026-09-24

The current evidence cannot reliably rank small improvements in image-finding
accuracy. More report-labelled validation or a narrow bootstrap interval does
not repair that limitation. Keep the user-reported 0.917 ensemble as a frozen
reference and preserve the earlier 0.914 artifacts; the
development target is one image model, with no blending gate or rescue blend.
No training, submission, relabeling, or historical artifact replacement was
performed for this reset.

**Verified:** supervision has both extraction errors and report omissions;
gold has repeatedly influenced development; selected-checkpoint validation
reuses selection observations; study-level exclusion is stronger than our
patient-level evidence; and recent architectural changes have not established
a transferable standalone gain. **Hypotheses:** scanner/language shortcuts,
remaining input failures, and optimisation limits may contribute. Their sizes
are unknown. A larger backbone is not the evidence-supported next purchase.

The reproducible audit is `tools/audit_validation_reset.py`; outputs are in
`outputs/validation_reset_20260924/`, with input and output SHA-256 manifests.
All new counts below were recomputed locally. Historical results are identified
as such rather than represented as new experiments.

## 1. Reconcile the actual latest state

| Artifact | What it establishes |
| --- | --- |
| `outputs/kaggle_combined20_submission/competition_receipt.json`, `score_update_20260924.json` | Latest saved submission: **56505208**, September 23 22:56 UTC, 20 members. Local proxy macro 0.885228 → 0.890410, reported paired CI [+0.0042,+0.0063]. User subsequently reports public **0.917**, **+0.003** over 0.914. This corrects the initially unavailable result: an ensemble improvement, not a failed score or proof of standalone improvement. The original null-score receipt is preserved; the score update is user-reported, not independently API-verified. |
| `outputs/kaggle_resolution384_submission/competition_receipt.json` | Earlier submission **56480789**; last saved status pending, score null. This is a historical status, not a current result check. |
| `outputs/kaggle_resolution384_submission/submissions_after.csv` | Saved authenticated list verifies 56196386 = **0.914** and 56222932 = **0.914**. The latter repairs the earlier documentation's uncertainty about the overlay entry ID; its blank description alone does not bind its notebook bytes. |
| `docs/label_quality_result.md`, `docs/label_quality/decisions.md` D7 | September 23 solo report audit completed and stopped; no candidate labels, training, or submission followed that audit. Separate from the submitted 20-model experiment. |
| `outputs/resolution384_comparison/summary.json` | Reproduced protocol224 pooled proxy 0.872879; 384 0.873693, delta **+0.000813**. This compares one held-out model per study, not standalone public scores or full test-time ensembles. |

The earlier campaign called for standalone +0.005 on folds 1/3 and a seed
repeat before expansion (`docs/multiplane_campaign_plan.md`). Saved 384 versus
historical protocol references is **−0.003832 / −0.003826** on those folds;
hardware/runtime differ, so this is not the missing matched causal experiment.
The available artifacts do not establish that the campaign's promotion gates
were met. Later blend improvement cannot establish them retroactively.

Stale statements to retire as current guidance:

- Gold is not an “honest/untouched” test merely because gradients exclude it.
  `src/labels/fuse.py` and `mode_fusion.py` themselves describe choices made
  after looking at gold. Their “in-sample and held-out are the same” comments
  are statistically incorrect for this development history.
- Report-label AUC is not a measured ceiling on image prediction. “Label
  quality is not the bottleneck” and “capacity was the binding constraint” in
  PROJECT.md are historical hypotheses, not universal findings.
- The often-repeated epoch-9/20 memorisation curve belonged to **DINOv2**.
  `docs/worklog.md`, August 29 discussion around lines 2100–2150, explicitly
  corrects its extrapolation to ConvNeXt. Current 384 curves do decline after
  their best mined-label epoch, but this alone does not prove memorisation of
  image-label errors or prove that EMA/SWA would help.
- D7's **0.01 leaderboard detection limit** has no recoverable paired hidden
  predictions or variance calculation supporting it. The arithmetic
  +0.03 PF OA / 12 = +0.0025 macro is correct; “therefore always inconclusive”
  and a universal 0.01 resolution are not established.
- A single reviewer agreeing with a same-day reread measures repeatability,
  affected by recall. It is neither a second reader nor image adjudication.

Historical files and receipts remain intact. This document and the new active
PROJECT.md entry supersede conflicting forward instructions.

## 2. Validation credibility audit

**Label provenance.** `data/meta/train.csv` has 4,407 unique studies; 58 have
all 12 organizer-supplied binary labels, 4,349 have none. Both local
`gold_truth_58` copies match train.csv exactly. Documentation attributes the
58 to radiologist image annotation; local files contain no original reader
IDs, blinding, adjudication, selection scheme, or per-case source-image
evidence. Treat them as the organizer's image reference with those provenance
limits, not a locally reverified infallible standard.

Training supervision comes from reports, not those 58 image references:
categorical 14B extraction (`llm_labels.parquet`), probability-mode 32B
extraction (`llm_probs.parquet`), categorical/probability score average,
0.10 lexicon contribution, then quantile sharpening against the categorical
marginal. Per-cell weights are 1.0/0.4/0.7/0.25 for corroboration, disagreement,
one source, and silence. Probability mode can infer unmentioned findings;
LLM agreement is not independent image evidence. The dual-mode pipeline means
“all silence targets are exactly zero” is too broad: probability estimates and
quantile sharpening can retain nonzero scores. The low weights do not make
missing findings reliable negatives.

`fuse.build()` reproduces **104,376/104,376** target/weight cells with maximum
difference **0**, and zero gold IDs enter the mined table. That verifies
reproduction, not truth. Label transforms are fitted on the entire mined
corpus before folds; future strict nested evaluation must fit learned label
marginals/calibration on training partitions only. Current historical labels
are fixed inputs with this transductive limitation disclosed.

Recomputed historical gold diagnostics: fused targets at 0.5 miss **110/240
positive slots (45.8%)** and call **33/456 negatives positive (7.2%)**. These
are micro counts within the reused 58, not new population estimates. Gold
positives range from 9 MCL to 35 Effusion. The mined corpus calls only 109/4,349
Fracture positive versus 18/58 gold: neither prevalence can simply be used as
the true deployment prevalence. Thresholding a severity/certainty target at
0.5 further entangles finding presence with extraction semantics.

The recent report audit is a different measurement: PF OA **23/100** random
controls judged incorrect under the participant's report rubric (Wilson
15.8–32.2%); the grade-II definition is a participant decision, not verified
competition annotation policy. The first 50 contained five agreed missed
mentions and six threshold/definition cases. Contusion had 24/50 controls
undecidable from text; its low definite error count does not establish that
its image labels are accurate. The proposed PF OA rule's approximately 57%
precision is on design reviews, not independent confirmation. References:
`docs/label_audit_v2.md`, `docs/report_review_rubric.md`, `docs/label_quality_result.md`.

**Splits and identity.** `src/model/splits.py` shuffles studies using seed 0,
stratifying by capped number of mined positives (>0.5); sizes are
871/871/870/869/868. Training generators set `GROUPED_FOLDS=False`. This is
study grouping, not patient or site grouping. Our checks of ten packaged
protocol224/384 selected models found exact expected training/validation
membership, zero study overlap, zero gold training IDs, and matching weight
hashes. Protocol reference predictions also match their historical receipts
exactly. Legacy 16/24 training exclusion remains reconstructed rather than
independently proved; see `docs/label_review_v1.md` and historical comparison
receipts.

The available index contains **819,078 instances, 24,371 series**, with no
duplicate SOP/series/study IDs or SOP/series IDs shared across studies.
Visible test IDs/series do not overlap training; only three visible test
studies are available. Hidden train/test patient overlap cannot be checked.

New duplicate checks found **46 exact-report groups / 177 studies**, 40 groups
spanning folds. Normalizing Unicode, case and whitespace yields **54 groups /
204 studies**, 48 spanning folds. One exact/normalized group includes gold
and mined studies. A deliberately narrow near-text search found 13 additional
pairs, 10 cross-fold. Reports can be templates for different people: these
are leads for image/identity review, not proof of leakage. CSVs preserve IDs,
hashes and similarity without report text. No full pixel hash/perceptual
duplicate scan was possible locally: only one raw development series is here.

PatientID, dates, accession, institution, station and scanner serial number
are absent from the available index. Repeat, bilateral and longitudinal
examinations therefore cannot be reliably linked. Request organizer-provided
deidentified patient/site groups if available; otherwise scan allowed raw
headers and full image data in their existing environment. Compute exact
decoded-pixel hashes and geometry-aware near-duplicate candidates; adjudicate
candidate matches. Quarantine unresolved links from opposite sides of a new
split. Do not infer patient identity from scanner or report language.

**Domains and inputs.** There are **58 manufacturer/model/field-strength
proxies**, with two studies carrying multiple proxies. Gold covers 29.
**4,340/4,349** validation studies have a proxy present in that fold's training
set; random folds mostly test familiar equipment. Scanner proxy is not site:
different hospitals can share one model and a hospital can use several models.
Even a leave-proxy-out split does not establish leave-site-out performance.
Heuristic language counts and scanner/vendor cross-tabs are saved. Gold has
28/58 English reports versus 1,786/4,349 mined; French has only 11 mined and
zero gold. Language is an imperfect reporting-domain marker, not patient
demographics or verified site identity.

Saved subgroup proxy diagnostics show PF OA 0.696 in Croatian reports
(120 mined positives/237 negatives) versus 0.823 in Turkish (50/376), and
0.637 for TOSHIBA (47/133). These exploratory strata combine image domains,
case mix and label bias; they do not isolate domain shift. No unseen-scanner
training experiment has been run in this reset.

Historical input checks are strong but bounded: 500 reconstructed 224 series
were byte-exact; geometry/plane agreed on all 24,371 series; protocol selection
removed loss of an acquired plane (228 studies under uniform selection → 0),
but still omitted 566 protocol combinations. References:
`data/meta/input_audit.json`, `outputs/geometry_audit/summary.json`, PROJECT.md.
Raw/cache equivalence does not prove clinically adequate detail or coverage.
The side classifier's 98.97% is **383/387** tagged validation studies;
Philips-held-out is 308/318 (96.86%). Extrapolating that to “45 actual wrong
mirrors” in the untagged population is not a measured count. Review orientation
and coverage alongside the new image reads. `side_validation.csv` is the
receipt. Qualified full-series review, not failure montages alone, must
establish whether an apparent error is actually a missed image finding.

**Selection and uncertainty.** `tools/make_notebook.py` monitors `val_macro`
in `ModelCheckpoint`; `src/model/lightning.py` computes it after converting
soft labels with >=0.5. Selected models' exported predictions are scored on
those **same** observations by `tools/compare_oof.py`. They are excluded from
gradients, but not from checkpoint selection. Outer unbiased evaluation would
require separate inner checkpoint selection. Repeated fold-1/3 architecture
and label screens add experiment-selection bias. Pooled OOF AUC also compares
scores from different models across folds; within-fold means differ:
protocol224 **0.876884** versus pooled **0.872879**, 384 **0.878580** versus
pooled **0.873693**. Neither should be called the test-time ensemble AUC.

Gold exposure is documented, not speculative: original gold-based epoch
selection; fitted per-finding fusion (0.860 in sample / 0.812 LOO); selection
of score-average and lexicon weight after gold sweeps; prompt variants and
self-consistency runs; anatomical/label rules; slice-count decisions; failure
montages; synovitis closure; repeated gold monitoring each epoch. Evidence:
PROJECT.md historical sections, `src/labels/fuse.py`, `mode_fusion.py`,
`docs/worklog.md`, `failures/`, `data/meta/gold_sweep*`. LOO/jackknife after
choosing a recipe on the whole gold set does not remove this adaptive history.

Keep three uncertainties separate:

1. **Sampling:** paired study bootstrap with fixed predictions estimates
   resampling uncertainty under those labels and that sampling frame. Resample
   patients jointly across findings/models once identity is available.
2. **Training/extraction variability:** estimate from matched repeated seeds
   and, where relevant, repeat extraction. The historical ~0.004 referred to
   extraction repeatability, not a universal CNN seed SD or gold sampling SE.
3. **Selection and systematic reference bias:** neither the fixed-prediction
   bootstrap nor more seeds removes them. A narrow interval around a proxy
   contrast can confidently describe the wrong target. Gold n=58 cannot
   certify tiny gains; no universal 0.02 significance threshold follows either.

The metric implementation uses tied ranks correctly but `nanmean` can silently
change the macro's denominator when a finding has one class. New evaluation
must report an unestimable primary metric rather than quietly averaging fewer
than twelve findings.

## 3. Which local signals transferred?

These are historical diagnostics, not a public-score optimization loop.

| Change | Local evidence | Public evidence / limitation |
| --- | --- | --- |
| More slices / series | Gold +0.023 for 8→16 slices; +0.021 for 3→5 series | Historical public +0.024 / +0.020. Large input changes transferred. Worklog's transfer table and PROJECT.md. |
| Stronger extracted labels | Gold extraction 0.854→0.891 | Categorical 0.845; dual-mode 0.827; sharpened 0.843. Better extractor ranking did not guarantee better image learning; pipelines also changed. |
| Report auxiliary loss | Validation improved; gold +0.006 | Same-fold public 0.889→0.880. A report-aligned objective can improve the proxy while hurting the target. |
| 20 epochs + warmup + fixes | Gold +0.021; mined val +0.002 | 0.889→0.888; bundled changes, not an isolated epoch experiment. |
| Same recipe, folds 1/3 | Fold 3 gold slightly higher | 0.889 versus 0.877. Split effects and gold uncertainty; not a measured seed distribution. |
| 16/24 families | Separate pooled proxy 0.86532/0.86569; combined 0.87668 | Both families' public ensembles 0.906; combined 0.911. Evidence about averaging, not improved single models. |
| Protocol selection | Matched fold-1/3 standalone proxy +0.009543/+0.005809 | Expanded ensemble 0.911→0.914. Supports keeping this baseline recipe, but does not establish protocol single-model public improvement. |
| Slice transformer | Standalone 0.877965→0.870383 on pooled folds 1/3; own-control blend rose | Overlay stayed 0.914. Complementarity did not establish a useful standalone improvement. |
| 384 / combined20 | 384 pooled +0.000813 vs protocol; combined20 proxy +0.005182 vs core | Combined20 public **0.917**, **+0.003** over the 0.914 core, per the user's score clarification. The local ensemble improvement transferred in direction. The 384-only score remains unavailable; no standalone transfer claim follows. |

The eleven-cell label patch failed its old-label screen (−0.003086, CI
[−0.010543,+0.004496]); that is neither proof those corrections were wrong
nor a test of corpus-wide supervision repair. Spatial, classwise and V2 also
failed standalone screens or lacked complete replication. Preserve their
artifacts, but do not rerun them because a blend looked attractive.

## 4. Practical new validation protocol

**Estimand:** equal-weight macro ROC AUC over all twelve independently reviewed
image findings, for one frozen image model on the eligible study population.
Also report every finding, label coverage, and prespecified domain strata.
Internal representativeness cannot prove representativeness of the hidden
competition population, whose site mixture and annotation details are unknown.

**Partitions and exposure:**

- Keep old gold and all historical experiment readouts in development history.
  Keep the existing 120-pair / 117-study *report* reserve closed. It is a
  text-rule check, not an untouched image test.
- The prepared 120-study simple-random pilot is development only. It excludes
  known reviews, protected reserve studies, gold, and detected text-linkage
  leads; eligible pool 3,476, inclusion probability 120/3,476. It represents
  that restricted pilot frame, not automatically all 4,407 studies. Prior
  informal exposure is unknown; historical models have trained on these studies.
- Aim for **300 image-reviewed development studies**: the pilot 120 for rubric,
  feasibility and checkpoint selection; a further 180 for screening frozen
  interventions. Keep enriched error-review/training cases separate. Reserve
  membership before opening predictions or findings, after identity checks.
- Prefer additional previously unseen, eligible studies from the target
  acquisition population for final evaluation. If only the existing corpus is
  available, a custodian can prospectively hide new image labels and hold out
  studies/patient components from *all future* training and selection. Retrain
  both arms from the original pretrained initialization excluding development
  evaluation and final membership. Label this a **prospectively locked internal
  evaluation after historical exposure**, not a pristine external test.
- Hash and freeze population, exclusions, duplicate/patient graph, membership,
  model recipe, weights and label versions. Check all exclusions, not only the
  58 gold IDs. No full old ensemble or old all-data checkpoint is eligible to
  provide independent scores on newly reviewed training studies. Verified
  same-fold predictions are useful interim diagnostics only.

**Sample size comes from counts and precision, not the old 1,000-study target.**
An initial useful goal is at least **100 resolved positives and 100 negatives
per finding** in final evaluation. At AUC 0.85, 100 positives/900 negatives
gives approximate individual-AUC 95% half-width **0.048**. For 200/800 it is
**0.035**. These support detecting gross problems, not 0.003 improvements.
The following planning calculations use the Hanley–McNeil approximation,
independent studies and no weighting; they are scenarios, not a power proof.

| True prevalence assumption | N=500: positives, half-width | N=1,000 | N=2,000 |
| --- | --- | --- | --- |
| 5% | 25, ±0.096 | 50, ±0.068 | 100, ±0.048 |
| 10% | 50, ±0.068 | 100, ±0.048 | 200, ±0.034 |
| 20% | 100, ±0.049 | 200, ±0.035 | 400, ±0.025 |

At 2.5% prevalence, 100 positives needs roughly 4,000 random studies; at 5%,
2,000; at 10%, 1,000. Missing/uncertain labels and clustering increase these
requirements. Do not use the selected 58 or noisy report rates as exact
prevalence. The first 40 pilot studies establish reader time/definitions;
complete 120 for provisional prevalence and uncertainty, then refine on the
rest of development before freezing final enrollment. A rare finding with
zero pilot positives has no useful lower prevalence bound from this pilot.

Default final design is probability sampling of the intended population,
stratified by available site or scanner proxy, language and protocol with
approximately proportional allocation. If random enrollment is too expensive,
use a **declared probability enrichment design**, not an error queue: e.g.
mutually exclusive strata of report-Fracture positive, remaining report-MCL
positive, remaining report-Lateral-Meniscus positive, and all remaining
studies. After the development pilot, fix stratum allocations to target the
100/100 resolved counts, retaining substantial random samples from the
report-negative stratum to find omitted findings. No stratum has zero inclusion
probability. Cross-stratify domains only as sizes allow; small-domain censuses
and oversampling get recorded weights.

For disproportionate sampling, the primary AUC must be a design-weighted
positive–negative pair estimator, including tie=0.5 and the pair's joint
inclusion probability. For simple random sampling within stratum h,
pi_ij=n_h(n_h−1)/[N_h(N_h−1)] for different studies in the same stratum;
between independent strata pi_ij=pi_i*pi_j. Preserve all denominators and
patient-component grouping; use a survey/stratified cluster bootstrap suited
to sampling fractions. Report effective sample size and sensitivity to weights.
This needs a checked analysis implementation before unblinding. If that is
not available, use the representative random core alone for the primary score
and report enriched panels separately. Do not pool them unweighted.

Plan the paired comparison on development scores using joint patient-level
resampling/simulation across all twelve labels, plausible prevalence, missingness,
reader disagreement, model-score correlation, and design effects. Target 80%
power at two-sided 5% for +0.005 macro; also publish power/MDE for +0.01 and
+0.003. For illustration, if paired macro SD at N=1,000 is 0.004, MDE is
about 2.8×0.004=0.0112; +0.005 requires about **5,018** studies under square-root
scaling, and +0.003 about **13,938**. That SD is an assumption, not a measured
property of this project. A 1,000-study set cannot be promised to pass the gate.
Use conservative pilot uncertainty; fix N/budget before final model readout.
If required enrollment is infeasible, report the achievable MDE and keep small
differences inconclusive. Do not add cases after looking for significance.

**Annotation and budget:** two qualified musculoskeletal readers inspect all
available diagnostic series independently, blinded to reports, proxy/gold
labels, predictions, selected-error status and each other's reads. A third
qualified reader adjudicates disagreements and unresolved cases. A lead reader
first reconciles definitions with available organizer guidance: especially
PF cartilage severity, tear versus degeneration, contusion etiology and
synovitis assessability. The participant's grade-II rule is not presumed
clinical or competition truth. Reports are reviewed only after image reads
are committed; original-language competence or translation with verification
is needed for text judgments.

Use present / absent / indeterminate / not-assessable, plus severity, location,
series/image evidence, reader ID, time and rubric version. Keep both original
reads and adjudication, including reasons for changes. Missing reference labels
are never zero. Primary per-finding AUC uses resolved labels with a visible
denominator; if any finding has one class, the twelve-finding macro is
unestimable. Report uncertain/missing proportions by arm-independent strata
and prespecified sensitivity analyses for plausible positive/negative
assignments. Material conclusions that reverse under these analyses remain
uncertain. Preserve revisions as new versions; no silent overwrites.

Budget scenario (assumptions to replace with measured pilot time): 8 minutes
per full-study read × 2, plus 20% of studies needing 5 adjudication minutes
= **17 expert minutes/study**. Add 20% preparation/QA capacity. At an
illustrative contracted rate of $200/hour (not a market quote):

| Scope | Expert read/adjudication hours | With 20% overhead | Illustrative cost |
| --- | ---: | ---: | ---: |
| First 40 | 11.3 | 13.6 | $2,720 |
| 120 pilot | 34 | 40.8 | $8,160 |
| 300 development total | 85 | 102 | $20,400 |
| 1,000 final | 283 | 340 | $68,000 |
| 2,000 final | 567 | 680 | $136,000 |

Six-to-twelve-minute reads, 15–30% adjudication at 3–10 minutes, imply roughly
25–54 expert hours for 120, before overhead. Access/viewer setup is additional.
Training-label annotation is additional too. Thus fully double-read evaluation
is substantial; first buying a 40-study feasibility read has high information
value. If the budget cannot support all-class precision, narrow the supported
claims rather than presenting a small set as decisive.

Reference methods: [Hanley & McNeil's original AUC paper](https://www.med.mcgill.ca/epidemiology/hanley/software/hanley_mcneil_radiology_82.pdf)
supports approximate precision planning; [STARD 2015](https://www.bmj.com/content/351/bmj.h5527)
supports explicit reference-standard, sampling, blinding and missingness
documentation. These do not establish this dataset's label validity or power.

**If qualified review is unavailable:** the existing solo report workflow can
verify quotation, negation, anatomy attribution and faithful extraction under
a declared text rubric. Finish a narrowly frozen rule, then use its untouched
text reserve once; its 20/20 target-based strata need separate estimates or
sampling weights. A same-person delayed reread still is not independent review.
LLM consensus/translation is assistance, not a new reference. Preserve uncertain
or unmentioned image findings as unsupported. Use nested study/component CV
with inner epoch selection and outer mined-label evaluation, plus scanner/domain
stress diagnostics, as the strongest computational fallback. It measures proxy
agreement only. No trustworthy image-level ranking of small supervision gains
is currently available; stop before model promotion on that basis. Public
leaderboard/private macro scores cannot substitute for per-finding labels or
patient linkage. Do not resume the discarded architecture campaign.

## 5. Single-model baseline and promotion rules

Use **ConvNeXt-Tiny V1, 224 pixels, protocol selection, 16 slices × at most
five series, 12 epochs, original labels, no report/ranking auxiliaries** as the
reproducible control recipe. It has replicated standalone proxy improvements
over uniform selection and avoids the extra 384 cost without a demonstrated
meaningful gain. This is the best-supported economical control, not a claim
that independent image evidence has proved it the best model.

Fixed historical reference: `outputs/ensemble_16_24_protocol/protocol_fold1.pt`,
SHA-256 `64b0d2515b3d214bf8175de3d083dd388794ff1013fb01a93d0a7c819022664c`.
Fold 1 is an existing reference, not selected as the luckiest fold. Its proxy
macro is 0.881118; **0.914 is the ensemble's score, not this model's**. The
historically best recorded single-fold public result is 24-slice fold1 0.891;
its +0.002 over 16 slices and cross-fold variance do not prove superiority to
the protocol recipe. Keep that record, but do not select a new fold by public
score. Train a fresh matched control after establishing new exclusions.

Predeclare before any new screening:

- Primary metric: all-twelve macro image-reference ROC AUC, same studies and
  label masks for both arms. Mined-label AUC is secondary. Report all twelve
  individual AUCs, counts and paired deltas, including direction disagreements.
- Checkpoint choice: best macro on the 120-study development-selection set
  within the fixed 12-epoch budget, earliest epoch for an exact tie. Gold and
  screening/final labels are not callback inputs. No post hoc EMA/SWA search.
  Require both resolved classes for every finding before this selection rule
  can be used; enlarge development first if necessary, never silently select
  on an eleven-finding macro. If adequate development annotation is impossible,
  the report-only fallback does not earn an image-performance claim.
  Small selection data is acknowledged; the independent screen/final sets
  handle its optimism. For nested CV, epoch choice occurs in inner folds only.
- Screen one isolated candidate on the separate development-screen set. A
  point macro delta >=+0.005 and no finding point delta below −0.02 merits
  replication, **not confirmation**. For a targeted supervision fix, credible
  +0.03 on its target with macro >=0 can merit one replication even if mined
  agreement falls; final macro requirements remain unchanged. All such
  screens are exploratory and their uncertainty must be shown.
- Only surviving candidates receive two more paired seeds (0/1/2 total),
  identical data, initialization per pair, steps, augmentation and hardware.
  Report each singleton's metrics, mean paired delta and range/SD. Three seeds
  are a limited stability check, not a precise variance estimate. At least
  two seeds must improve and the mean must meet the same screen. Never average
  their predictions to pass it; do not select the luckiest seed. Final seed 0
  is fixed. CV supports evaluation; deployment remains one checkpoint.
- Freeze one winning recipe, its control, prediction hashes and analysis
  before a **single** locked final readout. Confirmation requires standalone
  macro delta >=+0.005 and paired 95% lower bound >0. Across all twelve
  findings use prespecified simultaneous one-sided 95% lower bounds for a
  −0.02 noninferiority margin (e.g. Bonferroni or validated max-stat bootstrap).
  A point estimate above −0.02 alone does not pass a safeguard. An interval
  crossing the margin means regression is unresolved, not proven safe.
- The final set may be too small for the per-finding safeguard: report that,
  do not waive it after seeing scores. Report positive/negative and effective
  counts; no endpoint dropping, new thresholds, or subset cherry-picking.
  Extra data or a revised margin requires a new prospective protocol.
- Stop a branch on failed evidence gates, inconclusive affordable precision,
  material instability/regression, or poor correction validity. Stop the whole
  queue after the two interventions below. Opening final results retires that
  set for further tuning. No leaderboard-led loops and no blend promotion.

## 6. Ranked, bounded experiments (not launched)

**1. PF OA report-grounding repair, restricted to explicit missed mentions.**

Mechanism: the random-control audit found genuine missed cartilage mentions;
the broad rule mixed these with a disputed severity threshold and achieved
only ~57% design precision. Intervention: replace only PF OA training targets
with adjudicated, explicitly supported positive findings that the existing
extractor missed. Freeze a narrow vocabulary/assertion rule using existing
development reviews; retain original weights, other targets and uncertain
cases. No automatic grade-II promotion and no new image claims from text.
Matched control: original labels; otherwise the exact baseline recipe.

Before GPU work, require a fresh blinded text check (outside rule-design
data; existing reserve only after rule freeze), including proposed changes
and random negative/ambiguous controls. Seek at least 100 new proposed-change
reviews; a 95% Wilson lower bound >=0.90 on correction validity is a useful
high-precision gate (about 97/100 correct clears it). Estimate missed mentions
separately from random controls. If there are fewer proposals, review all and
report the achievable interval; do not expand the rule to make n. A qualified
rubric decision is necessary where clinical thresholds remain in dispute.

Cost: 100–200 fresh text pairs, two independent 1–3 minute reads where available
(~3–20 reader hours plus adjudication). Existing report audit is design data,
not paid again. One 12-epoch pair ~5 elapsed hours on 2×T4, **~10 T4 GPU-hours**,
based on prior ~2.5-hour arms; budget ~30 T4 GPU-hours total if it earns two
seed repeats. Benchmark the actual new training population; A100 times are
not measured for this exact proposed pair. Independent image evaluation cost
is in section 4.

Success: text precision gate plus standalone independent-image screening rule
above, with PF OA target gain >=0.03 as a targeted secondary screen. Reduced
old-label agreement does not veto it. Confirmation uses the new locked
all-twelve image evaluation, not the text reserve, not the development cases,
and not a blend. Stop if report proposals remain unreliable, definitions
unresolved, image gains fail, or seed gains disappear. This is not revival of
the eleven-cell patch or the unreviewed 648-proposal rule.

**2. Replace report supervision with image review on a bounded training subset.**

Mechanism: historical gold/report comparison shows omissions that extraction
cannot fix, particularly effusion/synovitis/fracture; the new pilot must check
this mechanism beyond gold. Intervention: for **200 separately sampled
training studies**, replace the twelve report targets with qualified
adjudicated image labels, masking unresolved cells; keep the original weights
on resolved cells. This is one supervision-source intervention. Include both
report-silent and explicit findings with declared sampling, but do not use this
enriched training sample for representative evaluation. Preserve all other
training labels and settings. Matched control sees those same images with
original report targets, same initial weights, schedule, sampling and budget.

Start only if the development image pilot verifies a substantial correction
burden and the 200 studies can supply resolved positive and negative training
examples for the affected findings. This avoids a costly eleven-cell-scale
perturbation. Do not initialize either arm from a historical checkpoint trained
on new evaluation members. Do not pool this intervention with experiment 1
in its screen.

Cost: 200 additional image reviews: ~56.7 expert hours / 68 including overhead,
approximately $13,600 at the illustrative rate; same ~10 T4 GPU-hours per pair,
up to ~30 if it earns replication. Screening success is standalone macro
>=+0.005 with the safeguards above, followed by paired seeds and the untouched
final readout. Stop if few reviewed labels differ meaningfully, missingness
prevents credible supervision, or gains do not survive screening/replication.
An inconclusive small effect is not grounds for thousands of new labels by
default. If experiment 1 already uses the final set, experiment 2 needs a new
final evaluation cohort; preferably choose between both using development
only and open the one final set once.

No third intervention is justified yet. Orientation, coverage and scanner
diagnostics should localize an actual input mechanism before a new image
pipeline; saved training curves do not justify a generic optimization sweep.
Representation limits remain possible but not isolated by current evidence.

## 7. Exact next action and prepared materials

**Have a lead qualified MSK reader and an independent second reader review the
first 40 cases (DEV0001–DEV0040) in the prepared development pilot, all twelve
findings, with full raw series and the blinded forms; measure time and adjudicate
disagreements.** Before the first scored read, settle the definition/uncertainty
rubric using separate calibration examples. Once image decisions are committed,
the custodian compares them with reports and verified held-out single-model
predictions. This directly estimates report omission, extraction error, image
assessability and annotation feasibility. It buys more information than another
backbone or seed campaign.

Prepared now in `outputs/validation_reset_20260924/`:

- `pilot_custodian_key.csv`: 120 reproducibly selected studies, inclusion
  probabilities and exposure limitations. Custodian only.
- `pilot_series_custodian_manifest.csv`: all original series IDs and descriptors
  for packaging complete exams; no selection down to model inputs.
- `pilot_reader_a_blank.csv`, `pilot_reader_b_blank.csv`: separate shuffled
  forms, 1,440 rows each, no reports, labels, model scores or sampling strata.
- `adjudication_blank.csv`, `README.md`: workflow and required provenance.
- `audit.json`, `checkpoint_checks.json`, duplicate/linkage leads, domain
  tables, label diagnostics, and precision scenarios.

Full raw exams and qualified readers are **not available locally**; the roster
and forms are ready, not an already annotated/viewer-ready image package.
The prior solo audit records no expert availability. If that remains true,
this is the limiting dependency: image-level model ranking remains unsupported.
The bounded report-only fallback can improve extraction evidence but must not
be represented as solving that dependency.

## Implementation status after user reply

The user confirmed no expert readers are available and authorized implementation.
The available fallback is now implemented and tested; see
[the implementation record](validation_fallback_implementation.md) for frozen
splits, executable commands, failed proposal gate and remaining limits. The
image-review steps described above remain conditional, not completed. No training started.
