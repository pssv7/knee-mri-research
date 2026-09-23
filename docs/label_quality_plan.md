# Label quality and independent validation plan

Proposed 2026-09-22. Scope: prioritize lateral meniscus, contusion, and PF OA
label errors, then measure whether better supervision improves image predictions.
This document proposes work; it does not launch review, relabeling, or training.

## Why this is the next priority

The existing 58 image-labeled studies have been inspected repeatedly. Keep them
as historical diagnostics, excluded from training; they cannot provide a fresh
confirmation of small improvements. A larger set needs independent image review,
adequate positive and negative counts, and protection from model selection.

The [initial audit](label_review_v1.md) screened 250 study–finding pairs but
checked only 23 full reports; 227 remain pending. Its targeted findings establish
extraction problems, not a population error rate. The
[eleven-cell correction experiment](reviewed_labels_candidate.md) produced
fold-1 macro delta -0.003086 with a paired interval crossing zero. That neither
validates nor rules out broader label repair.

## 1. Freeze definitions and evaluation membership first

- Version the original labels, weights, report hashes, split manifest, and
  reference predictions. Preserve the existing fold assignments for historical
  comparisons; create a separate manifest for the new evaluation protocol.
- Have qualified musculoskeletal readers reconcile the project's definitions
  in `src/labels/llm_extract.py` with the competition annotation criteria before
  review. Record severity thresholds and how borderline cases are handled.
- Separate three questions in the ledger: what the report states, whether the
  extractor represents it correctly, and what the images establish. An accurate
  extraction can still disagree with image truth.
- Assign studies to audit/development or locked evaluation before inspecting
  their findings. Exclude previously reviewed cases from the new locked set.
  Group repeat examinations by patient where identifiers permit; check duplicate
  images and record any limits on patient linkage.

## 2. Audit the three priority findings

Use development folds 0/2/4 for report-error discovery, excluding all newly
reserved validation studies. Finish the existing priority cases first, then
complete the other pending full-report checks as a secondary workstream.

For each priority finding, start with 150 unique development study–finding pairs:
50 suspected false positives, 50 suspected missed positives, and 50 random
controls. This is 450 pairs, potentially fewer unique studies. Count eligible
existing reviews toward the target and deduplicate. If a stratum is smaller,
review all available cases and record the shortfall. Select controls randomly
from the full eligible population, preserving inclusion probabilities and
overlap flags; keep disagreement-enriched results separate from control estimates.

Use held-out model disagreement only to select cases for review. Include report
language, site/vendor where known, target weight, and extraction mode in the
sampling summary. Cover uncertain and unmentioned findings as well as explicit
positives and negatives. The current review HTML exposes model scores and targets;
prepare a blinded reader view before using it for independent review.

| Priority | Questions to resolve under the agreed annotation rubric |
| --- | --- |
| Lateral Meniscus | True tear versus degeneration/intrasubstance signal; medial versus lateral attribution; negation; missed tear phrases across languages; postoperative or uncertain findings. |
| Contusion | Bone versus soft-tissue/tendon edema; traumatic marrow findings versus purely degenerative subchondral change; uncertain cause; coexisting fracture without assuming labels are mutually exclusive. |
| PF OA | Patellar/trochlear cartilage disease versus tibiofemoral OA; lateral patellar disease incorrectly assigned to Lateral OA; mild/borderline cartilage changes; unspecified OA compartment. |

Record exact source evidence, anatomical location, assertion/negation,
certainty, severity, old target/weight, proposed disposition, and reviewer IDs.
Classify errors as anatomy, negation, missed mention, severity/definition,
fusion/weighting, report–image disagreement, or unresolved. Review final fused
targets as well as intermediate extraction: fixing an intermediate call need
not require changing an already-correct final target. A soft target below 0.5
alone is not evidence of an error.

Have a second reviewer independently assess proposed corrections and random
controls before adjudication. Use original-language expertise where needed.
Report error counts and uncertainty separately by finding, direction, and
sampling stratum. Fifty controls per finding are an initial audit, not a precise
estimate of rare errors; enlarge controls if their intervals are too wide.

Deliverable: a reviewed error ledger and ranked, evidence-supported repair
categories. Report omissions and unresolved image questions stay distinct from
fixable extraction errors. Silence must not become a confident negative.

## 3. Build a substantially larger image-reviewed set

Planning target: **1,000 additional unique studies**, initially allocated to
300 development-validation studies and 700 locked confirmation studies.
The first 100 development studies form a reader-calibration and feasibility
pilot. The 58 historical studies are separate. These counts are a budget target,
not a claim that 700 cases can distinguish a 0.003–0.005 AUC gain.

- Sample a representative core across available sites, vendors, report languages,
  and acquisition protocols, with a fixed random seed and sampling manifest.
  Keep audit-selected hard cases outside the representative locked core.
- Review all twelve findings so the set supports the competition macro metric
  and detects regressions outside the three priorities. Expedite the three
  priority findings in the pilot to establish their prevalence and disagreement.
- Seek at least 100 adjudicated positives and 100 negatives per priority finding
  in locked evaluation as an initial feasibility check, not a power guarantee.
  Expand the representative sample if necessary. Any separately enriched rare-
  finding panel must be reported separately; do not pool it unweighted into the
  main score. AUC can still be distorted by selecting unusually easy/hard cases.
- Two qualified musculoskeletal readers independently review full diagnostic
  image series, blinded to reports, mined targets, model predictions, and each
  other's decisions. A third reader adjudicates disagreements and uncertainty.
  Report review happens only after the image reference has been committed.
- Capture present/absent/indeterminate, severity, location, assessability, and
  supporting series/image references. Preserve both original reads and the
  adjudication. Resolve disagreement through review rather than averaging labels.
- Predefine handling of unresolved or non-assessable study–finding pairs. Report
  exclusions and sensitivity analyses; never silently convert them to negatives.
  Summarize reader agreement and adjudication rates by finding and site.

Use appropriately authorized participating readers and permitted data access;
confirm applicable access requirements before distributing review packets.
Reader availability, review time, and adjudication capacity are execution
dependencies. Estimate the budget from measured pilot minutes per study:
two reads per study plus observed adjudication time, preparation, and QA.

### Prevent training and selection leakage

Newly reviewed training studies are not automatically held out from existing
models. The current five-fold ensemble includes members trained on those studies.
Do not score that full ensemble on them and call it independent validation.

For the new protocol, exclude both new validation partitions, all patient-linked
duplicates, and the old 58 from training of every reference/candidate member.
Train matched reference and candidate models on the same remaining population.
Development validation may guide selection; locked labels remain with the
evaluation custodian and are excluded from prompts, calibration, correction
rules, checkpoint selection, and blend fitting. Extend the current gold-only
leakage assertion to check the complete versioned exclusion manifest.

Verified OOF predictions can provide interim diagnostics on newly reviewed
studies only when each contributing checkpoint excluded that study/patient.
Historical model selection means those diagnostics are not a pristine test.
If no previously untouched studies are available, disclose that limitation;
relabeling cannot erase previous exposure.

## 4. Determine what improvement the sample can resolve

Before locking the final sample size and evaluating a candidate, use development
review data and matched held-out predictions to plan paired AUC comparisons.
Proposed effects worth detecting: +0.005 twelve-finding macro AUC and +0.02 on
each priority finding. Also estimate the sample needed for +0.003 macro AUC.

Simulate power across plausible prevalence, model-score correlation, reader
uncertainty, and between-patient variation; target 80% power with two-sided
5% significance for the primary macro comparison. Plan multiplicity control
for the three secondary finding-specific claims. Bootstrap patients jointly
across models and findings, respecting the sampling design. Document the method,
assumptions, positive/negative counts, and resulting minimum detectable effects.

Use the pilot estimates conservatively: 100 cases cannot precisely establish
all these quantities. Increase enrollment before opening locked results if the
projected precision is inadequate. Small gains may require thousands of studies.
If that is infeasible, explicitly limit the claim to the effects the available
sample can resolve. Do not repeatedly add cases after inspecting significance.

## 5. Test a label-repair candidate only after the audit

1. Develop narrowly scoped extraction/weighting changes on reviewed development
   data. Require exact evidence, correct anatomy, and explicit uncertainty.
   Keep evidence-backed corrections separate from conjectural image labels.
2. Validate extraction on a fresh, independently reviewed development sample
   unused in rule/prompt design. Include language-specific negative, positive,
   and ambiguous examples; verify that fixing false positives does not increase
   missed positives. Freeze the candidate before confirmation.
3. Produce a separate versioned label table and correction ledger with source
   hashes, old/new targets and weights, reviewer decisions, and rationale. Apply
   corrections only to training membership; leave evaluation reference labels
   independent of the candidate. Compare extraction quality and report coverage
   against the image-reviewed development set.
4. Run one matched supervision experiment: same architecture, input resolution,
   training population, initialization/seeds, compute, and selection procedure;
   only training supervision changes. Use development validation for screening
   and matched seeds to assess training variability before the final comparison.
5. Register the final reference/candidate, prediction hashes, sample size,
   endpoints, and analysis before the custodian opens locked evaluation. Report
   paired macro/per-finding AUC deltas and 95% intervals, plus missingness and
   site/language diagnostics. Keep mined-label scores as secondary diagnostics.

Proposed confirmation gate: macro delta at least +0.005 with its paired 95%
interval above zero, plus no priority finding showing a material regression.
Predeclare a -0.02 per-finding noninferiority margin and assess uncertainty;
an interval crossing that margin leaves safety against regression unresolved.
Report all twelve findings. A priority-finding improvement requires its own
multiplicity-adjusted evidence. An inconclusive result remains inconclusive.
Further tuning after opening the locked set requires new confirmation data.

## Delivery order and completion criteria

1. **Protocol and manifests:** rubric, reader workflow, blinded packets, sampling,
   partition membership, and expanded leakage checks ready for review.
2. **Audit and 100-study pilot:** prioritize the three findings; measure correction
   mechanisms, positive counts, reader disagreement, and annotation cost.
3. **Expanded review:** complete development and locked image review to the
   power-informed sample size, with adjudication and a versioned reference set.
4. **Candidate and comparison:** freeze reviewed corrections, train the matched
   pair, and issue the locked comparison with uncertainty and provenance.

Completion means an independently reviewed, auditable evaluation set and a
credible estimate of the effect of label repair. It does not require claiming
a gain when the data cannot establish one. The immediate next action is to
prepare the rubric, membership manifest, and blinded priority-review packets.
