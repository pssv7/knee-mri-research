# Implementation plan: label quality and independent validation

Source: `docs/label_quality_plan.md` (proposed 2026-09-22, untracked). This plan
implements that document **as written**. Every requirement in it is mapped to a
step in the traceability table at the end of `## Steps`. Where the source is
silent, this plan makes a choice and labels it *implementation choice*. Where the
source cannot be executed exactly as written, the item is in `## Risks and open
questions` for the user to decide; none is silently substituted.

## Goal

Done means all four delivery stages of the source plan are complete, in order,
each with its completion evidence committed:

1. **Protocol and manifests.** The annotation rubric, reader workflow, blinded
   packets, sampling manifests, partition membership and the expanded leakage
   checks exist, are versioned, and are ready for review.
2. **Audit and 100-study pilot.** A reviewed error ledger for Lateral Meniscus,
   Contusion and PF OA (150 pairs each where available) has second-reviewer
   assessment and adjudication. The 100-study image pilot has measured reader
   minutes, agreement, positive counts and correction mechanisms.
3. **Expanded review.** Development-validation and locked image review are done
   to the power-informed sample size, adjudicated, and frozen as a versioned
   reference set held by the evaluation custodian.
4. **Candidate and comparison.** A frozen, evidence-backed label-repair
   candidate and a matched reference/candidate training pair have been run. The
   pair was registered before the locked set was opened. The locked comparison
   is issued with paired macro and per-finding AUC deltas, 95% intervals,
   missingness, site and language diagnostics, and provenance.

"Done" does not require a gain. An inconclusive result is reported as
inconclusive. The first deliverable is the immediate action the source names:
the rubric, the membership manifest, and the blinded priority-review packets.

Nothing in this plan launches training, uploads data, or changes
`data/meta/train_labels.parquet`. Each GPU run and each distribution of review
packets is a separate, user-authorized step.

## Context

**Data (verified 2026-09-22 against the files):**
- `data/meta/train.csv`: 4,407 studies with `StudyInstanceUID` and `Report`, plus
  twelve label columns. Those columns are filled only for the 58 gold studies,
  which `src/model/splits.load_gold_uids` identifies.
- `data/meta/train_labels.parquet`: 4,349 studies × (12 targets + 12 `_weight`).
  Zero gold studies are in it, so gold is already structurally out of training.
- `data/meta/gold_truth_58.parquet` / `.csv`: gold reference.
- `data/meta/llm_labels.parquet` (categorical mode, with `<finding>_evidence`),
  `data/meta/llm_probs.parquet` (probability mode). Fusion: `src/labels/fuse.py`
  and `src/labels/mode_fusion.py` (`DEFAULT_STRATEGY = "score_average"`).
  Lexicon: `src/labels/lexicon.py`.
- `outputs/ensemble_16_24/reconstructed_folds.parquet`: the historical five-fold
  assignment (871/871/870/869/868). This is the fold file both
  `tools/build_label_review.py` and `tools/prepare_reviewed_labels.py` read.
- `data/meta/resolution_index/index_train.parquet`: 819,078 instance rows with
  `Manufacturer` (12 values), `ManufacturerModelName` (46),
  `MagneticFieldStrength` (4), `SeriesDescription` and `ProtocolName`. It does
  **not** contain `PatientID`. The raw DICOM headers do: `data/dev/*.dcm` carries
  a pseudonymised `PatientID` (e.g. `'857c0f7e-c11'`) and `PatientSex`, while
  `InstitutionName`, `StationName`, `StudyDate` and `AccessionNumber` are
  blank. So patient linkage needs a header rescan, and site can only be
  approximated by scanner identity (`src/model/splits.scanner_key`).
- `data/meta/train_series.csv`: `Fluid_Sensitive`, `Fat_Suppression`,
  `Anatomical_Plane` per series (the acquisition-protocol descriptors).

**Label code:**
- `src/labels/llm_extract.py`: `FINDING_DEFINITIONS` holds the project's
  definitions (e.g. Lateral Meniscus "Degeneration without a tear does not
  count"; Contusion excludes "purely degenerative subchondral change"; PF OA is
  "patella or trochlea"). These are what readers reconcile against the
  competition criteria. `docs/competition_pages.md` §Dataset Description holds
  the only competition definitions in the repo, and they are one line per
  finding with no severity thresholds.
- `tools/audit_labels.py`: `detect_language` (stem-based, no dependency), the
  reproduction check (104,376 cells), and the gold diagnostic.
- `src/labels/reviewed.py::apply_reviewed`: applies corrections, but hard-codes
  `heldout_fold in (1, 3)`, corrections only in folds 0/2/4, and
  `new_target == 0.0`. It cannot express weight changes, non-zero targets, or
  the new partition manifest.
- `tools/prepare_reviewed_labels.py`: the 11-correction candidate and the AST
  hook (`stage_source`) that injects corrections into the training source after
  `folded` is built, so stratification is unchanged. It is tested in
  `tests/test_reviewed_labels.py` and `tests/test_reviewed_recovery.py`.

**Existing review:**
- `tools/build_label_review.py`: builds `outputs/label_review_v1/`
  (`review_queue.jsonl`, `summary.json`, `review.html`) from folds 0/2/4. It
  selects 200 pairs where both the legacy blend and protocol OOF oppose the
  target at ≥0.7, plus 50 random controls (seed 20260915). Notes are in
  `tools/label_review_v1_notes.json` (23 full-report reviews; 227 pending).
  `review.html` shows target, weight, legacy and protocol scores on every card,
  so it is **not blinded**.
- `docs/label_review_v1.md` and `docs/reviewed_labels_candidate.md`: the audit
  and the fold-1 result (−0.003086, CI [−0.010543, +0.004496]).

**Splits and leakage:**
- `src/model/splits.py`: `load_gold_uids`, `assert_no_gold_leakage`,
  `make_folds`, `scanner_key`. `make_folds` stratifies on positive count, so any
  label change re-cuts folds unless the split is fixed first (the reason
  `stage_source` hooks in after `folded`).
- `assert_no_gold_leakage` is called from the training source generated by
  `tools/make_notebook.py` (lines ~1827 and ~3043), from
  `tools/kaggle_benchmark_body.py:49`, `tools/protocol_benchmark_body.py:43`,
  and `tools/package_16_24_ensemble.py:31`. It is tested in `tests/test_splits.py`.
- Checkpoint sidecars (`convnext_fold{N}.pt.json`) declare `train_uids` and
  `val_uids`; `tools/build_label_review.py` already checks them.

**Statistics:**
- `tools/compare_oof.py::compare`: a paired, fold-blocked, study-level bootstrap
  of macro delta (1,000 draws). It is not patient-clustered and has no
  sampling-design weights.
- `src/model/conditional.py`: `auc`, `auc_se` (Hanley–McNeil),
  `paired_bootstrap`.

**Training generators:** `tools/prepare_kaggle_training_pair.py::training_source`
(the matched-pair generator used by the reviewed-label and resolution pairs) and
`tools/make_notebook.py`. Reference OOF receipts are in
`data/meta/compare_protocol_incumbent_all_folds/summary.json` (protocol OOF
sha256 per fold) and `outputs/resolution384_downloads/oof_all_folds.parquet`.

**Constraints that bind this plan:**
- `docs/rules_notes.md`: competition data must not be provided "to any party not
  participating in the Competition", and it is under the MIRA license. Readers
  must therefore be competition participants, which in practice means team
  members. The team-merger deadline is **2026-10-15** and the final submission
  is **2026-10-22**.
- Memory and `docs/agent_playbook.md`: gold selects nothing, and a verification
  must say whether it is reproduction or correctness and give the interval.
- No file, commit or notebook may name an AI assistant.

## Approach

**One versioned manifest is the spine.** Every later artifact (audit samples,
image packets, training exclusions, evaluation opening) references a manifest
by sha256 and refuses to run if the hash differs. The manifest records the
partition of every one of the 4,407 study UIDs: `historical_gold`,
`dev_validation`, `pilot` (the first 100 of `dev_validation`), `locked`, `audit`
and `training_pool`. It also records patient-link groups and duplicate-image
groups, the sampling seed, inclusion probabilities, and the hashes of the frozen
inputs. Assignment happens before any finding in those studies is inspected. The
assignment code reads only non-finding metadata (scanner, language, protocol
descriptors, prior-review flags), and a test enforces that.

**Three questions, three columns.** The ledger stores `report_states`,
`extractor_correct` and `image_establishes` as separate fields. It stores both
the intermediate extractor call and the final fused target, and a disposition
applies to the final target only when that target is wrong.

**Blinding is structural, not procedural.** Packet builders take an explicit
allow-list of fields, and tests assert that the forbidden fields are absent from
every emitted file. For report packets the forbidden fields are scores,
targets, weights, extractor calls and sampling stratum. For image packets they
are the report, targets, predictions and other readers' reads. Report review of
an image-reviewed study is refused until that study's image reference is
committed (its hash is in the commitment log).

**Locked labels never enter the repo.** The repo holds only the hash commitment
of the locked reference set. The labels themselves live with the custodian. One
tool, `tools/open_locked_evaluation.py`, is the only code path that reads them.
It refuses unless a registration file exists whose hash was committed before
opening, and it records the opening.

**Statistics.** A new patient-clustered, design-aware paired bootstrap is used
for every locked-set claim and for the power simulation, so the two agree. The
existing `compare_oof.compare` stays unchanged for historical comparisons.

**Alternatives rejected:**
- *Modifying `reconstructed_folds.parquet` or re-cutting folds.* The source says
  to preserve historical folds and create a separate manifest.
- *Top-K deterministic selection for the disagreement strata (as v1 did).*
  Rejected in favour of a fixed opposition threshold that defines the stratum,
  then random sampling within it, with recorded inclusion probabilities. The
  source asks that inclusion probabilities be preserved, and that stratum error
  rates be estimable. *Implementation choice.*
- *Scoring the existing five-fold ensemble on newly reviewed studies.* Forbidden
  by the source: those members trained on them.
- *Extending `apply_reviewed` in place.* Its fold-1/3 contract is tested and
  backs the recovered fold-1 result. A new `apply_corrections` is added
  alongside it instead.
- *Averaging reader labels.* Forbidden by the source; disagreements are
  adjudicated.
- *Using a hosted API for any extraction or review aid.* Forbidden by the rules.

## Steps

Each step names its files and its verification. Steps 1–10 are Stage 1, which
is code only and needs no GPU. Steps 11–15 are Stage 2, 16–19 Stage 3, and 20–25
Stage 4. A step marked **[human]** is blocked on readers or the user, not on
code.

### Stage 1: Protocol and manifests

**1. Freeze the historical baseline.**
- *Files:* create `tools/freeze_label_baseline.py` →
  `outputs/label_quality/baseline_v1.json`.
- *Content:* the sha256 of `data/meta/train_labels.parquet` (targets and weights),
  a per-study sha256 of each `Report` (4,407 rows), and the sha256 of
  `outputs/ensemble_16_24/reconstructed_folds.parquet` (the historical split
  manifest). It also records the reference predictions: each protocol OOF
  listed in `data/meta/compare_protocol_incumbent_all_folds/summary.json`,
  `.../incumbent_oof.parquet`, `outputs/resolution384_downloads/oof_all_folds.parquet`,
  the `llm_labels`/`llm_probs` dumps, and `gold_truth_58.parquet`.
- *Verify:* the tool recomputes every hash and fails on a mismatch against the
  receipts that already exist (the protocol OOF sha256 in `summary.json`).
  Rerunning it gives a byte-identical JSON. This is a reproduction check, not a
  correctness check, and the doc says so.

**2. Scan for patient identifiers and duplicate images.**
- *Files:* create `tools/scan_patient_links.py`, a Kaggle CPU notebook body,
  internet off. For each study it reads one header per series with
  `stop_before_pixels` and records `PatientID`, `PatientSex`, `PatientAge`,
  `StudyDate` and `AccessionNumber`, keeping blanks as blanks. It also hashes
  the cached uint8 volumes: an exact sha256 per series, plus a 16×16×4
  downsampled perceptual hash for near-duplicates.
- *Output:* `outputs/label_quality/patient_links.parquet`,
  `duplicate_groups.parquet`, and `linkage_limits.json`. The limits file records
  the share of studies with a blank or constant `PatientID`, and whether IDs are
  pseudonymised per study (singletons only), which would make linkage impossible.
- *Verify:* first run locally on `data/dev/*.dcm`. A unit test gives two
  synthetic series with identical pixels different UIDs, and asserts that one
  duplicate group comes out. **[human]** The user launches the Kaggle CPU run.
- If patient linkage is impossible, record that in `linkage_limits.json`.
  Duplicate-image grouping then becomes the only linkage, and the final report
  says so, as the source requires ("record any limits on patient linkage").

**3. Build the membership manifest.**
- *Files:* create `src/labels/manifest.py` (pure functions) and
  `tools/build_membership_manifest.py` → `outputs/label_quality/manifest_v1.parquet`
  plus `manifest_v1.json` (seed, counts, hashes).
- *Inputs:* the output of step 1, the linkage from step 2, `scanner_key`,
  `detect_language` (it reads the report text for language only),
  `train_series.csv` protocol descriptors, and the prior-review flags (every
  study in `outputs/label_review_v1/review_queue.jsonl` plus the gold 58).
- *Rules:*
  - The 58 gold studies go to `historical_gold`, excluded from training and from
    every new set.
  - The representative core is a stratified random sample over scanner group ×
    language × plane-coverage, with a fixed seed (20260922) and inclusion
    probability recorded per study. Its size is 1,000: 300 `dev_validation`
    (whose first 100 by sampling order form the `pilot`) and 700 `locked`.
  - `locked` excludes previously reviewed studies and gold, and is later
    protected from audit-selected hard cases (see step 5).
  - All studies in one patient-link group or duplicate group share a partition.
    Conflicts are resolved by moving the whole group to the most restrictive
    partition: locked > dev_validation > audit > training_pool.
  - The remaining folds-0/2/4 studies become `audit`. All other studies become
    `training_pool`. The historical `fold` is stored alongside for historical
    comparisons, unchanged.
- *Verify:* tests confirm that the builder's call graph never reads label or
  score columns (it gets a frame with those columns removed and must produce an
  identical manifest). They also check group integrity, locked ∩ prior-review
  = ∅, locked ∩ gold = ∅, counts, and seed determinism.

**4. Expand the leakage checks.**
- *Files:* `src/model/splits.py` gets `load_exclusion_manifest(path,
  expected_sha256)` and `assert_no_excluded_leakage(train_uids, manifest)`. The
  latter covers gold, `dev_validation`, `locked`, and every study patient- or
  duplicate-linked to them. Keep `assert_no_gold_leakage` unchanged.
- Wire the new assertion into the generated training source in
  `tools/make_notebook.py` behind an `EXCLUSION_MANIFEST` constant (default
  `None`, so historical notebooks are byte-identical), and into
  `tools/prepare_kaggle_training_pair.py::training_source`.
- Create `tools/check_checkpoint_exclusion.py`. Given checkpoint sidecars and
  the manifest, it lists, for every newly reviewed study, which checkpoints
  excluded it and its linked patients. It is the gate for step 18's interim
  OOF diagnostics.
- *Verify:* extend `tests/test_splits.py` so that a linked-patient leak raises,
  a manifest-hash mismatch raises, and gold still raises. Run
  `tests/test_notebook.py` and `tools/check_notebook.py` to show that existing
  notebooks regenerate unchanged when the manifest is unset.

**5. Write the audit sampler.**
- *Files:* create `src/labels/audit_sampling.py` and
  `tools/build_priority_audit.py` → `outputs/label_quality/audit_v1/`.
- *Rules:* the population is `audit` partition pairs (folds 0/2/4 minus every
  reserved study) for the three priority findings.
  - Suspected FP means target ≥ 0.5 and the held-out protocol OOF ≤ 0.3.
    Suspected missed positive means target < 0.5 and OOF ≥ 0.7. The thresholds
    are fixed now, before sampling. *Implementation choice*: same 0.7 opposition
    as v1, one source (protocol) because it has verified sidecar exclusion.
  - Take 50 from each stratum at random within the stratum, plus 50 random
    controls from the full eligible population for that finding, each with its
    inclusion probability and overlap flags (a control that also meets a
    suspect criterion keeps a flag).
  - Existing reviewed pairs in `label_review_v1` that satisfy a stratum's
    definition and are in `audit` count toward the target, deduplicated by
    (study, finding). If a stratum has fewer than 50 cases, take all of them
    and record the shortfall.
  - Model disagreement selects cases only. No threshold or rule may be fit to it.
- *Sampling summary:* counts by language, scanner/vendor, target weight,
  extraction mode (the categorical call present/absent/None vs the
  probability-mode score), and assertion class. The classes are explicit
  positive, explicit negative, uncertain (categorical `unclear` or
  certainty = possible), and unmentioned.
- *Secondary workstream:* a queue of the 227 pending `label_review_v1` pairs,
  in priority-finding-first order.
- *Verify:* unit tests check stratum definitions, inclusion probabilities
  summing correctly, deduplication, the shortfall record, the exclusion of
  reserved studies, and that disagreement-stratum rows never enter control
  estimates.

**6. Define the error ledger schema.**
- *Files:* create `src/labels/ledger.py` (schema plus validation).
- *Row fields:* `review_id`, `StudyInstanceUID`, `finding`, `sampling_stratum`,
  `inclusion_probability`, `report_sha256`, `evidence_quote` (validated as an
  exact substring of the report), `anatomical_location`, `assertion`
  (present/negated/uncertain/unmentioned), `certainty`, `severity`,
  `report_states`, `extractor_correct`, `image_establishes` (nullable until
  image review), `intermediate_call` (categorical/probability/lexicon values),
  `old_target`, `old_weight`, `final_target_correct`, `proposed_disposition`
  (keep / correct_target / correct_weight / exclude / unresolved),
  `error_class` (anatomy, negation, missed_mention, severity_definition,
  fusion_weighting, report_image_disagreement, unresolved), `reviewer_1_id`,
  `reviewer_2_id`, `reviewer_2_assessment`, `adjudication`, `language_expert_id`,
  and `rationale`.
- *Validation rules:* a disposition that changes a target requires
  `final_target_correct == False`, so a correct final target is never changed to
  repair an intermediate call. `old_target < 0.5` alone is not accepted as
  evidence, and a missing mention cannot yield a confident negative. A
  `report_omission` or `unresolved_image_question` row can never carry a
  target-changing disposition.
- *Verify:* each rule has a failing-input unit test.

**7. Build the blinded reader views.**
- *Files:* create `tools/build_blinded_packets.py`.
  - *Report-review packet* for audit reviewers: a pseudonymous case ID, the full
    report, and the finding question, in randomized order. It excludes model
    scores, target, weight, extractor calls and evidence, and stratum. The
    unblinding key stays in a separate file that is not given to reviewers.
  - *Image-review packet* for readers: the case ID, the StudyInstanceUID for the
    reader's own permitted Kaggle data access, and the series list with plane
    and sequence descriptors. It excludes the report, targets, predictions and
    other reads.
  - *Capture forms* (CSV plus JSON schema): per finding
    present/absent/indeterminate, severity, location, assessability, supporting
    series/image references, reader ID, start/end time, and a free-text note.
- *Verify:* a test scans every emitted file for forbidden column names and for
  the values of scores and targets in the fixture. The existing
  `review.html` is left untouched and labelled unblinded in its title.

**8. Draft the annotation rubric. [human for sign-off]**
- *Files:* create `docs/annotation_rubric.md`. It has a table per finding:
  the competition definition (verbatim from `docs/competition_pages.md`), the
  project definition (verbatim from `FINDING_DEFINITIONS`), a reconciliation
  column, severity threshold, borderline handling, indeterminate criteria, and
  the priority questions from the source's table (Lateral Meniscus, Contusion,
  PF OA rows copied in full).
- Qualified MSK readers fill the reconciliation, thresholds and borderline
  rules and sign it. The rubric is then hashed into the manifest JSON, and every
  packet refers to that hash.
- *Verify:* `tools/build_blinded_packets.py` refuses to emit packets while the
  rubric carries no reader sign-off block or its hash differs from the manifest.

**9. Write the reader workflow and pilot budget template.**
- *Files:* create `docs/reader_workflow.md`. It covers two independent image
  reads blinded to each other; a third-reader adjudication of disagreements and
  indeterminates; the rule that report review follows image commitment; the
  predefined handling of unresolved and non-assessable pairs (excluded from the
  primary analysis, reported, plus sensitivity analyses treating them as
  positive and as negative, never silently negative); the access requirements
  (participant-only, the MIRA license); and the budget formula
  (2 × median read minutes + adjudication rate × adjudication minutes +
  preparation + QA).
- *Verify:* the user reviews it; nothing is distributed before that.

**10. Stage 1 gate.**
- Commit the manifest, rubric draft, workflow, packets for the priority audit,
  and leakage checks. Update `docs/worklog.md` and `PROJECT.md` (active update)
  with the manifest hash and counts.
- *Verify:* the full test suite `.venv/bin/python -m pytest` passes, and
  `tools/audit_labels.py` still reproduces 104,376 cells.

### Stage 2: Audit and 100-study pilot

**11. Priority report audit. [human]**
- Reviewer 1 completes the 450 priority pairs (fewer if shortfalls apply) in
  blinded packets, finishing existing priority cases first, into the ledger.
- Reviewer 2 independently assesses every proposed correction and every random
  control, before adjudication. Original-language expertise is recorded per row.
- *Files:* create `tools/ingest_ledger.py`, which validates against
  `src/labels/ledger.py`, unblinds via the key, and writes
  `outputs/label_quality/audit_v1/ledger.parquet`.

**12. Audit statistics.**
- *Files:* create `tools/audit_report.py` → `docs/label_audit_v2.md`.
- Error counts with Clopper–Pearson 95% intervals by finding × direction ×
  stratum. Controls are design-weighted by inclusion probability; enriched
  strata are reported separately and never pooled into a population rate.
- Include the reviewer agreement rate and a ranked list of repair categories
  with supporting ledger rows. Report omissions and unresolved image questions
  stay in their own section.
- *Enlargement rule, implementation choice:* if a control interval's width
  exceeds 0.20, draw another 50 controls with the same seed stream. Any
  enlargement is recorded.
- Also queue the 227 pending v1 reviews (step 5) as a secondary workstream and
  report them separately.
- *Verify:* a synthetic ledger with known error rates recovers them within the
  intervals (a test that can fail).

**13. Pilot image review. [human]**
- The first 100 `dev_validation` studies, all twelve findings, with the three
  priority findings expedited. Two independent reads plus adjudication.
- *Files:* create `tools/ingest_image_reads.py`, which validates the forms,
  preserves both original reads, and writes adjudication packets for the third
  reader. Create `tools/commit_image_reference.py`, which writes
  `outputs/label_quality/image_reference_pilot_v1.parquet` and appends its sha256
  to `outputs/label_quality/commitments.jsonl`.
- Report review of pilot studies is then allowed (the packet builder checks the
  commitment log).

**14. Pilot measurements.**
- *Files:* create `tools/pilot_report.py` → `docs/pilot_report.md`.
- Include reader minutes per study (median/IQR), adjudication rate and minutes,
  agreement (Cohen's κ and raw) by finding and scanner group, prevalence and
  positive counts per finding, indeterminate and non-assessable rates, and the
  projected budget via the step 9 formula.
- It also covers correction mechanisms from the audit, and misses vs false
  alarms of mined labels against the pilot image reference, split out rather
  than pooled.

**15. Power planning.**
- *Files:* create `src/model/power.py` and `tools/plan_sample_size.py` →
  `docs/sample_size_plan.md`.
- *Simulation:* a latent-score model per finding with prevalence, a
  reference–candidate score correlation, a candidate effect, reader error
  (misclassification from pilot κ), and patient clustering (cluster sizes from
  step 2). The parameter ranges come from pilot estimates plus matched held-out
  OOF, used conservatively (the pessimistic end of each interval).
- *Estimand:* the paired macro AUC delta. The test is two-sided at 5% with
  80% power, using the patient-clustered, stratum-respecting bootstrap of
  step 16.
- Output the n needed and the minimum detectable effect (MDE) for macro +0.005,
  macro +0.003, and +0.02 per priority finding. The three secondary claims get
  Holm correction. Also report the positive and negative counts per finding at
  each n.
- *Verify:* under a simulated null the rejection rate is 5% ± MC error, and a
  known effect recovers its analytic power (tests that can fail).

### Stage 3: Expanded review

**16. Patient-clustered paired bootstrap.**
- *Files:* create `src/model/paired_eval.py`: `paired_auc_delta(truth, a, b,
  clusters, strata, weights, draws, seed)`. It resamples patient clusters
  jointly across both models and all findings, within sampling strata, and
  applies inclusion weights. It returns macro and per-finding deltas with 95%
  percentile intervals and the positive/negative counts. It excludes
  unresolved pairs per finding and reports them.
- *Verify:* with singleton clusters, one stratum and unit weights it agrees with
  `tools/compare_oof.compare` within MC error on a fixture. Clustering widens
  intervals on a fixture with duplicated patients.

**17. Size lock. [human decision]**
- The user picks the final locked n from `docs/sample_size_plan.md`. If 700 is
  inadequate, enrollment is increased before any locked result is opened. If
  the needed n is infeasible, the claim is limited to the resolvable MDE, and
  that is written into the registration.
- `manifest_v2` is issued if n changes. It extends the same seed stream so the
  original 700 are a prefix, and the change happens only before opening. The
  manifest JSON records `sequential_additions_after_opening: forbidden`.

**18. Full image review. [human]**
- The remaining 200 `dev_validation` studies plus all `locked` studies go
  through the step 13 pipeline. Locked forms go to the custodian, and only the
  hash of the locked reference goes to `commitments.jsonl`.
- Reader agreement and adjudication rates are reported by finding and scanner
  group for dev; for locked they are computed by the custodian and released
  only as aggregates.
- Any enriched rare-finding panel gets its own partition in the manifest and is
  reported separately, never pooled unweighted.
- *Interim diagnostics:* `tools/check_checkpoint_exclusion.py` decides which
  dev-validation studies may be scored with existing OOF. Results are labelled
  "not a pristine test (historical model selection)".

**19. Report review on the image-reviewed dev set.**
- Only after image commitment. It fills `report_states` and
  `extractor_correct`, so extraction quality is measured against image truth.

### Stage 4: Candidate and comparison

**20. Develop the candidate on reviewed dev data.**
- *Files:* create `src/labels/repair.py`, rules scoped to the ledger's ranked
  categories. Each rule requires an exact quote, correct anatomy and explicit
  assertion, and emits an uncertainty flag. Rules run over the existing
  extractor dumps. If a targeted re-extraction is needed, it uses a new
  evidence-grounded prompt variant in `src/labels/llm_extract.py` run locally
  (Colab) like `notebooks/03_extract_labels.ipynb`, never a hosted API.
- Evidence-backed corrections are kept separate from conjectural image-derived
  labels (a `basis` field: report_evidence vs image_only).
- Split the reviewed dev data by the manifest seed before design into
  `design` and `extraction_validation`. The rules see only `design`.

**21. Validate the extraction.**
- *Files:* create `tools/validate_repair.py`. It runs on
  `extraction_validation`, which is independently reviewed and unused in design.
- Report per language: negatives, positives, ambiguous, misses vs false alarms
  before and after, and fail if missed positives rise while FPs fall.
- Compare coverage and extraction quality against the image-reviewed dev set.
- Freeze: `outputs/label_quality/candidate_v1/` with the rule source hash.

**22. Versioned label table and correction ledger.**
- *Files:* create `src/labels/corrections.py::apply_corrections(labels,
  manifest, corrections)`. It allows target and weight changes, only for
  `training_pool`/`audit` studies, verifies report hashes, quotes and old
  values, and asserts that the evaluation reference labels are untouched.
  `apply_reviewed` stays as is.
- Create `tools/build_candidate_labels.py` →
  `outputs/label_quality/candidate_v1/train_labels_candidate.parquet` and
  `corrections.json`. Corrections record source hashes, old/new targets and
  weights, reviewer decisions and rationale.

**23. Matched supervision pair.**
- *Files:* create `tools/prepare_label_quality_pair.py`, reusing
  `training_source` and the `stage_source` hook pattern from
  `tools/prepare_reviewed_labels.py`. Both arms are built from the same code.
  The historical `folded` is built first, then `assert_no_excluded_leakage`
  runs, then the excluded studies are dropped from training (gold,
  dev_validation, locked, linked). In the candidate arm, corrections are then
  applied to training rows only.
- Everything else is identical: the architecture (protocol ConvNeXt, 224 px,
  16 slices × 5 series), the initialization and seeds, epochs, the selection
  procedure, and compute.
- Each arm writes a contract JSON with the manifest hash, label hash,
  correction hash and exclusion counts.
- *Screening:* on `dev_validation` image labels, with matched seeds (see open
  question 4 for the count) to measure training variability.
- **[human]** The user launches it on Kaggle or Colab.
- *Verify:* a local smoke test on synthetic data shows the arms differ only in
  the label hash. `pd.testing` asserts equality of training membership across
  arms.

**24. Registration.**
- *Files:* create `tools/register_locked_comparison.py` →
  `outputs/label_quality/registration_v1.json`. It records the reference and
  candidate checkpoint hashes, prediction-file hashes on locked UIDs
  (predictions generated **before** opening), the manifest and reference
  commitment hashes, n, the endpoints, the analysis (step 16 function and seed),
  the gate, and the noninferiority margin. Its sha256 is appended to
  `commitments.jsonl`.

**25. Open and report. [custodian]**
- *Files:* create `tools/open_locked_evaluation.py`. It refuses unless the
  registration hash is in the commitments log, the prediction hashes match, and
  the manifest hash matches. It computes the step 16 statistics, missingness,
  diagnostics by scanner group and language, and the mined-label scores as
  secondary. It writes `docs/label_quality_result.md`.
- *Gate (from the source, verbatim numbers):*
  - The macro delta must be ≥ +0.005 with its paired 95% interval above zero.
  - No priority finding may show a material regression, against a predeclared
    −0.02 noninferiority margin. An interval crossing −0.02 means "regression
    safety unresolved".
  - All twelve findings are reported.
  - A priority-finding improvement requires Holm-adjusted evidence.
- Inconclusive stays inconclusive. Any tuning after opening requires new
  confirmation data. The tool writes an `opened_at` record, and a second
  opening with a different registration is refused.

### Traceability (source section → step)

| Source requirement | Step |
|---|---|
| Keep gold 58 as historical diagnostics, excluded | 3, 4 |
| Version labels, weights, report hashes, split manifest, reference predictions | 1 |
| Preserve historical folds; separate new manifest | 1, 3 |
| MSK readers reconcile `llm_extract.py` definitions with competition criteria; severity/borderline | 8 |
| Ledger separates report / extractor / image questions | 6 |
| Assign membership before inspecting findings; exclude reviewed from locked; patient grouping; duplicate images; linkage limits | 2, 3 |
| Folds 0/2/4 for discovery, minus reserved; finish priority first; other pending as secondary | 5, 11, 12 |
| 150 pairs per priority finding (50/50/50); count existing, dedupe; shortfall; random controls with inclusion probabilities and overlap flags; enriched kept separate | 5, 12 |
| Disagreement only for selection; summary by language, site/vendor, weight, extraction mode; uncertain and unmentioned covered | 5 |
| Blinded reader view | 7 |
| Priority-question table | 8 |
| Ledger fields; error classes; review final and intermediate; soft target < 0.5 not evidence | 6 |
| Second reviewer before adjudication; language expertise; counts by finding/direction/stratum; enlarge controls | 11, 12 |
| Deliverable: ledger and ranked categories; omissions distinct; silence not negative | 6, 12 |
| 1,000 studies: 300 dev / 700 locked; first 100 pilot; gold separate; counts are budget not power | 3, 15 |
| Representative core, seed, manifest; hard cases outside locked | 3, 5 |
| All twelve findings; expedite priority in pilot | 13, 18 |
| ≥100 pos/neg per priority finding in locked; expand; enriched panel separate | 15, 17, 18 |
| Two blinded readers, third adjudicator, report review after image commit | 9, 13, 18, 19 |
| Capture fields; preserve originals; no averaging | 7, 13 |
| Unresolved/non-assessable handling; sensitivity; agreement by finding and site | 9, 14, 18 |
| Authorized readers, access requirements, budget from pilot minutes | 9, 14 |
| No scoring the current ensemble as independent | 4, 18 |
| Exclude both new partitions, linked duplicates and gold 58 from every member; matched training population | 4, 23 |
| Locked labels with custodian, out of prompts/calibration/rules/selection/blend | Approach, 18, 25 |
| Extend gold leakage assertion to full manifest | 4 |
| OOF interim diagnostics only with verified exclusion; disclose non-pristine | 4, 18 |
| Power plan: +0.005 macro, +0.02 priority, +0.003 macro; 80%/5%; multiplicity; patient bootstrap; document | 15, 16 |
| Pilot conservative; increase before opening; limit claim; no sequential peeking | 15, 17 |
| Candidate steps 5.1–5.5 | 20–25 |
| Confirmation gate and noninferiority | 25 |
| Delivery order and completion criteria | 10, 14, 18, 25 |

## Testing

New test files, each with at least one input that must fail:
- `tests/test_label_manifest.py`: determinism, group integrity, exclusion of
  prior review and gold from locked, and blindness to label columns.
- `tests/test_splits.py` (extended): manifest leakage, linked-patient leakage,
  and hash mismatch.
- `tests/test_audit_sampling.py`: strata, inclusion probabilities,
  deduplication, shortfall, and reserved-study exclusion.
- `tests/test_ledger.py`: every validation rule.
- `tests/test_blinded_packets.py`: forbidden fields and values are absent from
  output; refusal without a signed rubric or image commitment.
- `tests/test_power.py`: null rejection ≈ 5%; a known-effect power check.
- `tests/test_paired_eval.py`: agreement with `compare_oof.compare` in the
  degenerate case; clustering widens intervals.
- `tests/test_corrections.py`: evaluation labels untouched; hash and quote
  binding; stale old values rejected.
- `tests/test_label_quality_pair.py`: arms differ only in label hash; excluded
  studies are absent from training; generated source compiles.
- `tests/test_locked_evaluation.py`: refuses without registration, on a
  prediction-hash mismatch, and on a second opening.
- `tests/test_patient_links.py`: synthetic duplicate detection, and handling of
  blank IDs.
- `tests/test_notebook.py`: historical notebooks are unchanged when
  `EXCLUSION_MANIFEST` is unset.

Regression: run the full `.venv/bin/python -m pytest`, plus
`tools/audit_labels.py` (104,376 cells reproduce) and `tools/check_notebook.py`.

## Risks and open questions

1. **Timeline versus human capacity (decision needed).** Final submission is
   2026-10-22, 30 days away. 1,000 studies × 2 MSK reads + adjudication, plus
   450+ report reviews, likely exceeds that window. The source's completion
   criteria do not depend on the competition deadline, but the user should
   decide whether this plan runs past it, or whether only Stages 1–2 are
   targeted before it.
2. **Reader eligibility (decision needed).** The rules forbid giving data to
   non-participants, so readers must be team members. The team-merger deadline
   is 2026-10-15. Who the readers, adjudicator and custodian are is not
   something code can resolve.
3. **Custodian independence.** If the user is both modeler and custodian,
   separation is procedural. The hash commitments make changes detectable, not
   impossible. This should be disclosed in the result.
4. **Seed count and CV structure for the matched pair.** The source says
   "matched seeds" and "every reference/candidate member" without a count or
   an ensemble size. Proposal: 3 seeds × fold 1 at 224 px for screening, and a
   five-fold matched pair for the final registration. Costs are measured at
   ~1.9 h per 224 fold on Kaggle, so ~57 GPU-h for the pair. The user decides.
5. **Checkpoint selection within the pair.** The existing recipe selects on the
   mined-label val fold. With the new exclusions, that val fold is the
   historical fold minus reserved studies. Keep that selector identical across
   arms (the source's requirement), or select on `dev_validation`, which the
   source permits ("may guide selection"). Implementation choice pending: keep
   the mined selector and use dev_validation only for screening.
6. **Patient linkage may be impossible.** If `PatientID` is pseudonymised per
   study, only duplicate-image grouping remains. The bootstrap then treats
   studies as clusters, and the result states that limit.
7. **Where the 1,000 come from.** The source does not say whether
   dev_validation and locked are drawn from all five historical folds or only
   folds 1/3. This plan draws from all five (excluding prior review and gold),
   which maximizes the representative pool. The cost is that folds 0/2/4
   discovery loses those studies, which the source explicitly accepts
   ("excluding all newly reserved validation studies").
8. **Enlargement threshold for controls** (interval width 0.20) and the
   **audit OOF thresholds** (0.3/0.7) are implementation choices. The user may
   set others before sampling, never after.
9. **Previous exposure.** All 4,349 studies have informed training or selection
   of existing models. No study is untouched by selection history, so this is
   disclosed per the source ("relabeling cannot erase previous exposure").
10. **Source doc is untracked.** `docs/label_quality_plan.md` should be
    committed before step 1, so the manifest can record its hash.
