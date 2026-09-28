## Goal

Implement all of `docs/label_quality_plan.md`: complete the report audit, establish an independently reviewed image reference, develop and validate evidence-backed supervision repair, and measure its effect in a matched experiment with protected confirmation data.

Deliver four stages in order:

1. Protocol, membership, blinded packets, and leakage checks.
2. Completed priority audit and 100-study image pilot.
3. Expanded development and locked review at a power-informed sample size.
4. Frozen candidate, matched training, registration, and locked comparison.

Completion requires auditable evidence and uncertainty estimates, not a positive result. Preserve the original label table, historical folds, and historical experiment behavior.

The immediate next action, as the source states, is to prepare the rubric, the membership manifest, and the blinded priority-review packets (steps 1–8).

Nothing in this plan launches GPU or Kaggle jobs, distributes review packets, uploads data, or changes `data/meta/train_labels.parquet` on its own authority. Steps marked **[user]** need the user to authorize or perform them. Steps marked **[readers]** are blocked on qualified MSK readers, not on code.

## Context

- `data/meta/train.csv` contains 4,407 studies, reports, and twelve finding columns. The 58 gold studies are separate from the 4,349 rows in `data/meta/train_labels.parquet`.
- Preserve `outputs/ensemble_16_24/reconstructed_folds.parquet`, which contains historical study/fold membership.
- Extraction and fusion live in `src/labels/llm_extract.py`, `src/labels/lexicon.py`, `src/labels/fuse.py`, and `src/labels/mode_fusion.py`.
- The existing categorical dump does not preserve categorical uncertainty separately from severity. Its missing presence state denotes unmentioned or unavailable extraction, not a verified uncertain report assertion.
- `tools/build_label_review.py` builds an unblinded queue of 250 pairs; 23 have full-report notes and 227 remain pending. Its protocol provenance checks include fallback paths for relocated artifacts.
- Keep `src/labels/reviewed.py::apply_reviewed` and its historical candidate unchanged. Add a separate correction implementation supporting targets, weights, and the new membership protocol.
- `src/model/splits.py` provides historical splitting, scanner proxies, and gold exclusion. Extend it without replacing those contracts.
- `tools/prepare_kaggle_training_pair.py::training_source` and `tools/prepare_reviewed_labels.py::stage_source` provide useful generation patterns, but their population, seed, and evaluation assumptions require explicit adaptation.
- `src/model/artifacts.py` binds checkpoints to actual training and validation UIDs. Extend provenance for the new protocol.
- `tools/compare_oof.py::compare` remains the historical fold-blocked, mined-label diagnostic. Implement independent evaluation separately.
- Use `requirements.txt`, `requirements-dev.txt`, pytest, and the existing notebook generators/checkers. `tests/test_notebook.py` and `tools/check_notebook.py` principally validate the extraction notebook; they do not establish that the new training pipeline works.
- `data/meta/resolution_index/index_train.parquet` carries `Manufacturer`, `ManufacturerModelName`, `MagneticFieldStrength`, `SeriesDescription` and `ProtocolName`, but **not** `PatientID`. The raw DICOM headers do carry a pseudonymised `PatientID` and `PatientSex` (checked on `data/dev/*.dcm`). `InstitutionName`, `StationName`, `StudyDate` and `AccessionNumber` are blank there. So patient linkage needs a new header scan on Kaggle, where the raw DICOM lives, and "site" can only be approximated by `src/model/splits.scanner_key`.
- `tools/audit_labels.py::detect_language` is the existing dependency-free language detector for the language strata and diagnostics.
- `docs/rules_notes.md`: competition data may not be provided "to any party not participating in the Competition", and it is under the MIRA license. Readers, adjudicators and the custodian must therefore be competition participants. The team-merger deadline is 2026-10-15; the final submission is 2026-10-22. No report text may go to a hosted API.
- `.gitignore` ignores `/outputs/`, `/data/` and `*.parquet`. `docs/label_quality_plan.md` is currently untracked.
- Store restricted artifacts in permitted private storage. Track code, documentation, artifact inventories, and commitments explicitly; do not assume ignored output files are versioned.

## Approach

Use immutable, versioned protocol artifacts with separate hashes for baseline inputs, linkage, membership, rubric, sampling, reference labels, candidate supervision, training contracts, and registration. A protocol receipt binds these hashes without circular dependencies.

Construct patient/duplicate connected components before sampling. Separate study partition, pilot membership, extraction-validation reservation, review status, and training exclusion reason rather than representing overlapping roles as mutually exclusive partitions.

Keep three judgments distinct: what the report says, whether extraction represents it correctly, and what the images establish. Report omissions and unresolved image questions never become automatic negative corrections.

Enforce blinding through separate packet builders, storage locations, and reader-specific exports. Locked labels remain under custodian control. Only the custodian’s evaluation workflow reads them for final scoring. Locked labels are excluded from prompts, calibration, correction rules, checkpoint selection, and blend fitting, and every one of those code paths is tested to refuse a locked UID.

Held-out model disagreement is used only to choose which cases get reviewed. No threshold, rule, weight, or correction may be fitted to model scores.

Reader disagreements are resolved by third-reader adjudication, never by averaging labels.

Alternatives rejected:
- Modifying `outputs/ensemble_16_24/reconstructed_folds.parquet` or re-cutting folds. The source requires the historical folds to be preserved, with a separate manifest.
- Scoring the current five-fold ensemble on newly reviewed studies and calling it independent validation. Its members trained on those studies.
- Extending `apply_reviewed` in place. Its fold-1/3 contract backs the recovered fold-1 result and is tested.
- A hosted LLM API for any extraction or review aid. The rules forbid it.

Preserve historical folds by loading their frozen assignments. Exclude both new validation partitions, gold, and all linked studies from every new training member. Measure training variability separately from uncertainty conditional on fitted models.

## Steps

### Stage 1: Protocol and manifests

1. **Freeze baseline inputs and establish artifact storage.**

   Create `tools/freeze_label_baseline.py`. Record hashes, schemas, row counts, label order, and immutable locations for:

   - Original targets and weights.
   - Reports and per-study report hashes.
   - Historical folds and gold references.
   - Categorical and probability extraction dumps.
   - Protocol, incumbent, and resolution OOF predictions and available checkpoint receipts.
   - The source requirements and relevant code versions. Commit `docs/label_quality_plan.md` first **[user]**, since it is untracked, so that its blob hash can be recorded.

   Reuse the hash-verified fallback resolution in `tools/build_label_review.py` for relocated OOF files. Fail on missing, ambiguous, or mismatched artifacts.

   Keep restricted tables and packets in versioned private storage; track a small artifact inventory and commitment records at an explicit nonignored path. Reproduction receipts must state that matching existing outputs does not establish label correctness.

2. **Define and sign the annotation protocol. [readers] [user]**

   Create `docs/annotation_rubric.md` and `docs/reader_workflow.md`. Qualified MSK readers reconcile all twelve `FINDING_DEFINITIONS` with the available competition criteria, documenting severity thresholds, borderline cases, uncertainty, and assessability.

   Include every priority question:

   - Lateral Meniscus: tear versus degeneration/intrasubstance signal, compartment attribution, negation, multilingual missed phrases, postoperative and uncertain findings.
   - Contusion: bone versus soft-tissue/tendon edema, traumatic versus degenerative marrow change, uncertain cause, and coexisting fracture without assumed exclusivity.
   - PF OA: patellar/trochlear versus tibiofemoral disease, lateral patellar misattribution, mild cartilage changes, and unspecified compartment.

   Specify two independent image readers, third-reader adjudication, original-language expertise, reader identities, and permitted data access. Confirm access requirements before packet distribution.

   Predefine unresolved/non-assessable handling: exclude affected pairs from primary AUC, report denominators and exclusions, and perform stated positive/negative sensitivity analyses without silently assigning negatives.

3. **Build patient and duplicate linkage.**

   Create `tools/scan_patient_links.py` and a permitted CPU scanning workflow. Develop it locally against `data/dev/*.dcm`, then run it as a Kaggle CPU job with internet off **[user]**. Read headers with `stop_before_pixels`. Inspect relevant raw DICOM identifiers across series (`PatientID`, `PatientSex`, `PatientAge`, `StudyDate`, `AccessionNumber`); preserve missing identifiers and flag inconsistent identifiers within a study.

   Combine usable patient links with confirmed duplicate-image links into transitive connected components. Use cached-volume hashes and perceptual comparisons as duplicate candidates, checking preprocessing provenance, low-information images, and near-match false positives.

   Record linkage coverage, ambiguity, and limitations. Singleton pseudonyms do not prove that patients have no repeat examinations. Retain the limitation when linkage cannot be established.

4. **Build membership before new finding review.**

   Create `src/labels/manifest.py` and `tools/build_membership_manifest.py`.

   Establish component eligibility before sampling:

   - Gold remains historical-only; linked studies are excluded from new training and representative reference enrollment.
   - Components containing previously reviewed cases cannot enter locked evaluation.
   - Prior exposure includes the existing review queue and other documented case-review artifacts.
   - Audit-selected hard cases remain outside the representative locked core.

   Sample a representative core using scanner/vendor proxies, language, and acquisition descriptors, with deterministic handling of sparse strata and missing metadata. Record component selection, study inclusion probabilities, seed, sampling order, and sampling frame.

   Target 300 development-validation and 700 locked studies. Preserve whole-component assignment and document any unavoidable allocation overshoot. The first 100 development studies are the pilot; pilot status is a flag within development validation.

   Assign remaining eligible folds-0/2/4 studies to audit/development and other eligible studies to the training pool. Store historical fold unchanged. Audit membership remains training-eligible unless another exclusion applies.

   Reserve a fresh extraction-validation sample and linked components before releasing design findings. Keep its detailed findings inaccessible to rule designers until candidate rules are frozen.

5. **Implement leakage and provenance checks.**

   Extend `src/model/splits.py` with manifest loading, checksum/schema validation, exclusion closure, and assertions on actual training UIDs. Preserve `assert_no_gold_leakage`.

   Create `tools/check_checkpoint_exclusion.py`. Verify checkpoint hashes, sidecars, complete UID provenance, and exclusion of each evaluated study and its linked component. Missing proof makes historical predictions ineligible for independent or interim held-out claims. In particular, the current five-fold ensemble is never scored on newly reviewed studies as independent validation. Interim OOF diagnostics are labelled "not a pristine test (historical model selection)". If no previously untouched studies exist, the report says so.

   Add an explicit optional protocol configuration to relevant training generation paths. Historical behavior remains unchanged when it is absent.

6. **Implement audit sampling and the ledger.**

   Create `src/labels/audit_sampling.py`, `src/labels/ledger.py`, and `tools/build_priority_audit.py`.

   For each priority finding, target 150 unique eligible pairs:

   - 50 random controls from the full eligible finding population.
   - 50 suspected false positives.
   - 50 suspected missed positives.

   As a fixed implementation choice, define suspect strata using target ≥0.5 with verified held-out protocol prediction ≤0.3, and target <0.5 with prediction ≥0.7.

   Draw controls first. Retain their suspect-stratum overlap flags and always retain them in control estimates. Select the two additional groups without duplicating control pairs. Reuse eligible historical reviews, identifying certainty selections separately from new random draws and retaining valid selection probabilities. Record genuine population shortfalls.

   Use disagreement only to select; the sampler writes no rule, threshold or weight. Summarize language, scanner/vendor, weight, extraction mode, and assertion coverage. Use report annotation or explicitly validated structured extraction for positive, negative, uncertain, and unmentioned classes; preserve unknowns where legacy dumps are insufficient. Add separately identified coverage cases if necessary.

   The ledger records exact evidence, report hash, anatomy, assertion, certainty, severity, old target/weight, intermediate calls, final-target assessment, disposition, reviewer decisions, adjudication, and rationale. Error classes are exactly the source's seven: anatomy, negation, missed mention, severity/definition, fusion/weighting, report–image disagreement, unresolved. Report omissions and unresolved image questions are separate categories from fixable extraction errors. The ledger also carries sampling stratum, inclusion probability, and reviewer IDs for both reviewers.

   Evidence can be absent for an omission observation; a changing correction requires adequate exact evidence. Prevent changes based solely on a target below 0.5 or an incorrect intermediate call when the final target is already correct.

7. **Build blinded packets and ingestion contracts.**

   Create `tools/build_blinded_packets.py` with explicit field allowlists.

   Report packets contain pseudonymous IDs, full reports, and finding questions. Keep scores, targets, weights, extraction outputs, prior decisions, and sampling roles outside reviewer exports.

   Image packets provide permitted access to full diagnostic series and necessary series descriptors, without reports, mined labels, predictions, or another reader’s decisions. Do not substitute training-cache slices for full diagnostic review.

   Capture present/absent/indeterminate, severity, location, assessability, supporting image references, reader identity, timestamps, and notes.

   Preserve independent report interpretations before controlled comparison with extractor/fused outputs for correction assessment. Require image-reference commitment before report review of an image-reviewed study.

8. **Complete the Stage 1 gate.**

   Version the signed rubric, workflow, membership, sampling receipt, blinded priority packets, and leakage checks. Record artifact hashes and readiness in `docs/worklog.md` and `PROJECT.md`.

   Changes to rubric, linkage, or membership create new versions and explicitly invalidate affected packets, samples, references, or model eligibility. Never silently replace committed files.

### Stage 2: Audit and 100-study pilot

9. **Complete the report audit. [readers]**

   Create `tools/ingest_ledger.py`. Finish eligible existing priority cases first, then the new priority sample.

   A second reviewer independently assesses proposed corrections and every random control before adjudication. Record original-language expertise where needed.

   Complete the other pending v1 full-report checks as the secondary workstream, deduplicating work already completed. Reserved image-review studies follow the image-first rule and cannot contribute to training-rule discovery while protected.

10. **Report audit results and enlarge controls when necessary.**

    Create `tools/audit_report.py` and `docs/label_audit_v2.md`.

    Report errors by finding, direction, and sampling stratum, preserving separate control and enriched results. Include agreement, uncertainty, evidence-linked repair categories, omissions, and unresolved image questions.

    Use intervals appropriate to the actual selection design and patient clustering. Ordinary binomial intervals are permissible only for an explicitly justified corresponding analysis.

    Predeclare control enlargement and its inferential treatment. For example, assess the initial interval once and, if its width exceeds 0.20, choose a fixed additional random-control batch using a conservative precision calculation. Do not repeatedly inspect ordinary fixed-sample intervals without accounting for the stopping rule.

11. **Conduct and commit the pilot. [readers] [user]**

    Create `tools/ingest_image_reads.py`, `tools/commit_image_reference.py`, and `tools/pilot_report.py`.

    Two readers independently assess all twelve findings in the first 100 development studies, expediting the three priorities. A third adjudicates disagreements and uncertainty. Disagreements are resolved by adjudication, never by averaging labels. Preserve originals and adjudication separately.

    Commit the image reference before report review. Report prevalence and positive/negative counts, raw agreement and κ, adjudication, assessability, and reader minutes by finding and scanner proxy.

    Estimate total cost from measured two-reader time, adjudication, preparation, and QA. Compare mined labels with image truth while keeping report omissions distinct from extraction errors.

12. **Implement inference statistics before running power planning.**

    Create `src/model/paired_eval.py`.

    Define weighted AUC, the twelve-finding macro estimand, paired deltas, patient/component resampling, and the relationship between sampling strata and clusters. Resample clusters jointly across both models and findings.

    Report per-finding assessable positive/negative counts and missingness. Specify handling of single-class findings and invalid bootstrap draws; do not silently change the primary macro’s finding set.

    Implement paired confidence intervals and a defined hypothesis-testing procedure for Holm-adjusted priority claims. Register the noninferiority procedure and −0.02 margin.

13. **Plan sample size using development evidence.**

    Create `src/model/power.py`, `tools/plan_sample_size.py`, and `docs/sample_size_plan.md`.

    Identify exact matched held-out prediction artifacts and verify patient exclusion before use. The recovered reviewed-label fold-1 pair is a potential source, subject to artifact and overlap checks. If coverage is insufficient, obtain an authorized development-only matched prediction pilot before final size lock.

    Simulate conservative ranges of prevalence, paired score correlation, dependence across findings, patient variation, and reader uncertainty. Use κ as agreement evidence, not a direct estimate of misclassification probability.

    Estimate sample size and MDE for +0.005 and +0.003 macro AUC and +0.02 for each priority finding. Target 80% power with two-sided 5% primary significance; include secondary multiplicity control and positive/negative counts.

    Disclose uncertainty from the small pilot and any historical model-selection exposure.

### Stage 3: Expanded review

14. **Complete development and locked image review. [readers] [custodian]**

    Review the remaining development studies and representative locked sample using the same independent-read/adjudication workflow.

    Locked reads, adjudications, and labels remain with the custodian. Publish only permitted commitments and prespecified aggregate feasibility information before final evaluation.

    The custodian checks for at least 100 adjudicated positives and 100 negatives per priority finding. If deficient, expand the representative sample under a prespecified blinded enrollment rule, without examining model performance.

    Keep any rare-finding enriched panel separately identified and separately reported, never pooled unweighted into the main score. Selecting unusually easy or hard cases can still distort AUC, so the enriched panel carries its selection rule in the report.

    Summarize reader agreement and adjudication rates by finding and scanner/site proxy for the full review. Locked aggregates are released by the custodian.

15. **Freeze final enrollment and references.**

    Combine power projections, blinded count feasibility, reader capacity, and available population to select final enrollment before opening locked performance.

    Add only eligible, unreviewed reserve components from the frozen sampling procedure. Exclude audited, extraction-validation, and otherwise exposed components. Recalculate valid inclusion probabilities for the final design.

    Issue a new membership version for enlargement and regenerate downstream exclusions and contracts. Any model trained on newly reserved studies is ineligible.

    If adequate precision or counts are infeasible, document the shortfall and restrict claims. Do not add cases after inspecting significance.

16. **Complete development report/image comparison.**

    After image commitment, review reports on the image-reviewed development set. Measure report coverage, extraction correctness, and image disagreement separately.

    Preserve the reserved extraction-validation sample from design access. Interim model diagnostics require checkpoint-level study/patient exclusion and carry the historical-selection limitation.

### Stage 4: Candidate and comparison

17. **Develop and independently validate repair.**

    Create `src/labels/repair.py` and `tools/validate_repair.py`.

    Develop narrowly scoped rules from the completed design audit, requiring report evidence, anatomy, and explicit uncertainty. If re-extraction is necessary, use permitted local open-weight execution and preserve structured evidence; version prompt and model configuration.

    Keep report-evidence corrections separate from image-only conjectures.

    Freeze rules before opening fresh extraction-validation results. Independently review language-specific positive, negative, and ambiguous examples. Report before/after false positives, missed positives, uncertainty, and coverage, with a predefined rule preventing acceptance of a false-positive reduction that increases missed positives.

    If validation drives further design, retire that validation sample and obtain a fresh one.

18. **Produce candidate supervision.**

    Create `src/labels/corrections.py::apply_corrections` and `tools/build_candidate_labels.py`.

    Write a separate versioned candidate table and correction ledger. Bind every change to source hashes, exact evidence, old/new target and weight, reviewer decisions, and rationale.

    Apply changes only to training-eligible membership. Represent exclusions through a documented common-population weighting policy so both arms retain the same study population. Preserve independent evaluation labels and historical validation labels used for checkpoint selection.

19. **Implement the matched training and prediction pipeline.** Training launches are **[user]**.

    Create `tools/prepare_label_quality_pair.py` using explicit adaptations of the existing generator.

    - Load and verify frozen historical fold assignments.
    - Derive training and historical-validation membership from those assignments and the final exclusion manifest.
    - Remove reserved and linked studies before constructing loaders.
    - Assert exclusions on actual loader UIDs.
    - Apply candidate changes only to each member’s training rows.
    - Replace hard-coded study counts with manifest-derived expected counts.
    - Validate all required images are available; fail or record a prospective protocol change instead of silently dropping studies.
    - Write actual membership, initialization, supervision, manifest, compute, and checkpoint hashes.

    Use the same protocol ConvNeXt architecture, 224-pixel inputs, 16 slices, five series, epochs, optimization, selection procedure, and matched initializations in both arms.

    Use supported seeds 0 and 1 for the development screen. Specify final fold/seed membership and aggregation before registration; any additional seeds require an explicit generator extension and tests.

    Add a label-independent prediction exporter for development and locked UIDs, using metadata without truth labels. Verify every contributing member’s exclusion and preserve exact UID coverage, label order, finite predictions, model hashes, and aggregation settings.

    Screen on development image labels and report seed variability. Gold and locked labels select nothing.

20. **Register the final comparison.**

    Create `tools/register_locked_comparison.py`.

    Before opening, commit final member identities, checkpoint and prediction hashes, membership/reference commitments, sample size, endpoints, missingness policy, analysis implementation, resampling configuration, multiplicity procedure, and decision gates.

    Validate complete prediction coverage and all provenance. Bind registration to an append-only custodian-controlled opening record.

21. **Open once and report. [custodian]**

    Create `tools/open_locked_evaluation.py` and `docs/label_quality_result.md`.

    Require matching registration, reference, predictions, and membership. Permit reproducible reruns of the identical registered analysis; prohibit a new candidate or changed analysis from reusing the set as fresh confirmation.

    Report paired macro and all twelve per-finding AUC deltas with 95% intervals, assessable counts, missingness, exclusions, sensitivity analyses, reader agreement/adjudication, scanner/site-proxy and language diagnostics, and provenance. Keep mined-label metrics secondary.

    Apply the source’s gate:

    - Macro delta ≥+0.005 and its paired 95% interval above zero.
    - Priority-finding regression assessed against the predeclared −0.02 margin; an interval crossing the margin leaves safety unresolved.
    - Priority-specific improvement requires multiplicity-adjusted evidence.

    State inconclusive results as inconclusive. Further tuning after opening requires new confirmation data.

### Traceability (source requirement → step)

This table exists because the request was to miss nothing. Every requirement in `docs/label_quality_plan.md` has a row.

| Source requirement | Step |
|---|---|
| Keep the 58 as historical diagnostics, excluded from training | 1, 4, 5 |
| Version original labels, weights, report hashes, split manifest, reference predictions | 1 |
| Preserve historical folds; separate manifest for the new protocol | 1, 4, 19 |
| MSK readers reconcile `llm_extract.py` definitions with competition criteria; severity thresholds; borderline handling | 2 |
| Ledger separates report / extractor / image questions | Approach, 6 |
| Assign membership before inspecting findings; reviewed cases out of locked; patient grouping; duplicate images; linkage limits | 3, 4 |
| Folds 0/2/4 for discovery minus reserved; existing priority cases first; other pending checks completed as secondary | 4, 9 |
| 150 unique pairs per priority finding (50 FP / 50 missed / 50 controls); count existing reviews and dedupe; shortfall; controls with inclusion probabilities and overlap flags; enriched separate from controls | 6, 10 |
| Disagreement only for selection; summary by language, site/vendor, weight, extraction mode; uncertain and unmentioned covered | Approach, 6 |
| Blinded reader view replacing the score-exposing HTML | 7 |
| Priority-question table (Lateral Meniscus, Contusion, PF OA) | 2 |
| Ledger fields; seven error classes; review final and intermediate targets; soft target < 0.5 not evidence | 6 |
| Second reviewer before adjudication; original-language expertise; counts by finding/direction/stratum; enlarge controls if wide | 9, 10 |
| Deliverable: ledger and ranked repair categories; omissions and image questions distinct; silence never a confident negative | 6, 10 |
| 1,000 studies = 300 dev + 700 locked; first 100 = pilot; 58 separate; counts are a budget, not power | 4, 13 |
| Representative core: sites, vendors, languages, protocols; fixed seed and sampling manifest; hard cases outside locked | 4 |
| All twelve findings; priority findings expedited in the pilot | 11, 14 |
| ≥100 adjudicated positives and negatives per priority finding in locked; expand; enriched panel separate, not pooled unweighted; easy/hard-case distortion | 14, 15 |
| Two independent blinded readers; third adjudicator; report review only after image commitment | 7, 11, 14, 16 |
| Capture fields; preserve originals and adjudication; no averaging | Approach, 7, 11 |
| Predefined unresolved/non-assessable handling; sensitivity analyses; agreement and adjudication by finding and site | 2, 11, 14, 21 |
| Authorized participating readers; access requirements; budget from measured pilot minutes | 2, 11, Risks |
| Current ensemble not scored as independent validation | Approach, 5 |
| Exclude both new partitions, linked duplicates and the 58 from every member; matched training population | 5, 19 |
| Locked labels with custodian; out of prompts, calibration, correction rules, checkpoint selection, blend fitting | Approach, 14, 19, 21 |
| Extend gold-only leakage assertion to the full versioned manifest | 5 |
| OOF interim diagnostics only with per-checkpoint exclusion; disclose non-pristine; disclose if no untouched studies | 5, 16 |
| Power: +0.005 macro, +0.02 per priority finding, +0.003 macro; 80% / two-sided 5%; multiplicity; patient bootstrap across models and findings; document assumptions, counts, MDEs | 12, 13 |
| Pilot used conservatively; increase before opening; limit the claim if infeasible; no adding cases after significance | 13, 15 |
| 5.1 narrow rules, exact evidence, anatomy, uncertainty; evidence separate from conjecture | 17 |
| 5.2 fresh independently reviewed validation; language-specific examples; FP fix must not raise misses; freeze | 4, 17 |
| 5.3 versioned table and ledger; training-only corrections; evaluation labels independent; compare extraction quality and coverage to image-reviewed dev | 16, 18 |
| 5.4 one matched experiment; dev screening; matched seeds for variability | 19 |
| 5.5 registration before opening; paired deltas, 95% intervals, missingness, site/language diagnostics; mined scores secondary | 20, 21 |
| Gate: macro ≥ +0.005 with CI > 0; −0.02 noninferiority; all twelve reported; multiplicity-adjusted priority claims; inconclusive stays inconclusive; no post-opening tuning | 21 |
| Delivery order and completion criteria; immediate next action | Goal, 8, 11, 15, 21 |

## Testing

- **Membership and linkage:** deterministic component construction; transitive links; blank/conflicting identifiers; duplicate candidates; gold-linked and previously reviewed components; sparse strata; valid selection probabilities; label-independent assignment; safe enlargement.
- **Audit:** 150 unique pairs when feasible; genuine shortfalls; legacy selection provenance; controls retain disagreement overlaps; assertion unknowns; reserved-study exclusion; appropriate interval coverage.
- **Ledger and blinding:** exact evidence/hash checks; unchanged correct final targets; silence protection; independent reads; role-specific exports; image-before-report enforcement; no holdout findings in design artifacts.
- **Statistics:** hand-check weighted/tied AUC; alignment and missingness failures; patient-cluster resampling; cross-finding dependence; calibrated null rejection and power; multiplicity; noninferiority boundaries. Compare with historical `compare` only under matched resampling assumptions.
- **Corrections:** stale values, invalid weights, unknown findings, duplicate changes, unsupported evidence, evaluation membership, and immutable input rejection.
- **Training:** execute injected setup/preflight with synthetic reduced-population data, following the execution pattern in `tests/test_reviewed_labels.py`; verify fixed folds, actual loader exclusions, both supported seeds, matched initialization, expected counts, unchanged selector labels, and equal training membership.
- **Inference and opening:** exact UID/label alignment; missing images or predictions; mismatched checkpoint/manifest hashes; linked-patient leakage; registration enforcement; immutable replay and prohibited changed comparisons.
- **Locked isolation:** every prompt, calibration, correction-rule, checkpoint-selection and blend-fitting entry point refuses a locked UID; adjudication ingestion rejects an averaged label in place of an adjudicated one.
- **Regression:** run relevant existing tests, then `.venv/bin/python -m pytest`, label reproduction via `tools/audit_labels.py`, and applicable notebook checks. Regenerate affected notebooks in a scratch checkout and inspect intended changes. Update dependencies explicitly for any new statistics library.
- Perform an authorized remote smoke run before full training. Distinguish source compilation, executed local checks, GPU validation, reproduction, and correctness evidence.

## Risks and open questions

- Reader eligibility, language coverage, adjudicator availability, permitted access, and custodian assignment are execution dependencies.
- The 1,000-study target and competition deadline may not accommodate the required review or statistical power. Resolve capacity using pilot measurements without weakening confirmation claims.
- Patient linkage and scanner-based site proxies may remain incomplete; disclose their limitations.
- Prior training and model selection have exposed the existing corpus. New image labels cannot erase that history.
- Small strata, rare findings, unresolved reads, and finite population size may prevent adequate power or valid twelve-finding confirmation.
- Final compute allocation and ensemble membership must be settled before registration. Development screening may inform that choice; locked results may not.
- Hashes demonstrate integrity, not independent custody. Access separation and the custodian’s opening record remain necessary.
- Preserve unrelated work in the current dirty working tree; implement and verify changes in an isolated working copy.
- **Decision for the user: timeline.** 1,000 studies × two MSK reads plus adjudication, plus 450+ report reviews, very likely exceeds the 30 days to the 2026-10-22 final submission. The source's completion criteria do not depend on the deadline. The user decides whether the plan runs past it, or whether only Stages 1–2 are targeted before it.
- **Decision for the user: who the readers and custodian are.** Under `docs/rules_notes.md` they must be competition participants (team members), and team mergers close 2026-10-15. If the user is both modeler and custodian, separation is procedural only, and the result must disclose that.
- **Patient linkage may be impossible.** If `PatientID` is pseudonymised per study, only duplicate-image grouping remains; clusters fall back to studies and the report states the limit.

## Review history

Converged after 3 reviews (v1 drafted by Claude).

| Round | Reviewer | Verdict | Main changes |
|---|---|---|---|
| 1 | Codex on v1 | REVISE → v2 | Patient/duplicate components built *before* sampling, so linked moves can't pull reviewed studies into locked or break inclusion probabilities. Extraction-validation sample reserved at membership time, not after results. Controls drawn first and kept in control estimates; design- and cluster-appropriate intervals instead of Clopper–Pearson; pending reviews completed, not queued. Assertion classes from report annotation (the dumps carry no certainty and `unclear` means unmentioned). Evaluator built before power planning; κ not treated as misclassification; ≥100 pos/neg enforced blind by the custodian. Training generator adapted for manifest-derived counts, seeds 0/1, loader-level exclusion assertions and a label-free prediction exporter. Ignored-output storage and the fold-1/3 receipt fallback handled. |
| 2 | Claude on v2 | REVISE → v3 | Restored source requirements v2 dropped: traceability table, the seven error classes, no-averaging, the full locked-label exclusion list, disagreement-for-selection-only, explicit non-independence of the current ensemble, enriched-panel pooling rules, immediate next action, [user]/[readers]/[custodian] gates, and the PatientID / participant-only / deadline facts as user decisions. |
| 3 | Codex on v3 | APPROVE | — |
