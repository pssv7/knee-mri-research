VERDICT: REVISE

## Critique

1. **The proposed training integration will fail and lacks independent inference.** Step 23 of `.cross-plan/label-quality-plan/PLAN.v1.md` asserts exclusion before removing reserved studies. `tools/prepare_kaggle_training_pair.py::PREFLIGHT` also requires 4,349 indexed studies and the original fold counts; `training_source` accepts only seeds 0 and 1, conflicting with the proposed three-seed screen. `tools/make_notebook.py` recomputes folds and constructs evaluation around historical validation and gold loaders. The plan needs explicit fixed-fold loading, manifest-derived preflight counts, assertions on actual loader membership, and a separate prediction path for development and locked studies.

2. **The membership algorithm does not preserve its promised exclusions or sampling design.** Step 3 samples studies and subsequently moves linked groups using `locked > dev_validation > audit`. This can move previously reviewed studies into locked evaluation, alter the 300/700 allocation, and invalidate the recorded study inclusion probabilities. Gold-linked groups are not covered by that precedence rule. Patient and duplicate connected components must be constructed before eligibility and sampling. `src/model/splits.py` currently supplies no such component-level machinery.

3. **The audit sampling and uncertainty rules contradict the source requirements.** Step 5 permits overlapping selections, so deduplication can yield fewer than 150 unique pairs despite sufficient population. Its test excluding disagreement-stratum rows from control estimates would discard legitimate randomly selected controls and bias the estimate. Step 12 combines design-weighted estimates with ordinary Clopper–Pearson intervals, which do not generally apply to weighted, clustered observations. The deterministic legacy selections in `tools/build_label_review.py` also need distinct selection provenance. Finally, steps 5 and 12 only queue the remaining reviews; `docs/label_quality_plan.md` requires completing them.

4. **The assertion categories cannot be recovered as described.** Step 5 identifies categorical `unclear` as uncertainty and refers to `certainty = possible`. In `src/labels/llm_extract.py`, `unclear` explicitly means unmentioned; `parse_response` folds certainty into a numeric severity value, and `aggregate_samples` does not preserve certainty. The actual `data/meta/llm_labels.parquet` contains presence, severity, confidence, and evidence, without a certainty field. The plan needs report-based classification or new structured extraction, with unknown states preserved.

5. **The extraction-validation sample is protected too late.** Steps 12, 14, and 19 expose reviewed development results before step 20 divides those data into design and extraction validation. That cannot establish the fresh, independently reviewed sample unused in rule design required by `docs/label_quality_plan.md` §5. Its membership and access restrictions must be established before development findings are released.

6. **Power planning and locked feasibility are incomplete.** Step 15 depends on the evaluator implemented in step 16 without an earlier implementation gate, does not identify usable matched held-out prediction artifacts, and treats pilot κ as identifying reader misclassification. Agreement alone cannot identify reader error against truth. Cross-finding dependence also matters for macro-AUC power. Although the traceability table claims coverage, steps 15–18 never enforce at least 100 adjudicated positives and negatives per priority finding or specify how blinded count shortfalls trigger expansion. `tools/compare_oof.py::compare` supplies neither design-aware inference nor finding-specific hypothesis tests; Holm-adjusted claims need an explicitly implemented testing procedure.

7. **Versioning and baseline recovery are not executable as specified.** Step 1 refers directly to receipt paths that are absent for protocol folds 1 and 3; `tools/build_label_review.py` already implements verified fallback discovery. Additionally, `.gitignore` excludes `/outputs/` and all Parquet files, while the plan requires committing artifacts under `outputs/label_quality/`. The rubric is added after manifest creation, and later enrollment changes the manifest again, without a clear dependency and invalidation protocol. Explicit artifact storage, tracked commitments, immutable versions, and regeneration rules are necessary.

## Revised plan

### Goal

Implement all of `docs/label_quality_plan.md`: complete the report audit, establish an independently reviewed image reference, develop and validate evidence-backed supervision repair, and measure its effect in a matched experiment with protected confirmation data.

Deliver four stages in order:

1. Protocol, membership, blinded packets, and leakage checks.
2. Completed priority audit and 100-study image pilot.
3. Expanded development and locked review at a power-informed sample size.
4. Frozen candidate, matched training, registration, and locked comparison.

Completion requires auditable evidence and uncertainty estimates, not a positive result. Preserve the original label table, historical folds, and historical experiment behavior.

### Context

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
- Store restricted artifacts in permitted private storage. Track code, documentation, artifact inventories, and commitments explicitly; do not assume ignored output files are versioned.

### Approach

Use immutable, versioned protocol artifacts with separate hashes for baseline inputs, linkage, membership, rubric, sampling, reference labels, candidate supervision, training contracts, and registration. A protocol receipt binds these hashes without circular dependencies.

Construct patient/duplicate connected components before sampling. Separate study partition, pilot membership, extraction-validation reservation, review status, and training exclusion reason rather than representing overlapping roles as mutually exclusive partitions.

Keep three judgments distinct: what the report says, whether extraction represents it correctly, and what the images establish. Report omissions and unresolved image questions never become automatic negative corrections.

Enforce blinding through separate packet builders, storage locations, and reader-specific exports. Locked labels remain under custodian control. Only the custodian’s evaluation workflow reads them for final scoring.

Preserve historical folds by loading their frozen assignments. Exclude both new validation partitions, gold, and all linked studies from every new training member. Measure training variability separately from uncertainty conditional on fitted models.

### Steps

#### Stage 1: Protocol and manifests

1. **Freeze baseline inputs and establish artifact storage.**

   Create `tools/freeze_label_baseline.py`. Record hashes, schemas, row counts, label order, and immutable locations for:

   - Original targets and weights.
   - Reports and per-study report hashes.
   - Historical folds and gold references.
   - Categorical and probability extraction dumps.
   - Protocol, incumbent, and resolution OOF predictions and available checkpoint receipts.
   - The source requirements and relevant code versions.

   Reuse the hash-verified fallback resolution in `tools/build_label_review.py` for relocated OOF files. Fail on missing, ambiguous, or mismatched artifacts.

   Keep restricted tables and packets in versioned private storage; track a small artifact inventory and commitment records at an explicit nonignored path. Reproduction receipts must state that matching existing outputs does not establish label correctness.

2. **Define and sign the annotation protocol.**

   Create `docs/annotation_rubric.md` and `docs/reader_workflow.md`. Qualified MSK readers reconcile all twelve `FINDING_DEFINITIONS` with the available competition criteria, documenting severity thresholds, borderline cases, uncertainty, and assessability.

   Include every priority question:

   - Lateral Meniscus: tear versus degeneration/intrasubstance signal, compartment attribution, negation, multilingual missed phrases, postoperative and uncertain findings.
   - Contusion: bone versus soft-tissue/tendon edema, traumatic versus degenerative marrow change, uncertain cause, and coexisting fracture without assumed exclusivity.
   - PF OA: patellar/trochlear versus tibiofemoral disease, lateral patellar misattribution, mild cartilage changes, and unspecified compartment.

   Specify two independent image readers, third-reader adjudication, original-language expertise, reader identities, and permitted data access. Confirm access requirements before packet distribution.

   Predefine unresolved/non-assessable handling: exclude affected pairs from primary AUC, report denominators and exclusions, and perform stated positive/negative sensitivity analyses without silently assigning negatives.

3. **Build patient and duplicate linkage.**

   Create `tools/scan_patient_links.py` and a permitted CPU scanning workflow. Inspect relevant raw DICOM identifiers across series; preserve missing identifiers and flag inconsistent identifiers within a study.

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

   Create `tools/check_checkpoint_exclusion.py`. Verify checkpoint hashes, sidecars, complete UID provenance, and exclusion of each evaluated study and its linked component. Missing proof makes historical predictions ineligible for independent or interim held-out claims.

   Add an explicit optional protocol configuration to relevant training generation paths. Historical behavior remains unchanged when it is absent.

6. **Implement audit sampling and the ledger.**

   Create `src/labels/audit_sampling.py`, `src/labels/ledger.py`, and `tools/build_priority_audit.py`.

   For each priority finding, target 150 unique eligible pairs:

   - 50 random controls from the full eligible finding population.
   - 50 suspected false positives.
   - 50 suspected missed positives.

   As a fixed implementation choice, define suspect strata using target ≥0.5 with verified held-out protocol prediction ≤0.3, and target <0.5 with prediction ≥0.7.

   Draw controls first. Retain their suspect-stratum overlap flags and always retain them in control estimates. Select the two additional groups without duplicating control pairs. Reuse eligible historical reviews, identifying certainty selections separately from new random draws and retaining valid selection probabilities. Record genuine population shortfalls.

   Summarize language, scanner/vendor, weight, extraction mode, and assertion coverage. Use report annotation or explicitly validated structured extraction for positive, negative, uncertain, and unmentioned classes; preserve unknowns where legacy dumps are insufficient. Add separately identified coverage cases if necessary.

   The ledger records exact evidence, report hash, anatomy, assertion, certainty, severity, old target/weight, intermediate calls, final-target assessment, disposition, reviewer decisions, adjudication, and rationale. Include all required error classes and distinct omission/unresolved-image categories.

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

#### Stage 2: Audit and 100-study pilot

9. **Complete the report audit.**

   Create `tools/ingest_ledger.py`. Finish eligible existing priority cases first, then the new priority sample.

   A second reviewer independently assesses proposed corrections and every random control before adjudication. Record original-language expertise where needed.

   Complete the other pending v1 full-report checks as the secondary workstream, deduplicating work already completed. Reserved image-review studies follow the image-first rule and cannot contribute to training-rule discovery while protected.

10. **Report audit results and enlarge controls when necessary.**

    Create `tools/audit_report.py` and `docs/label_audit_v2.md`.

    Report errors by finding, direction, and sampling stratum, preserving separate control and enriched results. Include agreement, uncertainty, evidence-linked repair categories, omissions, and unresolved image questions.

    Use intervals appropriate to the actual selection design and patient clustering. Ordinary binomial intervals are permissible only for an explicitly justified corresponding analysis.

    Predeclare control enlargement and its inferential treatment. For example, assess the initial interval once and, if its width exceeds 0.20, choose a fixed additional random-control batch using a conservative precision calculation. Do not repeatedly inspect ordinary fixed-sample intervals without accounting for the stopping rule.

11. **Conduct and commit the pilot.**

    Create `tools/ingest_image_reads.py`, `tools/commit_image_reference.py`, and `tools/pilot_report.py`.

    Two readers independently assess all twelve findings in the first 100 development studies, expediting the three priorities. A third adjudicates disagreements and uncertainty. Preserve originals and adjudication separately.

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

#### Stage 3: Expanded review

14. **Complete development and locked image review.**

    Review the remaining development studies and representative locked sample using the same independent-read/adjudication workflow.

    Locked reads, adjudications, and labels remain with the custodian. Publish only permitted commitments and prespecified aggregate feasibility information before final evaluation.

    The custodian checks for at least 100 adjudicated positives and 100 negatives per priority finding. If deficient, expand the representative sample under a prespecified blinded enrollment rule, without examining model performance.

    Keep any rare-finding enriched panel separately identified and separately reported.

15. **Freeze final enrollment and references.**

    Combine power projections, blinded count feasibility, reader capacity, and available population to select final enrollment before opening locked performance.

    Add only eligible, unreviewed reserve components from the frozen sampling procedure. Exclude audited, extraction-validation, and otherwise exposed components. Recalculate valid inclusion probabilities for the final design.

    Issue a new membership version for enlargement and regenerate downstream exclusions and contracts. Any model trained on newly reserved studies is ineligible.

    If adequate precision or counts are infeasible, document the shortfall and restrict claims. Do not add cases after inspecting significance.

16. **Complete development report/image comparison.**

    After image commitment, review reports on the image-reviewed development set. Measure report coverage, extraction correctness, and image disagreement separately.

    Preserve the reserved extraction-validation sample from design access. Interim model diagnostics require checkpoint-level study/patient exclusion and carry the historical-selection limitation.

#### Stage 4: Candidate and comparison

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

19. **Implement the matched training and prediction pipeline.**

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

21. **Open once and report.**

    Create `tools/open_locked_evaluation.py` and `docs/label_quality_result.md`.

    Require matching registration, reference, predictions, and membership. Permit reproducible reruns of the identical registered analysis; prohibit a new candidate or changed analysis from reusing the set as fresh confirmation.

    Report paired macro and all twelve per-finding AUC deltas with 95% intervals, assessable counts, missingness, exclusions, sensitivity analyses, reader agreement/adjudication, scanner/site-proxy and language diagnostics, and provenance. Keep mined-label metrics secondary.

    Apply the source’s gate:

    - Macro delta ≥+0.005 and its paired 95% interval above zero.
    - Priority-finding regression assessed against the predeclared −0.02 margin; an interval crossing the margin leaves safety unresolved.
    - Priority-specific improvement requires multiplicity-adjusted evidence.

    State inconclusive results as inconclusive. Further tuning after opening requires new confirmation data.

### Testing

- **Membership and linkage:** deterministic component construction; transitive links; blank/conflicting identifiers; duplicate candidates; gold-linked and previously reviewed components; sparse strata; valid selection probabilities; label-independent assignment; safe enlargement.
- **Audit:** 150 unique pairs when feasible; genuine shortfalls; legacy selection provenance; controls retain disagreement overlaps; assertion unknowns; reserved-study exclusion; appropriate interval coverage.
- **Ledger and blinding:** exact evidence/hash checks; unchanged correct final targets; silence protection; independent reads; role-specific exports; image-before-report enforcement; no holdout findings in design artifacts.
- **Statistics:** hand-check weighted/tied AUC; alignment and missingness failures; patient-cluster resampling; cross-finding dependence; calibrated null rejection and power; multiplicity; noninferiority boundaries. Compare with historical `compare` only under matched resampling assumptions.
- **Corrections:** stale values, invalid weights, unknown findings, duplicate changes, unsupported evidence, evaluation membership, and immutable input rejection.
- **Training:** execute injected setup/preflight with synthetic reduced-population data, following the execution pattern in `tests/test_reviewed_labels.py`; verify fixed folds, actual loader exclusions, both supported seeds, matched initialization, expected counts, unchanged selector labels, and equal training membership.
- **Inference and opening:** exact UID/label alignment; missing images or predictions; mismatched checkpoint/manifest hashes; linked-patient leakage; registration enforcement; immutable replay and prohibited changed comparisons.
- **Regression:** run relevant existing tests, then `.venv/bin/python -m pytest`, label reproduction via `tools/audit_labels.py`, and applicable notebook checks. Regenerate affected notebooks in a scratch checkout and inspect intended changes. Update dependencies explicitly for any new statistics library.
- Perform an authorized remote smoke run before full training. Distinguish source compilation, executed local checks, GPU validation, reproduction, and correctness evidence.

### Risks and open questions

- Reader eligibility, language coverage, adjudicator availability, permitted access, and custodian assignment are execution dependencies.
- The 1,000-study target and competition deadline may not accommodate the required review or statistical power. Resolve capacity using pilot measurements without weakening confirmation claims.
- Patient linkage and scanner-based site proxies may remain incomplete; disclose their limitations.
- Prior training and model selection have exposed the existing corpus. New image labels cannot erase that history.
- Small strata, rare findings, unresolved reads, and finite population size may prevent adequate power or valid twelve-finding confirmation.
- Final compute allocation and ensemble membership must be settled before registration. Development screening may inform that choice; locked results may not.
- Hashes demonstrate integrity, not independent custody. Access separation and the custodian’s opening record remain necessary.
- Preserve unrelated work in the current dirty working tree; implement and verify changes in an isolated working copy.