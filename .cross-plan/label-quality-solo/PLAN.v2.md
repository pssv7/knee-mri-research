## Goal

Execute `docs/label_quality_plan.md` with one person. No additional reviewers, MSK readers, adjudicator, language expert, or team custodian will join.

Deliver, in order:

1. A frozen baseline and solo report-review rubric.
2. A blinded audit of Lateral Meniscus, Contusion, and PF OA, with an error ledger, sampling-aware estimates, and delayed self re-review.
3. A decision under a stop rule fixed before sampling.
4. If feasible, a frozen repair candidate validated on previously uninspected reports.
5. If the review, validation, compute, and submission gates pass, a matched five-fold comparison submitted before the competition deadline and read once from the private leaderboard.

Record each source requirement as done, adapted, deferred, or not executable, with reasons. No-go, inconclusive, and an audit-only result are valid outcomes.

## Context

The source is `docs/label_quality_plan.md`; preserve its committed version before implementation.

Data facts supplied by the local reviewer:

- `data/meta/train.csv` contains 4,407 reports; only the 58 gold studies have the twelve supplied labels.
- `data/meta/train_labels.parquet` contains 4,349 studies, twelve targets, and twelve weights, excluding gold.
- Categorical dumps contain finding, severity, confidence, and evidence fields, but no certainty field. Probability dumps contain finding and confidence.
- Historical membership is in `outputs/ensemble_16_24/reconstructed_folds.parquet`.

Relevant code:

- `src/labels/llm_extract.py::FINDING_DEFINITIONS`, `src/labels/fuse.py`, and `src/labels/mode_fusion.py` define existing extraction and fusion behavior.
- `tools/audit_labels.py` provides language detection and the recorded 104,376-cell reproduction check.
- `tools/build_label_review.py` produces an unblinded queue with scores, targets, extractor evidence, and full reports. Existing review provenance must be retained.
- `src/labels/reviewed.py::apply_reviewed` supports held-out folds 1 and 3 and zero-target corrections only.
- `tools/prepare_reviewed_labels.py::stage_source` inserts corrections after fold construction and before index construction.
- `tools/prepare_kaggle_training_pair.py::training_source` supports seeds 0 and 1; its `match_slice=True` baseline path already supports folds 0–4 with protocol series selection.
- Existing preflight counts match the unchanged 4,349-study training population, but do not prove exact historical fold membership.
- `tools/prepare_protocol_submission.py` and `tools/prepare_protocol_benchmark.py` implement a specific historical 15-model ensemble, not arbitrary five-model arms.
- Tests use pytest. `requirements-dev.txt` includes the training and test dependencies. Some existing tests require actual data, outputs, or notebooks.

The historical fold-3 pair took approximately 4.91 GPU-hours, giving an initial estimate of 25 GPU-hours for ten arm-folds and another five for the seed-1 fold-1 pair. Translation, packaging benchmarks, inference, and retries are additional costs.

The historical eleven-cell experiment’s mined-OOF delta was −0.003086 with an interval crossing zero. It neither establishes a benefit nor excludes broader repair.

Competition rules, deadline, submission limits, and private-score availability must be checked by the participant before execution. Competition content stays outside hosted assistants.

## Approach

Retain the source’s report audit, blinded packets, provenance, sampling design, validation reserve, and matched experiment. Adapt independent review to delayed self re-review and independent evaluation to the registered Kaggle private-score comparison.

A report review establishes what the report says, not image truth. Omission, uncertainty, and unresolved image questions remain separate from fixable extraction errors. Silence never becomes a confident negative.

Every applied target or weight change requires two agreeing report-based decisions from the same person, separated by at least seven days. This measures consistency, not independence. Corpus-wide rule matches are proposals until they satisfy this requirement; unseen matches never inherit another case’s review status.

Reserve extraction-validation studies before discovery. Their complete reports and all findings stay out of rule design, including the secondary review workstream. This reserve does not create an image-model holdout: historical image-training membership remains unchanged, and extraction validation is described accordingly.

Use a locally executed open-weight translation model where needed. Original-language evidence must match the source report exactly. Translation is an aid; uncertain language interpretation leads to abstention.

Fix the initial stop rule before sampling: go only if at least one priority finding has five or more agreed fixable errors among its original 50 random controls. Any extra control batch improves estimation but cannot change this denominator or retrospectively redefine the gate.

Use mined OOF and gold-58 scores only as secondary diagnostics. Keep the settled approximate 0.01 decision threshold derived from the historical paired-OOF precision proxy, with its assumptions stated. It is not a private-test confidence interval or proof of statistical significance. Per-finding private-test safety and multiplicity claims remain unavailable.

## Steps

1. **Freeze inputs and protocol by September 27.**  
   Create `tools/freeze_label_baseline.py`. Inventory hashes of original labels and weights, report hashes, historical fold membership, gold membership, extractor dumps, verified reference predictions, source document, and code. Reuse `build_label_review.py`’s verified protocol fallback logic. Provide separate creation and verification modes; verification rejects changed inputs.

   Write the rubric and sampling protocol before the final manifest is sealed. Keep only safe hashes, counts, and protocol metadata in tracked documents; keep report-bearing artifacts and review mappings in ignored locations.

2. **Write `docs/report_review_rubric.md`.**  
   Reconcile competition definitions with `FINDING_DEFINITIONS` as far as a solo text reviewer can. Cover:

   - Lateral Meniscus: tear versus degeneration, laterality, negation, multilingual mentions, postoperative and uncertain findings.
   - Contusion: bone versus soft tissue, traumatic versus degenerative marrow change, uncertain cause, and coexistence with fracture.
   - PF OA: patellar/trochlear versus tibiofemoral disease, lateral compartment confusion, mild changes, and unspecified compartment.

   Mark unresolved definition questions explicitly. State that this is not an MSK image-annotation standard.

3. **Create protected sampling manifests.**  
   Implement `src/labels/audit_sampling.py` and `tools/build_priority_audit.py`.

   Inventory prior report exposure from existing reviews and other known inspection records, conservatively marking uncertain exposure. From folds 0/2/4, reserve 40 pairs per priority finding, stratified by target threshold and language using seed 20260923, from studies without known prior exposure. Protect the union of their study IDs across every finding and review workflow.

   If sufficiently fresh reports are unavailable, report the shortfall and limit or abandon fresh-validation claims.

   For each finding, draw 50 controls uniformly from the eligible discovery population first. Preserve inclusion probabilities and disagreement overlap flags. Then fill separate 50-pair suspected-false-positive and 50-pair suspected-miss strata using target/OOF thresholds 0.5 and 0.3/0.7.

   Count eligible existing reviews toward enriched quotas, documenting their nonrandom provenance; sample remaining slots randomly. Existing reviews count as controls only if selected by the new control draw. Never pool enriched cases into control error-rate estimates.

   Obtain the user’s threshold choices before running sampling, then freeze them.

4. **Build review infrastructure and translation.**  
   Create `src/labels/ledger.py`, `tools/build_blinded_packets.py`, `tools/ingest_ledger.py`, and `tools/prepare_translation_colab.py`.

   Packets show pseudonymous identifiers, finding/rubric, original report, and translation. They omit targets, scores, weights, extractor outputs, strata, and previous decisions. Keep identifier mappings outside the packet directory.

   Capture report interpretation before revealing the original fused target and intermediate extraction for error classification. Preserve both stages rather than overwriting the blinded judgment.

   The ledger records source hashes, exact evidence, anatomy, assertion, certainty, severity, old/proposed targets and weights, dispositions, timestamps, reviewer identity, and the seven source error classes. Reject unsupported changes and changes justified only by a low soft target or an incorrect intermediate call when the final target is already correct.

   Generate translation in the participant’s permitted local/Colab environment. Test execution with synthetic multilingual examples covering negation, anatomy, and uncertainty; an English identity check alone is insufficient.

5. **Perform the priority audit.**  
   Start controls and eligible existing priority cases early. Complete first passes in time for their seven-day delayed review.

   Re-review all controls and all proposed target or weight changes with new IDs and order, without first-pass decisions. Record disagreements as unresolved; do not apply their changes.

   Assess the predefined interval-width trigger early. If the initial control interval exceeds width 0.20, draw at most one additional batch of 50 controls for that finding. Complete its two passes or report the enlarged estimate as incomplete. Keep the original 50-control go/no-go gate unchanged.

6. **Report the audit by October 10.**  
   Create `tools/audit_report.py` and `docs/label_audit_v2.md`.

   Report error counts by finding, direction, and sampling stratum; control estimates with suitable intervals; language, scanner, weight, and extraction-mode coverage; assertion classes; ranked mechanisms; and unresolved questions.

   Report agreement on the random controls separately from agreement on selectively re-reviewed proposed changes. Document any adaptive sampling and its inferential limits.

   Issue the predefined go/no-go decision. If required audit reviews are incomplete by October 10, deliver the audit alone. Complete remaining v1 checks only if time permits and they cannot expose the validation reserve.

7. **Develop and review a feasible candidate.**  
   If go, implement narrow rules in `src/labels/repair.py` using the reviewed mechanisms. A missed mention may require searching the original report locally; missing extractor evidence is not evidence of absence.

   Generate proposals outside the validation reserve, with exact quotes, hashes, old/new values, anatomy, assertion, and uncertainty. Require individual two-pass agreement for every applied change, including weight-only changes.

   Reuse valid completed audit reviews. Any new case starts its own seven-day clock. Thus, newly proposed cases first read October 10 cannot be approved before October 17.

   Before proceeding, assess the actual proposal count and remaining review/compute time. If full corpus-wide coverage cannot fit, freeze a smaller, explicitly review-gated candidate or deliver the audit alone. Do not describe a partially reviewed subset as an unrestricted corpus-wide repair.

8. **Freeze and validate the candidate.**  
   Freeze rule code, parameters, acceptance policy, and reviewed correction ledger before opening the reserved reports.

   Implement `tools/validate_repair.py`. Review the reserved 120 pairs blind once, disclosing the single-pass limitation. Compare baseline and frozen rule outputs against those report interpretations, including per-language positives, negatives, ambiguity, false alarms, misses, and change coverage.

   Reject if missed positives rise. Sparse or untouched strata cannot establish safety. If validation causes any rule or policy revision, the reserve is spent; require another fresh reserve or stop this candidate’s promotion. Do not silently reuse it as confirmation.

9. **Implement fold-specific correction application.**  
   Create `src/labels/corrections.py::apply_corrections` and `tools/build_candidate_labels.py`; leave `apply_reviewed` unchanged.

   Validate IDs, finding names, finite target/weight ranges, stale old values, duplicates, source hashes, quotes, and two-pass approval. Apply only corrections belonging to the current training fold membership. Preserve every held-out row exactly and leave inputs immutable.

   Keep one immutable correction ledger and produce fold-specific effective-label hashes. Do not train from a globally corrected table that also changes validation labels.

10. **Train and enforce the matched comparison.**  
    Create `tools/prepare_label_quality_pair.py` using the existing baseline protocol-selection path.

    Load the exact frozen historical study-to-fold mapping independently of training seed. Validate membership, not just counts. Keep gold excluded.

    Both arms use protocol ConvNeXt, 224 pixels, 16 slices, five series, 12 epochs, effective batch eight, and identical checkpoint selection. Compare arm contracts explicitly, allowing differences only in arm identity, output locations, correction metadata, and effective supervision. Require matching initial weights, settings, environment, and fold membership.

    Run fold 1 at seeds 0 and 1 first. Stop if its macro mined-OOF delta is below −0.01 on both seeds. Otherwise complete seed-0 folds 0/2/3/4. Gold and mined-OOF results remain secondary diagnostics; discovery-fold OOF is not untouched validation of rule design.

    Keep each matched pair on the same platform/environment. The user launches GPU jobs.

11. **Package and benchmark both five-model arms.**  
    Implement dedicated packaging/submission adapters or parameterize shared internals while preserving the historical tools’ existing contracts.

    Each arm contains exactly one selected checkpoint from each seed-0 fold, equal probability weights of 1/5, identical inference settings, and isolated dataset/kernel identities.

    Verify checkpoint sidecars and hashes, reference-prediction reproduction, raw-input inference, runtime, output schema/order, finite predictions, and fallback behavior for each exact package. Benchmark receipts must bind manifest, checkpoint, and inference-source hashes.

    Reserve budget and time for these checks and retries. If completion cannot meet the submission buffer, deliver the completed work without claiming a private comparison.

12. **Register, submit, and read once.**  
    Before either submission, commit `docs/label_quality/registration_v1.json` with model/package hashes, benchmark receipts, notebook and dataset versions, inference settings, threshold derivation, endpoints, and readout rules.

    The user submits both arms by October 20 if feasible. Append submission IDs and available prediction hashes in a separately bound receipt without changing the registered analysis.

    Confirm private-score access for both submissions before relying on it, including whether final-entry selection affects availability. Public scores are interim diagnostics and cannot trigger changes to this comparison.

    After results become available, `tools/read_lb_comparison.py` verifies registration and submission identities and records the private delta once. Use API retrieval if supported; otherwise use a documented, provenance-preserving participant export.

13. **Deliver `docs/label_quality_result.md`.**  
    Report audit results, candidate scope, all gates, actual time/cost, diagnostics, private readout if available, and limitations.

    Include a source-requirement table covering every source bullet. Mark independent readers, adjudication, image-set partitions, image-reference capture, reader-power simulations, and private per-finding inference as not executable. Distinguish the protected report reserve from the unchanged image-training population.

## Testing

Use synthetic fixtures for development; no competition text in assistant-visible output or test failures.

- **Baseline:** deterministic serialization; changed labels, reports, folds, predictions, rubric, or source rejected.
- **Sampling:** study-level reserve isolation across findings; prior-exposure exclusion; deterministic strata; controls first; inclusion probabilities; existing-review quotas; deduplication; shortfalls; extra-batch behavior.
- **Packets and ledger:** schema allowlists, no hidden review metadata, separate mappings, changed pass-2 IDs/order, seven-day enforcement, target/weight approval, unsupported-change rejection, and safe logging.
- **Translation:** generated cells parse and execute under stubs; synthetic multilingual checks cover negation, anatomy, uncertainty, and model failures.
- **Repair/validation:** positive and negative rule fixtures, abstention, missed-mention handling, exact evidence, immutable freeze, reserve isolation, miss-regression rejection, and spent-reserve handling.
- **Corrections:** all five held-out folds unchanged; immutable inputs; invalid IDs, stale values, duplicates, nonfinite values, invalid weights, and quote/hash mismatches rejected.
- **Training:** execute injected hooks on synthetic frames; verify historical membership under both seeds; reject changed initialization/settings; test both-arm contract comparison and paths; compile all generated fold/seed variants.
- **Packaging:** five-member manifests; no mixed arms, missing or duplicate folds; exact benchmark binding; reproduction/runtime failures; submission schema and hash checks. Preserve historical 15-model regression tests.
- **Registration/readout:** immutable registration, separately bound submission receipts, ID/hash mismatches, missing private scores, and threshold boundaries.
- **Statistics:** deterministic synthetic examples and repeated simulations for estimator behavior and interval coverage.

Run data-independent tests in the redacted environment where dependencies permit. The participant runs data-dependent regressions and `tools/audit_labels.py` locally in the full repository, reporting only safe summaries. Do not characterize the unrestricted full suite as runnable in the redacted copy.

## Risks and open questions

- One person remains responsible for every judgment; delayed agreement and a fresh report reserve do not create independent clinical review.
- Corpus-wide proposals may exceed the review budget. Enforcing the settled two-pass policy may force a narrower candidate or audit-only delivery.
- Seven-day delays, adaptive controls, validation, training, and benchmarks make the deadline tight. No stage may bypass its gate to meet a date.
- Translation may mishandle critical findings. Unresolved meaning requires abstention.
- Previously inspected reports may be incompletely documented. Freshness claims must reflect that uncertainty.
- Validation may contain too few examples of a rule or language to establish useful safety evidence.
- The private leaderboard supplies a macro score without paired uncertainty or per-finding safety information; the historical precision proxy may transfer poorly.
- Private-score retrieval and access to both entries must be confirmed before committing compute to that endpoint.
- Training costs exclude translation, inference benchmarks, retries, and competing GPU work.
- Competition data, translations, quotes, generated notebooks containing text, and error traces must remain in participant-controlled permitted environments.
