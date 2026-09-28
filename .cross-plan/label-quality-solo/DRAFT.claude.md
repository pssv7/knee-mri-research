## Goal

Carry out `docs/label_quality_plan.md` as far as **one person** can: no MSK readers, no adjudicator, no second reviewer, no custodian from the team. Every requirement in the source ends in one of three states, recorded in a status table in the result:
- **done**: as written;
- **adapted**: the solo substitute, and what it can and cannot show;
- **not executable**: with the reason.

Deliverables, in order:
1. A frozen baseline, and a solo report-review rubric.
2. A blinded, sampled report audit of Lateral Meniscus, Contusion and PF OA. It has an error ledger, design-appropriate intervals, and a delayed blind self re-review.
3. A go/no-go decision under a stop rule declared before the audit starts.
4. If go: an evidence-backed repair candidate. It is validated on a reserved report sample that was never used for rule design.
5. If go: a matched five-fold control/candidate pair. Both arms are registered, then submitted before **2026-10-22**, and read once from the **private** leaderboard after the competition closes. Kaggle's hidden, radiologist-labelled test set is the only independent evaluation available to one person, and Kaggle is a custodian the user cannot influence.

Done does not require a gain. "No-go", "inconclusive" and "not executable" are valid outcomes when stated with their reasons.

## Context

- Source: `docs/label_quality_plan.md` (untracked; commit it before step 1).
- Data (verified locally; not visible to Codex's redacted copy):
  - `data/meta/train.csv`: 4,407 studies with `Report`. Its twelve label columns are filled only for the 58 gold studies.
  - `data/meta/train_labels.parquet`: 4,349 rows, 12 targets plus 12 `_weight`; no gold rows.
  - `data/meta/llm_labels.parquet` has `<finding>`, `_severity`, `_confidence` and `_evidence`, but no certainty field. `data/meta/llm_probs.parquet` has `<finding>` and `_confidence`.
  - `outputs/ensemble_16_24/reconstructed_folds.parquet` is the historical 5-fold split.
- Labels code:
  - `src/labels/llm_extract.py` (`FINDING_DEFINITIONS`; in the prompt, `unclear` means unmentioned);
  - `src/labels/fuse.py` and `src/labels/mode_fusion.py`;
  - `tools/audit_labels.py` (`detect_language`, and the 104,376-cell reproduction check).
- Existing review:
  - `tools/build_label_review.py` builds `outputs/label_review_v1/` (250 pairs, 23 reviewed, 227 pending). Its `review.html` shows scores and targets, so it is not blinded.
  - Notes are in `tools/label_review_v1_notes.json`.
- Corrections and pairs:
  - `src/labels/reviewed.py::apply_reviewed` supports held-out folds 1 and 3 only, and `new_target == 0.0` only.
  - `tools/prepare_reviewed_labels.py::stage_source` injects corrections after `folded` is built, so stratification is unchanged. The fold-3 Colab variant is `tools/prepare_colab_reviewed_labels.py`.
  - `tools/prepare_kaggle_training_pair.py::training_source` accepts seeds 0/1 only. Its `PREFLIGHT` asserts 4,349 studies and the historical fold sizes, which is correct here because this plan adds no new exclusions.
- Measured cost: the fold-3 matched pair took 4.91 h on Kaggle (`docs/protocol_fold3.md`), so about 2.45 GPU-h per arm-fold at 224 px, 16×5, 12 epochs.
- Evaluation facts:
  - The fold-1 reviewed-label pair gave −0.003086 on mined OOF, with paired interval [−0.010543, +0.004496] on 871 studies (`docs/reviewed_labels_candidate.md`). That interval's half-width (~0.0075) is the best available proxy for the precision of one paired AUC comparison at this scale.
  - `docs/competition_pages.md`: 5 submissions a day, and a public leaderboard that is a "representative sample" of about 1,300 test studies; the private LB decides standings. The public/private split size is not stated.
- Rules (`docs/rules_notes.md`): no competition data to non-participants, which includes hosted LLMs and assistants. Final submission is 2026-10-22.
- Checks: `tests/`, run with `.venv/bin/python -m pytest`.

## Approach

**Keep everything the source requires that one person can do, and relabel honestly what changes.** The report audit, blinding, sampling design, error ledger, reserved validation sample and matched pair all survive. The image reference, readers, adjudication and team custodian do not.

**Second reviewer → delayed blind self re-review.** At least seven days after the first pass, the same person re-reviews every random control and every proposed target change. The packet uses new pseudonymous IDs, a new order, and no first-pass decisions. **A correction is applied only if both passes propose it independently.** Intra-rater agreement is reported. It is disclosed as a measure of consistency, not independence.

**Original-language expertise → local translation aid.** Packets carry a translation made by an open-weight model run locally (Colab, as in `notebooks/03_extract_labels.ipynb`), never a hosted service. The ledger's evidence quote must always be an exact substring of the original-language report.

**Locked evaluation → the Kaggle private leaderboard.** Both arms are submitted before the deadline under a registration committed first. After the close, the private scores of both submissions are read once. It is paired (same studies), radiologist-labelled, held by Kaggle, and opened once.
- *Limits:* one macro number with no interval and no per-finding scores, and the public LB has been used for earlier decisions.
- *Consequence:* the source's per-finding claims, noninferiority margin and multiplicity control are not executable on it. The macro claim is limited to what the sample can resolve. The MDE estimate from the fold-1 bootstrap is about 0.01, so the source's +0.005 gate cannot be confirmed. Anything under the MDE is reported as inconclusive (source §4, "limit the claim").

**The stop rule is fixed before the audit** (implementation choice; the user confirms it at step 3). **Go** if at least one priority finding has ≥ 5 of its 50 random controls with a fixable extraction error that both review passes agree on. Otherwise **no-go**: deliver the ledger and stop. The rule exists because label repair has never moved the leaderboard here, and one person's time is the binding cost.

**Protection of report text.** Report text stays local. Tools never print report text to the console. They report counts, IDs and hashes, so development driven through an assistant doesn't expose it. Packets and ledgers live under ignored paths.

**Alternatives rejected:**
- *The user acting as image reader.* That is one person, not two independent qualified readers plus an adjudicator. It would produce labels that look independent and aren't.
- *An assistant or hosted model as second reviewer or translator.* This breaks the data rule, and it wouldn't be independent clinical review.
- *Mined OOF or the gold 58 as the confirmation set.* Mined OOF is circular for label repair, since repaired labels are scored against unrepaired ones. The gold 58 has been inspected repeatedly and the source forbids fresh confirmation from it. Both stay as secondary diagnostics.
- *The public LB as the confirmation.* It has been reused and gets consulted during selection. It is an interim diagnostic only.
- *Cell-only corrections, like the fold-1 11.* They were already measured at −0.003, inside noise. A corpus-wide rule candidate is the thing still untested.

## Steps

**Stage 1: Protocol (target 2026-09-27)**

1. **Freeze the baseline.** Create `tools/freeze_label_baseline.py`. It hashes targets, weights, per-study report hashes, historical folds, gold, the extractor dumps, reference OOF (reusing `build_label_review.py`'s verified fallback paths), and the source document's blob. The inventory JSON goes to a tracked path, `docs/label_quality/baseline_v1.json`, with hashes only.
   - *Verify:* a rerun is byte-identical, and a tampered input fails. This is reproduction, not correctness.
2. **Write the report-review rubric.** Create `docs/report_review_rubric.md`. For each finding, it sets the competition definition (`docs/competition_pages.md`) beside `FINDING_DEFINITIONS`, then states the text-review decision rules and the priority questions from the source's table (all three rows).
   - It is marked "solo text-review rubric: not an MSK-reader image annotation standard".
   - Its hash is recorded in the manifest.
   - *Verify:* each of the source's priority questions has a rule or an explicit "unresolved".
3. **Reserve the validation sample, then build the audit sample.**
   - Create `src/labels/audit_sampling.py` and `tools/build_priority_audit.py`.
   - Population: folds 0/2/4 pairs for the three findings.
   - First, reserve an extraction-validation sample before anything is reviewed: 40 pairs per finding, stratified by target ≥ 0.5 and language (seed 20260923). It is excluded from the audit and from rule design.
   - Then, for each finding:
     - draw 50 random controls first, keeping inclusion probabilities and overlap flags;
     - draw 50 suspected false positives (target ≥ 0.5, protocol OOF ≤ 0.3) and 50 suspected misses (target < 0.5, OOF ≥ 0.7), at random within each stratum, with no duplicate pairs;
     - reuse eligible `label_review_v1` reviews with their selection provenance marked;
     - record any shortfall.
   - Disagreement only selects; nothing is fitted to it.
   - The summary covers language, scanner group, weight, extraction mode, and assertion class. Assertion class comes from the review itself, since the dumps can't separate uncertain from unmentioned.
   - The user confirms the stop rule and the thresholds before sampling runs.
   - *Verify:* the tests below.
4. **Build the ledger and blinded packets.**
   - Create `src/labels/ledger.py` with all the source's fields and the seven error classes. It keeps report omissions and unresolved image questions separate. It rejects evidence a changing correction needs but doesn't have, a change justified only by "target < 0.5", and a change to a correct final target made to repair an intermediate call.
   - Create `tools/build_blinded_packets.py` for report packets. They contain a pseudonymous ID, the original report and the local translation. They contain no scores, targets, weights, extractor output or stratum.
   - `--pass 2` builds the re-review packet with new IDs and order.
   - *Verify:* forbidden fields and values are absent from every output file.
5. **Prepare the local translation.** Create `tools/prepare_translation_colab.py` → `notebooks/22_translate_packets_colab.ipynb`. It uses an open-weight model, runs on Colab with the packets on the user's Drive, and has no hosted API.
   - *Verify:* the notebook passes `tools/check_notebook.py`-style static checks, and a dry run on 3 English packets returns them unchanged in meaning.

**Stage 2: Solo audit (2026-09-28 → 2026-10-10)**

6. **First pass.** Review the 450 priority pairs, existing priority cases first, controls before strata so the re-review clock starts early. Ingest with `tools/ingest_ledger.py`.
7. **Second pass.** At least seven days after each item's first pass, re-review all controls and all proposed changes blind. The agreement rule decides which corrections stand.
8. **Audit report.** Create `tools/audit_report.py` → `docs/label_audit_v2.md`.
   - Error counts by finding × direction × stratum.
   - Design-weighted control estimates with intervals suited to the design (Wilson for unweighted controls; a bootstrap for weighted ones).
   - Intra-rater κ, ranked repair categories, and omissions and image questions kept separate.
   - If a control interval is wider than 0.20, draw one more batch of 50, declared now, once.
   - The go/no-go decision under the stop rule.
   - *Verify:* a synthetic ledger with known rates is recovered within the intervals.
9. **Secondary workstream.** The 227 pending v1 checks are done after the priority audit if time allows before 2026-10-10. Otherwise they are recorded as deferred.

**Stage 3: Candidate (only if go; 2026-10-10 → 2026-10-13)**

10. **Rules.** Create `src/labels/repair.py`: narrow, corpus-wide rules over the existing dumps for the ranked categories. Each requires an exact quote, correct anatomy and explicit assertion, and flags uncertainty. Evidence-based changes are kept separate from anything conjectural. Freeze the rule-source hash.
11. **Validation.** Create `tools/validate_repair.py`. Review the reserved 120 pairs blind (first pass only, with the time limit stated) *after* the freeze. Report per language: positives, negatives, ambiguous; misses vs false alarms before and after. **Reject** the candidate if missed positives rise. If validation prompts a rule change, the sample is spent and the claim is limited.
12. **Label table.** Create `src/labels/corrections.py::apply_corrections(labels, folded, corrections, heldout_fold)`. It works for any held-out fold, changes training rows only, can change targets and weights, and binds hashes and quotes. Leave `apply_reviewed` as it is. Create `tools/build_candidate_labels.py`, which writes the versioned table and correction ledger with old/new values, the two-pass decisions and the rationale.
    - *Verify:* validation-fold rows are byte-identical for each of the 5 folds.

**Stage 4: Matched pair and registration (2026-10-13 → 2026-10-20)**

13. **Pair generator.** Create `tools/prepare_label_quality_pair.py`, generalising `stage_source` to folds 0–4.
    - Both arms use the same recipe (protocol ConvNeXt, 224 px, 16×5, 12 epochs, effective batch 8), seed 0, fixed historical folds and identical selection.
    - The candidate applies corrections to training rows only.
    - Each arm writes a contract with label, correction and code hashes.
    - The control is trained fresh alongside the candidate, not reused, so both come from the same code and campaign.
    - Cost: 10 arm-folds × ~2.45 GPU-h ≈ 25 GPU-h, split across Kaggle (45 h/week) and Colab. **[user]** launches.
    - *Verify:* the generated sources differ only in the correction hook; `PREFLIGHT` passes; a smoke run on synthetic data works.
14. **Screen.** Seed 1 on fold 1 for both arms (+~5 GPU-h) to measure training variability. Mined-OOF deltas and gold-58 deltas are safety diagnostics only: stop if the macro OOF delta is below −0.01 on both seeds. Neither selects anything.
15. **Registration.** Create `tools/register_lb_comparison.py` → `docs/label_quality/registration_v1.json` (tracked).
    - It records the checkpoint hashes, the Kaggle model-dataset and notebook versions of both arms, the MDE and its derivation, and the decision rule: private delta ≥ MDE = "consistent with benefit"; |delta| < MDE = inconclusive; ≤ −MDE = "consistent with harm".
    - It is committed **before** either submission.
16. **Submit both arms.** Reuse the `tools/prepare_protocol_submission.py` packaging for each arm. **[user]** submits both before 2026-10-20, leaving a 2-day buffer. Public scores are recorded as interim diagnostics. Which submissions become the two final entries is a separate competition decision. It does not decide the label-repair result.

**Stage 5: Readout (after 2026-10-22)**

17. **Read the private scores once.** Create `tools/read_lb_comparison.py`. It reads both registered submissions' private scores through the Kaggle API, refuses on an ID or registration-hash mismatch, and writes `docs/label_quality_result.md`. The result covers:
    - the delta against the registered rule;
    - the public delta (interim);
    - mined-OOF and gold diagnostics (secondary);
    - the full requirement-status table;
    - every limitation.

## Testing

- `tests/test_audit_sampling.py`: controls drawn first; strata disjoint from controls; the reserved sample excluded; inclusion probabilities; shortfall; determinism.
- `tests/test_ledger.py`: every validation rule has a failing case.
- `tests/test_blinded_packets.py`: no forbidden fields or values; pass-2 IDs and order differ; no report text on stdout (capture it and assert only counts and hashes appear).
- `tests/test_corrections.py`: each held-out fold unchanged; stale values and quote/hash mismatches rejected; weights validated.
- `tests/test_label_quality_pair.py`: arms differ only in the hook; folds 0–4 compile; contracts written.
- `tests/test_lb_registration.py`: readout refuses on a changed registration or a submission-ID mismatch.
- A synthetic-ledger statistics check (step 8).
- Regression: `.venv/bin/python -m pytest`, plus `tools/audit_labels.py` (104,376 cells reproduce).

## Risks and open questions

1. **Workload.** Reads come to about 450 + 150 re-review + ~100 changed-cell re-review + 120 validation ≈ 820, at about 2 minutes each: roughly 27 hours in 12 days, alongside competition work. If Stage 2 isn't finished by 2026-10-10, Stage 3 doesn't start. The ledger is delivered alone.
2. **Stop-rule thresholds.** The ≥ 5/50 rule, the 0.3/0.7 suspect thresholds, the 40-per-finding validation reserve and the 0.20 enlargement width are implementation choices. The user confirms or changes them before step 3 runs, never after.
3. **LB precision.** The public/private sizes are not published. The MDE (~0.01) comes from a mined-label OOF proxy, which is an assumption. The private LB cannot confirm +0.005, and the result must say so.
4. **Kaggle shows private scores for all submissions after the close** by standard practice. Confirm this on the competition page before relying on it. If not, only the two final entries would be readable.
5. **No independence anywhere in the review.** One person selects, reviews, re-reviews and designs rules. The reserved sample and the two-pass agreement rule limit this but don't remove it. The result says so.
6. **Compute contention.** The ~30 GPU-h compete with any other final-submission work in the same week.
7. **Not executable** (recorded in the result, with reasons):
   - MSK rubric sign-off;
   - the 1,000-study image set and its dev/locked/pilot partitions;
   - the two readers and the adjudicator;
   - image capture fields and reader agreement;
   - the budget from pilot minutes;
   - patient linkage for a reserved training set;
   - per-finding locked AUC, noninferiority and multiplicity;
   - reader-uncertainty power simulation.
