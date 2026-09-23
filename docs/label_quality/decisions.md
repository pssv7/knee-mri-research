# Label-quality protocol decisions

Decisions taken while executing the one-person plan
(`.cross-plan/label-quality-solo/PLAN.final.md`, which adapts
`docs/label_quality_plan.md`). Each entry is fixed once recorded: a later change is a
new entry that says what it replaces. The sealed manifest (plan step 3) records this
file's hash.

## D1 — Private leaderboard scores as the locked evaluation (2026-09-23)

**Question (plan step 1):** after the close, does Kaggle show private scores for
every submission, or only for the two selected final entries?

**What was checked:**

- The competition rules allow up to two final submissions for judging
  (`docs/competition_pages.md`, "You may select up to two (2) Final Submissions"). They
  say nothing about private scores for the other submissions.
- `kaggle competitions submissions -c rsna-knee-abnormality-detection --csv` (CLI 2.2.4)
  returns a `privateScore` column for every submission, empty while the competition
  runs.
- Kaggle's competitions documentation page renders client-side and could not be read,
  and this account has no ended competition to check against.

Kaggle's usual practice is to show private scores for all submissions after the close.
That is **unverified** here.

**Decision:** option 1. Proceed as if every submission gets a private score. The two
final entries are chosen on competition grounds alone, and the label-repair arms do not
reserve them.

**Fallback, fixed now:** if either arm has no private score after the close, the
label-repair comparison is reported from public-leaderboard scores only. Those are
interim evidence, with the limitation stated, and are never presented as the locked
result.

## D2 — Report-review rubric signed (2026-09-23)

The participant, as sole reviewer, confirmed every proposed rule in
`docs/report_review_rubric.md` v1 as written, including:

- a grade III or surface-reaching meniscal signal counts as a tear;
- grade I–II degeneration does not;
- bone oedema tied to osteoarthritis is not a contusion, and oedema of unstated cause stays
  uncertain;
- the PF OA threshold is chondromalacia grade II;
- the secondary rules for the other nine findings.

These are reviewer decisions, not known competition criteria. The rubric's hash is sealed
into the step-3 manifest.

## D3 — Audit sampling settings and the stop rule (2026-09-23)

Confirmed by the participant ("confirm all") before any sample was drawn:

- Seed 20260923.
- Discovery population: folds 0/2/4.
- Validation reserve first: 40 pairs per priority finding, split 20 with target ≥ 0.5 and
  20 below, allocated across report languages in proportion. It is drawn only from studies
  with no known prior exposure, and the union of reserved studies is protected across every
  finding and workflow.
- Then 50 uniform random controls per finding.
- Then 50 suspected false positives (target ≥ 0.5, protocol OOF ≤ 0.3) and 50 suspected
  missed positives (target < 0.5, protocol OOF ≥ 0.7) per finding. Eligible existing
  `label_review_v1` pairs fill these quotas first, marked as non-random.
- Control enlargement: at most one extra batch of 50 when a control interval is wider than
  0.20.
- **Stop rule:** go only if at least one priority finding has five or more fixable errors,
  agreed by both review passes, among its **original** 50 controls. Extra controls never
  change that denominator.

## D4 — Definition of the control-interval trigger (2026-09-23, before any review)

D3 allows one extra batch of 50 controls when "a control interval is wider than 0.20".
Fixed here, before any case has been read:

- **Interval:** the 95% Wilson score interval for the share of a finding's original 50
  controls whose final target the first pass judges incorrect (`final_target_correct` is
  false after pass-1 stage B). Controls are a uniform draw, so the unweighted interval is
  the right one.
- **When:** assessed once per finding, as soon as all 50 of its original controls have a
  pass-1 stage B. The first assessment is stored and never recomputed, so the trigger
  can't be re-read until it fires.
- **Extra draw:** 50 pairs, uniformly, from the same finding's discovery population minus
  its original 50 controls. They are not drawn from outside the enriched strata: that keeps
  the 100 a uniform sample. A drawn pair that was already reviewed as an enriched case
  reuses that review, since reviews are blind to role.
- **Scope:** the extra controls improve the estimate only. The go/no-go stop rule stays on
  the original 50 (D3).

## D5 — What counts toward the stop rule, and when it is decided (2026-09-23, before any review)

D3's stop rule counts "fixable errors agreed by both review passes" among each finding's
original 50 controls. Fixed here, before any case has been read:

- **Counted:** a control where both passes independently propose a target correction on the
  same side of 0.5 (`ledger.agreed_change`). Only those count, because only they mean the
  final label is wrong. A weight-only change is recorded and reported, but it does not count.
- **Decidable:** for a finding, once all 50 original controls have a pass-1 stage B and every
  control with a pass-1 proposed change also has a pass-2 stage B. A finding that reaches five
  counted errors decides **go** as soon as that is established.
- **No-go:** every finding is decidable and none reaches five.
- **Deadline:** if the decision still can't be made on 2026-10-10, the audit is delivered
  alone (audit-only). A candidate is not built from an incomplete audit.

## D6 — Seven-day re-review delay waived (2026-09-23)

The participant, who wrote the plan, chose to start pass 2 on 2026-09-23, the same day as
the first readings (ingested 14:00, 16:11 and 20:57 UTC), instead of waiting the seven days
the approved plan requires. The reason given was to finish the audit as soon as possible.

**What changes:** the second pass is no longer separated from the first by time, so
agreement between the passes partly reflects recall of the first readings. That inflates
agreement, and it inflates the count of corrections both passes agree on, which is the
count the D5 go/no-go rule uses. The decision is therefore biased toward "go".

**What stays:** pass-2 packets still carry new case IDs, a new order, and no first-pass
answers. Every pass-2 key row records `delay_waived = true`, and `docs/label_audit_v2.md`
states this limitation next to the decision.

## D7 — Stop at the audit; steps 8–13 not run (2026-09-23)

The audit decided **go** for PF OA (D5; 11 agreed errors among its 50 original controls).
Step 7 then measured the proposal rule (`src/labels/repair.py`) before any further work:

- **Design data** (reviewed PF OA pairs with target < 0.5; the validation reserve untouched):
  - it finds 20 of the 36 agreed corrections (56%);
  - it proposes on 15 of the 85 labels the reviewer kept;
  - precision is about 57% (missed mentions 7/11, severity cases 13/26).
- **Corpus scale:** 648 proposals outside the reserve (211 missed mentions, 437 severity
  cases). Two-pass review would take about 1,300 reads before the 2026-10-13 freeze.
- **What the comparison could show:** the private leaderboard reports one macro AUC over
  twelve findings. Moving it by the ~0.01 the paired comparison can resolve would need a
  PF OA gain of about +0.12 AUC (its OOF AUC is about 0.816). A plausible +0.03 is
  +0.0025 macro, below both that limit and the source plan's +0.005 gate. No per-finding
  private score exists. The registered comparison would therefore read "inconclusive"
  whatever the repair did.

**Decision (participant, option 1):** stop at the audit. No candidate labels are built, no
training pair runs, and nothing is registered or submitted for label repair. Steps 8–13
are recorded as not run for this reason. The proposal rule stays in the repository as
documentation of the PF OA wording gaps; nothing in the pipeline imports it.
