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
