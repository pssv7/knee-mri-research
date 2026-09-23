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
