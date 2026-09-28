## Goal

Change the `cross-plan` skill so that Codex writes its own plan from scratch, not only reviews Claude's.

Done means `~/.claude/skills/cross-plan/SKILL.md` runs this flow:

1. Round 0 is **two independent drafts**. Codex drafts `DRAFT.codex.md` in the background while Claude drafts `DRAFT.claude.md`. Neither author sees the other's draft until both drafts exist.
2. Claude merges the two into `PLAN.v1.md`. Every point where the drafts differ is recorded in a divergence table in `DISCUSSION.md`, saying which way it went and why.
3. The existing alternating review loop then runs unchanged, starting with Codex. Codex's first review is told to check that the merge did not drop or misrepresent anything from its own draft.
4. Round 0 is optional. A `--single-draft` flag restores today's behaviour (Claude drafts alone), and a Codex draft failure falls back to it with a recorded note.

A dry run on a scratch repository produces both drafts, the merged v1, the divergence table, and at least one review round, with all files valid.

## Context

- `~/.claude/skills/cross-plan/SKILL.md` (161 lines) is the whole skill: no scripts or helpers. Relevant parts:
  - frontmatter (lines 1–6): `description`, `argument-hint: <feature or change to plan> [--max-rounds N]`, and `allowed-tools: Bash(codex exec:*), Bash(mkdir:*), Bash(git rev-parse:*), Bash(git status:*), Bash(ls:*), Bash(cp:*), Bash(head:*), Read, Write, Glob, Grep`.
  - lines 12–14: argument parsing (`--max-rounds N`).
  - lines 24–27: the file list for the output directory.
  - lines 29–38: the six-section plan format.
  - lines 60–65: "Round 1: Claude drafts". This is the only drafting step today, and it is Claude's alone.
  - lines 72–105: "Codex's turn". Codex runs as `codex exec -s read-only -C "<root>" -o <file> "<prompt>" < /dev/null`. It can only return a review; its revised plan is copied in by Claude.
  - lines 128–138: stopping rules. Convergence is an APPROVE on a version written by the other model.
  - lines 140–159: finish and report.
- `codex exec --help` (codex-cli 0.156.1) confirms the flags the plan relies on: `-o/--output-last-message <FILE>`, `-s read-only`, `-C`, `--skip-git-repo-check`, and `--add-dir <DIR>` (extra *writable* directories).
- Evidence from the previous run (`/Users/pssv/PycharmProjects/knee_competition/.cross-plan/label-quality-plan/`):
  - `review.r1.codex.md` put its revised plan under `### Goal` … headings instead of `## Goal`. The skill has no check for this, and Claude normalised it by hand before writing v3.
  - Codex ran ~5–10 minutes per call at `model_reasoning_effort = "high"` (from `~/.codex/config.toml`).
  - Codex browsed the repo freely, including data files and a source file embedding report excerpts. The skill gives no guidance on sensitive content.

## Approach

**Independent parallel drafts, then a recorded merge.** Codex drafting only after seeing Claude's v1 anchors it to Claude's framing, which is the weakness the user wants removed. A blind first draft from each model surfaces genuinely different designs. The merge makes those differences explicit instead of losing them.

Ordering that keeps the drafts independent:
1. Start the Codex draft command with `run_in_background: true` **before** Claude explores the repo.
2. Claude explores and writes `DRAFT.claude.md`. Claude does not open `DRAFT.codex.md` until its own draft is saved.
3. When the background task finishes, validate the Codex draft, then merge.

The Codex draft prompt receives only the feature text, the six-section format, and the rule that paths must exist. It gets no mention of Claude's approach.

**The merge is Claude's, and Codex audits it.** Claude writes `PLAN.v1.md` from both drafts and appends a `### Round 0: drafts merged` table to `DISCUSSION.md` with the columns `# | Topic | Claude draft | Codex draft | Taken into v1 | Reason`. Codex's round-1 review prompt adds: read `DRAFT.codex.md` and the round-0 table, and treat any dropped or distorted point from your draft as a REVISE issue if it is substantive. Round 1 is therefore where Codex contests the merge. After that, the existing loop and stopping rules apply unchanged. v1 is Claude-written, so a Codex APPROVE on v1 is convergence, as today.

**Codex stays read-only.** Its output still arrives only through `-o`, and Claude writes it to disk verbatim. That keeps the "Codex never modifies files" guarantee.

**Output validation, extended.** A Codex draft or revised plan must contain the six `## ` section headings. If the sections are present but at the wrong level (e.g. `###`), Claude normalises the heading level only, without touching content, and records that in `DISCUSSION.md`. If sections are missing, it is a failed run: retry once, as the existing rule does.

**Alternatives rejected:**
- *Codex writes its plan file itself* (`-s workspace-write`, or `--add-dir <dir>`). Rejected: workspace-write also lets it modify the repository, and `-o` already captures the output losslessly. `--add-dir` would still need workspace-write mode.
- *Codex drafts v1 alone, or the two alternate drafting authorship.* That swaps one anchor for another; it isn't independence.
- *Cross-review both drafts in round 0, then merge.* Rejected because it costs two extra reviews before the loop starts. The round-1 Codex audit of the merge gets the same check for one call.
- *Picking the better draft instead of merging.* That discards the other model's unique points, which is what the feature is for.
- *Counting drafts toward `--max-rounds`.* Rejected: the limit caps reviews, and drafts are not reviews. The report states the total number of Codex calls instead.

## Steps

1. **Argument parsing and frontmatter.** Edit `~/.claude/skills/cross-plan/SKILL.md` lines 1–14.
   - Add `[--single-draft]` to `argument-hint`.
   - Update `description` to say both models draft independently, then alternate reviews.
   - Parse `--single-draft` alongside `--max-rounds N` and strip it from the feature text.
   - *Verify:* re-read the frontmatter; YAML still parses (`python3 -c "import yaml,sys; yaml.safe_load(open(p).read().split('---')[1])"`).

2. **File list.** Edit lines 24–27 to add `DRAFT.claude.md`, `DRAFT.codex.md`, and `draft.codex.log` (Codex stdout/stderr, for diagnosing failures).
   - *Verify:* read the section.

3. **Replace "Round 1: Claude drafts" with "Round 0: independent drafts".** Edit lines 60–65 to add:
   - 3a. The Codex draft command. It is the same shape as the review command: `codex exec -s read-only -C "<root>" [--skip-git-repo-check] -o "<dir>/DRAFT.codex.md" "<draft prompt>" < /dev/null > "<dir>/draft.codex.log" 2>&1`, launched with `run_in_background: true` and a 600000 ms timeout. The draft prompt text is written out in full in the skill, as the review prompt is.
   - 3b. Claude's draft. Explore the repo as today, then write `DRAFT.claude.md`. An explicit instruction forbids reading `DRAFT.codex.md` or `draft.codex.log` before `DRAFT.claude.md` is saved.
   - 3c. Validation. Check for the six sections and apply the heading-level normalisation rule. On failure, retry once in the background. If it fails again, record the error and continue as `--single-draft`. Never write a Codex draft yourself.
   - 3d. Merge. Write `PLAN.v1.md` and the round-0 divergence table. Every path cited must exist, as today. Start `DISCUSSION.md` with the feature text, plus "v1 merged by Claude from independent Claude and Codex drafts", or "v1 drafted by Claude (single draft: <reason>)".
   - 3e. `--single-draft` skips 3a, 3c and the table; 3b's output becomes `PLAN.v1.md` directly.
   - *Verify:* read the section end to end, and check that every placeholder (`<root>`, `<dir>`) matches the names used in section 0.

4. **Codex's round-1 review prompt.** Edit lines 72–105: when `DRAFT.codex.md` exists and N = 1, append to the prompt: "Also read `<dir>/DRAFT.codex.md` (your independent draft) and the Round 0 table in DISCUSSION.md. If the merge dropped or distorted a substantive point from your draft, raise it as a REVISE issue citing the draft." Add the heading-level normalisation rule to the existing validity check for review files: the first line must be a verdict; if it is `REVISE`, the `## Revised plan` must contain the six sections.
   - *Verify:* read it; the prompt is still a single quoted argument, with no unescaped double quotes inside.

5. **Report.** Edit lines 154–159 to add:
   - whether two drafts were produced (or why not);
   - the number of round-0 divergences, and how many were taken from each draft;
   - total Codex calls (draft + reviews).
   - *Verify:* read the section.

6. **Sensitive-content note.** Add a short "Codex's view of the repo" paragraph to section 0. Codex reads files under `-C <root>` freely and sends them to a hosted model. If the repo holds data that must not leave the machine, run Codex with `-C` set to a directory without it, or ask the user first. *Verify:* read it. (Previous-run evidence is in Context.)

7. **Dry run.** Create a scratch git repo in the session scratchpad (a ~50-line Python module plus one test) and run `/cross-plan add a --verbose flag` with `--max-rounds 2`.
   - Check that both drafts exist, and that `DRAFT.claude.md`'s mtime precedes the first read of `DRAFT.codex.md` (Claude records the order in the round-0 section).
   - Check that the round-0 table is present, that `PLAN.v1.md` has the six sections, and that at least one review round ran with a valid verdict line.
   - Then run once with `--single-draft` and confirm that no Codex draft call happens.

## Testing

The skill is prose, so testing means the dry runs in step 7 plus these static checks:

- The frontmatter YAML parses.
- Every `<placeholder>` in the skill is defined in section 0.
- Every command in the skill starts with a prefix allowed by `allowed-tools`. `codex exec` is covered; the background run uses the same Bash prefix. No new shell tools are introduced, since validation uses Read/Grep.
- A fault-injection dry run: point `-o` at an unwritable path, or pass `-m nonexistent-model`, to confirm that the retry-then-fall-back-to-single-draft path runs and is recorded in `DISCUSSION.md`.
- A heading-normalisation check: hand-feed a draft whose sections are at `###` and confirm that only the heading levels change (diff shows heading lines only).

## Risks and open questions

1. **Cost and wall time.** Adding a draft call adds one Codex run (~5–10 minutes at high effort, per the previous run). The drafts run in parallel with Claude's exploration, so the added wall time is mostly the gap, if any, after Claude finishes. `--single-draft` is the escape hatch.
2. **Independence is only as good as instruction-following.** Claude could read the Codex draft early; nothing mechanical prevents it. Mitigations: the draft files are named distinctly, the skill orders the reads explicitly, and the recorded order in DISCUSSION.md is auditable. Open question: is that sufficient, or should the Codex draft go to a temp path that is moved into `<dir>` only after `DRAFT.claude.md` is written?
3. **Merge bias.** Claude merges and could favour its own draft. The round-1 Codex audit against its own draft is the counterweight. If the user wants symmetry, an alternative is to have Codex merge (it would output v1 via `-o`), at the cost of one more call.
4. **Placement of runs for skills outside a git repo.** Section 0 already handles this with `--skip-git-repo-check`. This run placed its output in the calling repo and pointed Codex at the skill directory; the skill does not say which to prefer when the feature's code lives outside the current repo. Open question for the user.
5. **The skill file is outside any git repo** (`git rev-parse` fails in `~/.claude/skills/cross-plan`). Take a backup copy (`SKILL.md.bak-<date>`) before editing so the change can be reverted.
