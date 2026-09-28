VERDICT: REVISE

## Critique

1. **Independence is enforced only for Claude.** `/Users/pssv/PycharmProjects/knee_competition/.cross-plan/codex-drafts-own-plan/PLAN.v1.md`, Approach and step 3, explicitly prevent Claude from reading Codex’s draft, but give Codex only the feature, format, and path rule. Codex can discover `DRAFT.claude.md` during exploration, especially on a retry after Claude has finished. The existing exploration pattern in `/Users/pssv/.claude/skills/cross-plan/SKILL.md:62–65` provides no artifact exclusion. Both authors need explicit restrictions covering current-run artifacts and logs, including retry prompts, and testing must check both directions.

2. **Fallback can incorrectly activate the merge audit.** Plan step 4 tests whether `DRAFT.codex.md` exists. A nonzero exit or malformed output can leave a nonempty file, so step 3’s fallback can subsequently tell Codex to audit a merge and divergence table that never happened. `/Users/pssv/.claude/skills/cross-plan/SKILL.md:98–101` already distinguishes file existence from successful output. Gate the audit on a validated draft actually used in a merge, and test failure with a residual output file.

3. **The proposed sensitive-content mitigation is not an access boundary.** Plan step 6 suggests changing `-C` to a directory without sensitive data. The installed `codex exec --help` describes `-C` as selecting the working root and `-s` as the sandbox policy for model-generated commands; neither establishes confinement of reads to that directory. The current run’s `/Users/pssv/PycharmProjects/knee_competition/.cross-plan/codex-drafts-own-plan/DISCUSSION.md` likewise assumes this separation keeps Codex away from competition data. The skill must distinguish a working directory from enforced read isolation and carry actual content restrictions into draft and review prompts.

4. **Normalization conflicts with the unchanged-copy requirement.** Plan step 4 adds heading normalization but does not amend `/Users/pssv/.claude/skills/cross-plan/SKILL.md:102–105`, which requires copying the revised plan unchanged. The Approach also promises verbatim output while allowing edits. Define raw Codex artifacts as immutable and apply the documented heading-only exception to the extracted plan. This preserves provenance and removes conflicting implementation instructions.

## Revised plan

## Goal

Update `/Users/pssv/.claude/skills/cross-plan/SKILL.md` so Claude and Codex independently draft plans before Claude merges them and the existing alternating review loop begins.

Successful behavior includes:

- Independent drafts without either author reading the other’s draft before completion.
- A merged `PLAN.v1.md` and a recorded divergence table.
- Codex’s first review auditing the merge against its accepted draft.
- `--single-draft` and a recorded fallback after two failed draft attempts.
- Verified normal, single-draft, and failure paths.

## Context

The implementation is the 161-line `/Users/pssv/.claude/skills/cross-plan/SKILL.md`, outside a git repository.

Relevant locations are:

- Lines 1–14: frontmatter and argument parsing.
- Lines 16–38: setup, output files, and plan format.
- Lines 60–65: Claude-only drafting.
- Lines 72–105: Codex invocation, retry handling, and revised-plan extraction.
- Lines 128–159: stopping, finishing, and reporting.

The installed `codex exec --help` confirms `-s read-only`, `-C`, `-o`, `--skip-git-repo-check`, and writable-directory semantics for `--add-dir`. `-C` selects the working root; it is not a read-isolation guarantee.

The allowed earlier evidence confirms that `label-quality-plan/review.r1.codex.md` and `PLAN.v2.md` use `###` plan sections. Later versions use `##`. Runtime and configuration claims are not prerequisites for this change.

## Approach

Start Codex’s draft in the background, then have Claude explore and draft. Give both authors the same feature and applicable constraints, with explicit prohibitions against reading the other author’s draft or current-run planning artifacts before completing their own draft.

Codex remains read-only and returns its draft through `-o`. Preserve that raw output. Claude validates it, merges accepted content, and records substantive differences and their resolutions.

Track whether a valid Codex draft was actually merged. Only that state enables the first-review merge audit.

Retain existing review alternation, review limits, convergence rules, oscillation handling, and final reporting behavior, extending validation with an explicit heading-only normalization exception.

## Steps

1. **Back up and update arguments.** Before implementation, copy `SKILL.md` to a uniquely named backup. Add `[--single-draft]` to `argument-hint`, update the description, parse both flags, and reject an empty feature after stripping flags. Draft attempts do not count toward `--max-rounds`.

2. **Extend setup and artifact documentation.** Define `<root>` and `<dir>` explicitly. Retain existing output placement and collision avoidance. Add `DRAFT.claude.md`, raw `DRAFT.codex.md`, and per-attempt Codex draft logs/output names so retries cannot accidentally reuse stale results. Record draft status and Codex invocation count in `DISCUSSION.md`.

3. **Replace Claude-only drafting with round 0.** Write the complete Codex draft prompt into the skill. It must:
   - Include the feature, applicable constraints, and six required sections.
   - Require checking relevant code and permitting nonexistent paths only for proposed new files.
   - Prohibit reading current-run drafts, plans, discussion, and logs, including Claude’s draft.
   - Prohibit file modification and return only the complete draft.

   Launch the established `codex exec -s read-only -C "<root>" -o ...` command with stdin redirected from `/dev/null`, logs redirected to the attempt’s log, a 600000 ms timeout, and `run_in_background: true`. Add `--skip-git-repo-check` when needed.

   Claude then explores and saves `DRAFT.claude.md`, without opening Codex output or logs. Wait for completion notification before validating Codex’s result. Apply the same independence restrictions to retries.

4. **Define validation and fallback.** A draft succeeds only after successful process completion and nonempty output containing the six complete plan sections. Permit heading-level normalization when section boundaries are unambiguous; otherwise retry once using a fresh output path.

   Preserve raw outputs. Record any heading-only normalization when deriving a plan from them. After two failures, record the errors, mark the draft unusable, and use Claude’s draft as `PLAN.v1.md`. Residual files never imply success.

   With `--single-draft`, skip Codex drafting and merging entirely.

5. **Merge accepted drafts.** After both drafts are complete, Claude reads Codex’s accepted draft and writes `PLAN.v1.md`. Record substantive differences, including contributions present in only one draft, in:

   `# | Topic | Claude draft | Codex draft | Taken into v1 | Reason`

   Record whether each resolution adopts Claude’s proposal, Codex’s proposal, a combination, or neither. Identify v1 as Claude-authored and set merge status only after completing the merge and table.

6. **Extend review validation and the first-review audit.** When round 0 successfully merged both drafts, tell Codex’s first reviewer to read its accepted draft and the divergence table and raise substantive omissions or distortions.

   Validate the verdict and, for REVISE, the complete revised plan before accepting the output. Explicitly require the six `##` section headings in the prompt. Amend the unchanged-copy instruction to allow only documented heading-level normalization during extraction; preserve the original review file. Keep existing retry and stopping behavior.

7. **Document content boundaries accurately.** Explain that read-only prevents model-directed writes and that changing `-C` does not guarantee read confinement. Carry known content restrictions into every Codex prompt. When content must remain inaccessible, require an appropriately isolated environment containing only permitted material before launching Codex; changing working directories alone is insufficient.

8. **Extend reporting.** Report drafting mode or fallback reason, substantive divergence counts and resolution categories, and total Codex invocations including retries. Preserve existing final-plan and unresolved-item reporting.

## Testing

- Parse the updated frontmatter using an available YAML parser with an explicitly supplied filename.
- Check argument stripping, placeholder definitions, command quoting, and compatibility with `allowed-tools`.
- In a scratch repository, run the normal flow with `--max-rounds 2`; verify both drafts, the merge table, six-section v1, and at least one valid review.
- Inspect tool traces for both authors to verify neither reads the other’s draft before completing its own. Include a Codex retry after Claude’s draft exists; modification times alone do not establish read order.
- Run `--single-draft`; verify no draft invocation or merge audit.
- Exercise two draft failures that leave malformed, nonempty output. Verify fallback, recorded errors, and absence of a merge audit.
- Exercise a failed first attempt followed by success; verify only the successful output is merged.
- Feed a draft and a REVISE response with `###` section headings; verify raw artifacts remain unchanged and extracted plans differ only in permitted heading levels.
- Verify missing sections trigger retry and that malformed reviews retain the existing nonconverged failure behavior.
- Verify a non-git scratch directory uses `--skip-git-repo-check`.

## Risks and open questions

- An additional Codex invocation increases cost and potentially latency; parallel drafting and `--single-draft` mitigate this.
- Independence remains instruction-enforced unless filesystem isolation is added. Symmetric restrictions and trace inspection provide the initial safeguard.
- Claude controls the merge; the divergence table and Codex’s first-review audit expose omissions and disputed choices.
- Raw output preservation and explicit merge status are necessary to avoid misleading provenance and fallback behavior.
- Changing `-C` does not isolate sensitive content. Actual access restrictions depend on the execution environment.
- Keep existing output placement in this change. Broader support for separate code roots and artifact roots can be addressed independently.