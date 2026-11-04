# Claude report-label pilot

**Completed:** 40 reports processed; 39 accepted and 1 rejected. Canonical outputs
are in `outputs/claude_label_pilot_v4/`. See the [result and baseline diagnostic](claude_label_pilot_result.md)
before interpreting the differences as accuracy gains.

The user chose their existing Claude API key after approving a $10 report-label
pilot. Claude is now the active provider; do not also run the OpenAI pilot.
The host's supplied clarification allows hosted inference subject to the stated
accessibility, minimal-cost, and other competition requirements.

## Prepared run

`outputs/claude_label_pilot_v4/` preserves the same 40 training studies, duplicate
components, report hashes, baseline targets, and blind-review packet as the
original pilot. Provider-specific request hashes are newly frozen. The original
OpenAI sources and experiment files remain intact.

The candidate is `claude-sonnet-5-5`, with adaptive thinking, low effort, a 6,000
output-token limit, global inference, and JSON Schema output. The shared rubric,
six assertion states, and exact-quotation checks are unchanged. This is a
candidate to evaluate, not a demonstrated accuracy improvement. See the
[model documentation](https://platform.claude.com/docs/en/models/sonnet-5-5/overview)
and [structured-output documentation](https://platform.claude.com/docs/en/build-with-claude/structured-outputs).

Published standard prices checked September 29: $2 per million input tokens and
$10 per million output tokens. No caching or tools are requested. The reservation
uses $4 per million input tokens to cover even one-hour cache-write pricing;
actual usage accounting includes cache writes and reads if returned. The 40
compact-schema requests reserve **$6.425536** conservatively against the **$10 cap**,
plus $0.376256 carried from two earlier rejected requests. This is not
a measured bill; invoice terms and taxes remain authoritative.
[Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing).

The same independent-review and image-model evaluation gates used by the
OpenAI pilot apply. Quotation matching does not verify meaning. Training labels are not
automatically replaced, and Kaggle remains user-managed.

## Credentials and execution

The runner reads literal settings from the environment, `.env`, or `claude.env`.
It never executes a credential file as shell code. Both files are Git-ignored;
`claude.env` has owner-only permissions. Do not print or commit its contents.

The user replaced the original key; access is verified and the replacement needs
no explicit workspace header. For a future multi-workspace key, set
`ANTHROPIC_WORKSPACE_ID=wrkspc_...` from Claude Console → Settings → Workspaces.
[Workspace authentication](https://platform.claude.com/docs/en/manage-claude/authentication#select-a-workspace).

```sh
.venv/bin/python tools/claude_label_pilot.py check-access
.venv/bin/python tools/claude_label_pilot.py run --pilot outputs/claude_label_pilot_v4 --max-cost-usd 10
```

The access check sends no reports. Paid requests are journalled before sending;
transport or accounting failures stop the run for review. Responses that have
valid usage but fail label validation are retained, charged to the cap, and
excluded from accepted labels while the remaining sample proceeds without a
retry. Both accepted and rejected requests are verified and skipped on resume.
Any original OpenAI
attempts block the Claude runner until the shared budget is reconciled.

Current verification: **41 local tests pass**. The completed pilot retains 40
API responses and two earlier HTTP-error reservations. A no-network replay
confirmed resume sends no duplicate requests. Improved extraction accuracy and
image-model accuracy have not been established; training labels are unchanged.
