# Discussion: label-quality-solo

Feature request: `docs/label_quality_plan.md use the cross-plan skill on this codex written plan and count this as first version of codex and write this plan for only one person no one will be joining my team`

v1 merged by Claude from independent Claude and Codex drafts, with two departures from the skill, both by user instruction or unavoidable:
- **Codex draft supplied by the user.** `DRAFT.codex.md` is a byte-identical copy of `docs/label_quality_plan.md`, which the user identifies as Codex-written. No Codex draft call was made in this run. It is not in the six-section format and was accepted as-is by user instruction.
- **Claude's draft is not independent.** Claude had read `docs/label_quality_plan.md` in depth earlier in this session before drafting.

Codex invocations so far: 0.

`<restrictions>` for every Codex call: competition data must not reach a hosted model (`docs/rules_notes.md`). Codex runs with `-C` on a redacted copy (scratchpad `knee-sanitized/`) holding `src/`, `tools/`, `tests/`, `docs/*.md` (minus `docs/worklog.md`) and the requirements files, with the `EVIDENCE` report excerpts in `tools/prepare_reviewed_labels.py` replaced by `<redacted report excerpt>`. `data/`, `outputs/`, `notebooks/`, `PROJECT.md` and `docs/worklog.md` are absent. The prompt also forbids reading them in the real repo. This is instruction-plus-copy isolation, not an enforced boundary.

### Round 0: drafts merged
| # | Topic | Claude draft | Codex draft | Taken into v1 | Reason |
|---|---|---|---|---|---|
| 1 | Staffing | One person | Two MSK readers, adjudicator, custodian, second reviewer | Claude | User: no one will join the team |
| 2 | Independent image reference (1,000 studies, 300 dev / 700 locked, 100 pilot) | Not executable | Core of stages 3–4 | neither (recorded as not executable) | Requires readers |
| 3 | Locked evaluation and custodian | Kaggle private LB, registered before submission, read once after close | Custodian-held locked image labels | Claude | The only independent, tamper-proof evaluation available to one person |
| 4 | Second reviewer | Delayed blind self re-review; a correction needs both passes | Independent second reviewer before adjudication | Claude (adapted) | No second person; disclosed as consistency, not independence |
| 5 | Original-language expertise | Local open-weight translation aid; quotes in the original language | Language experts | Claude (adapted) | No experts; hosted translation forbidden by the rules |
| 6 | Report audit design (150 per finding, controls, strata, ledger, error classes, blinding) | Kept | Specified | Codex | Executable solo; kept as written |
| 7 | Extraction-validation sample | Reserved before any review (40 per finding) | Fresh, independently reviewed sample | combined | Fresh sample kept; reviewer not independent |
| 8 | Stop rule before the audit | ≥ 5/50 controls with an agreed fixable error in some priority finding | none | Claude | Solo time is the binding cost; label repair's LB record |
| 9 | Power and the confirmation gate | MDE ≈ 0.01 from the fold-1 bootstrap; +0.005 not confirmable; per-finding/noninferiority/multiplicity not executable | Simulation; +0.005 gate; −0.02 noninferiority; multiplicity | combined | The source's own "limit the claim" rule applied to the LB |
| 10 | Patient linkage and the expanded leakage manifest | Not executable / not needed | Required | neither (recorded) | No reserved training subset; test disjoint by design |
| 11 | Matched pair | Five folds, fresh control, seed 0 plus a fold-1 seed-1 screen | Matched pair on remaining population | combined | Same principle; population unchanged |
| 12 | Report-text handling | No console printing; local only | Blinded packets | combined | Solo work runs through an assistant; protects against exposure |
| 13 | Timeline | Dated stages with hard cutoffs before 2026-10-22 | No dates | Claude | Final submission deadline |

Merge status: merged

### Round 1: Codex reviewed v1 → REVISE
| # | Point | Raised by | Resolution in v2 | Evidence |
|---|---|---|---|---|
| 1 | Protocol submission helper hard-codes the 15-member ensemble; needs five-member packaging and benchmarks | Codex | Accepted (v2 step 11) | `tools/prepare_protocol_submission.py:16,24,41,75,77` |
| 2 | Validation reserve must be study-level and exclude previously inspected reports | Codex | Accepted (v2 step 3) | `tools/build_label_review.py` stores full reports; source §2/§5.2 |
| 3 | Corpus-wide rules vs the two-pass requirement; weight changes; delay on extra controls | Codex | Accepted: every applied change needs two passes (v2 steps 5, 7) | `apply_reviewed` / `prepare_reviewed_labels.build` are per-case |
| 4 | Pair adapter doesn't enforce matched contracts (same `baseline` experiment); verify exact fold membership | Codex | Accepted (v2 step 10); Claude notes the fold split uses fixed `splits.SEED=0`, not the training seed | `prepare_kaggle_training_pair.py:100`; notebook 08 `make_folds(labels, groups=groups)` |

### Round 2: Claude reviewed v2 → REVISE
| # | Point | Raised by | Resolution in v3 | Evidence |
|---|---|---|---|---|
| R1.1–R1.4 | Round-1 points | Codex | settled (verified; the R1.4 remedy stands) | see Round 1 |
| 1 | Status table moved out of the plan | Claude | Restored in-plan, renumbered | round-0 row; user's "do not miss out anything" |
| 2 | Rejected alternatives dropped | Claude | Restored | plan format |
| 3 | Critical path infeasible (approvals ≥ Oct 17 vs submission Oct 20) | Claude | Rule drafting starts on first passes; dated hard cutoffs | v2 step 7 text; ~30 GPU-h (`docs/protocol_fold3.md` 4.91 h/pair) |
| 4 | Serial fold-1 screen doesn't fit the window | Claude | Parallel; the screen gates submission | same |
| 5 | Private-score availability checked after compute | Claude | Moved to step 1 as a user decision | v2 step 12 |

Codex invocations so far: 1.

### Round 3: Codex reviewed v3 → APPROVE
| # | Point | Raised by | Resolution | Evidence |
|---|---|---|---|---|
| R2.1–R2.5 | Round-2 points | Claude | settled (Codex approved v3 without objection) | review.r3.codex.md |

Codex invocations: 2 (both reviews; the Codex draft was user-supplied). Data audit: both review logs show no command touching the real repo outside `.cross-plan/`; the only data-path mentions are Codex's own exclusion globs inside the redacted copy.

**Converged** after 2 review rounds plus Claude's review: v3 written by Claude, approved unchanged by Codex.
