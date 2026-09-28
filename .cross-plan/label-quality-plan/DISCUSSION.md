# Discussion: label-quality-plan

Feature request: `docs/label_quality_plan.md this plan do not miss out anything`

Interpretation: produce an implementation plan that realises every requirement of
`docs/label_quality_plan.md` as written (the user's standing rule: follow a given plan;
flag departures rather than substitute).

v1 drafted by Claude.

### Round 1: Codex reviewed v1 → REVISE
| # | Point | Raised by | Resolution in v2 | Evidence |
|---|---|---|---|---|
| 1 | Training integration: PREFLIGHT hard-codes 4,349 and fold counts; seeds only 0/1; need fixed-fold loading, loader-level assertions, separate prediction path | Codex | Accepted (v2 step 19) | `tools/prepare_kaggle_training_pair.py:53-54,113` |
| 2 | Linked-group moves after sampling can put reviewed studies in locked and break inclusion probabilities; gold-linked groups uncovered | Codex | Accepted: components built before sampling (v2 steps 3–4) | v1 step 3 precedence rule |
| 3 | Overlap/dedupe can undercut 150; controls meeting suspect criteria must stay in control estimates; Clopper–Pearson invalid under weights/clustering; legacy selection provenance; pending reviews must be *completed* | Codex | Accepted (v2 steps 6, 9, 10) | `tools/build_label_review.py`; source §2 |
| 4 | `unclear` means unmentioned; no certainty field in dumps | Codex | Accepted (v2 step 6) | `src/labels/llm_extract.py` prompt; `llm_labels.parquet` columns |
| 5 | Extraction-validation sample reserved too late | Codex | Accepted: reserved at membership (v2 step 4) | source §5.2 |
| 6 | Power before evaluator; κ ≠ misclassification; cross-finding dependence; ≥100 pos/neg not enforced; Holm needs a test procedure | Codex | Accepted (v2 steps 12–15) | `tools/compare_oof.py::compare` |
| 7 | fold 1/3 receipt paths absent (fallback needed); outputs/ and parquet gitignored; rubric/manifest versioning | Codex | Accepted (v2 steps 1, 8) | `summary.json` receipts; `.gitignore` |

### Round 2: Claude reviewed v2 → REVISE
| # | Point | Raised by | Resolution in v3 | Evidence |
|---|---|---|---|---|
| R1.1–R1.7 | Round-1 points | Codex | settled (Claude verified each in code, no objection) | see Round 1 |
| 1 | Traceability table dropped | Claude | Restored, mapped to v2/v3 numbering | user request "do not miss out anything" |
| 2 | Seven error classes not enumerated | Claude | Named in step 6 | source §2 |
| 3 | No-averaging rule missing | Claude | Approach, step 11, test | source §3 |
| 4 | Locked-label exclusion list incomplete | Claude | Approach + per-entry-point test | source §3 leakage |
| 5 | Disagreement-only-for-selection missing | Claude | Approach, step 6 | source §2; `docs/label_review_v1.md` |
| 6 | Full-ensemble non-independence only implicit | Claude | Step 5 explicit | source §3 |
| 7 | Enriched panel pooling / distortion / full-review agreement stats | Claude | Step 14 | source §3 |
| 8 | Immediate next action missing | Claude | Goal | source closing paragraph |
| 9 | Human/authorization gates lost | Claude | [user]/[readers]/[custodian] tags | project practice (all runs user-launched) |
| 10 | Feasibility facts dropped (no PatientID in index; participant-only readers, 2026-10-15; untracked source doc; `detect_language`) | Claude | Context + Risks decisions | `index_train.parquet` columns; `data/dev` header; `docs/rules_notes.md` |

### Round 3: Codex reviewed v3 → APPROVE
| # | Point | Raised by | Resolution | Evidence |
|---|---|---|---|---|
| R2.1–R2.10 | Round-2 points | Claude | settled (Codex approved v3 without objection) | review.r3.codex.md |
| — | none | Codex | — | — |

**Converged** after 3 reviews: v3 written by Claude, approved unchanged by Codex.
