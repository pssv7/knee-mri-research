# Opus 5.5 blind review — measured result

**Completed:** all 40 original reports reviewed with `claude-opus-5-5`, medium
effort. All responses passed the structural and literal-evidence validator,
yielding 480 assertions. Sonnet outputs and old labels were withheld from Opus.
The requests were frozen before inference and the comparison opened only after
all 40 responses finished. See the [method](opus_blind_review.md).

## What the review established

| Measurement | Result |
|---|---:|
| Opus accepted / rejected reports | 40 / 0 |
| Distinct provider responses | 40 |
| Comparable finding slots with accepted Sonnet and Opus outputs | 468 of 480 |
| Exact six-state agreements | **444 / 468 (94.87%)** |
| State disagreements | **24 / 468 (5.13%)** |
| Both models present / either model present, comparable slots | **91 / 97 (93.81%)** |
| Certainty disagreements, comparable slots | 19 |
| Severity disagreements when both models say present | 5 |
| Original-whitespace restorations, with raw-response audit | 5 |

The remaining 12 comparison slots belong to the report Sonnet's evidence
validator rejected. Opus produced accepted assertions for it. Those slots are
unavailable for pairwise agreement, not agreements or Sonnet-negative labels.
Evidence validation checks quotation integrity, not whether a quotation
semantically supports the assigned finding.

The 24 state disagreements across 14 studies include ten Sonnet-unmentioned → Opus-negated cases,
six cases crossing into or out of `present`, and eight other state changes.
The offline queue contains 45 rows across 19 studies: state, certainty or
severity differences, plus the 12 slots without an accepted Sonnet annotation.
Reasons overlap; they must not be summed as independent errors.

## The suspected missed positives

Of the 55 Sonnet-present findings with old fused targets below 0.5:

| Old categorical extractor | Cases | Opus present | Opus uncertain | Opus negated |
|---|---:|---:|---:|---:|
| Already positive | 39 | 37 | 2 | 0 |
| Silent | 15 | 14 | 0 | 1 |
| Negative | 1 | 1 | 0 | 0 |
| Total | 55 | 52 | 2 | 1 |

Thus Opus corroborates **15 of 16** potential missed positive calls (15 findings
across nine studies), separately
from the 39 cases already called positive by the old extractor. These remain
candidate corrections, not verified errors or evidence that all low soft targets
should become one. Severity/probability/fusion explains a different class of
low targets and requires a separate comparison.

Concrete observations from the recorded outputs:

- Both models assign all three OA compartments to the report phrase
  “Incipient OA of all three compartments”; the old categorical extractor was
  silent for those three slots. This is a useful compartment-attribution
  candidate for report adjudication.
- Sonnet calls “Trace joint fluid.” present for Effusion; Opus calls it negated.
  This is the one old-silent candidate Opus does not corroborate. It needs
  full-report and threshold review; model size does not settle it.
- Sonnet calls the Spanish phrase beginning “Probable rotura periférica del
  menisco medial” present; Opus preserves uncertainty. The fixed rubric
  explicitly requires hedged assertions to remain uncertain. This provides a
  concrete prompt/validation audit case without needing to infer an image label.
- Ten silent-versus-negated differences expose a separate question about the
  scope of blanket normal statements. Opus's negative calls are not automatically
  preferable; check their scope against the original report and rubric.

## Per-finding agreement

Each row has 39 comparable studies. The positive column is the number called
present by both models divided by the number called present by either.

| Finding | Exact state agreement | Both/either present |
|---|---:|---:|
| ACL | 37/39 | 5/6 |
| MCL | 38/39 | 5/6 |
| Medial Meniscus | 35/39 | 11/12 |
| Lateral Meniscus | 38/39 | 3/3 |
| Medial OA | 37/39 | 11/12 |
| Lateral OA | 39/39 | 4/4 |
| PF OA | 37/39 | 12/13 |
| Effusion | 38/39 | 24/25 |
| Synovitis | 37/39 | 2/2 |
| Baker's | 39/39 | 9/9 |
| Contusion | 36/39 | 5/5 |
| Fracture | 33/39 | 0/0 — no positive comparison |

The few positive examples for Lateral Meniscus, Contusion and other rare
findings cannot establish reliable class-specific accuracy. Overall agreement
also includes 222 jointly negated and 113 jointly unmentioned slots.

## Cost and verification

Opus used **316,897 input tokens and 68,772 output tokens**, with no cache tokens.
At the frozen rates, its token cost was **$2.643028**. The shared accounted total
is **$4.110458 / $10**, comprising $1.091174 Sonnet token cost, $2.643028 Opus token
cost and $0.376256 conservatively reserved for two earlier failed requests. This
is usage-based accounting plus reservations, not a provider invoice.

All 54 targeted offline tests passed. A completed-run replay with a sender that
raises on any network call verified no further inference requests. All frozen
request/source/input hashes and original extractor dump hashes verified. No
production training labels changed, and Kaggle was left user-managed.

## What should happen next

1. Adjudicate the 24 state disagreements and 15 corroborated missed-positive
   candidates against full original reports and the frozen rubric. Also audit
   model agreements; the same provider and rubric can produce shared errors.
2. Keep report presence, severity and certainty separate. Prepare any extraction
   repair separately from severity/fusion changes, and retain unresolved
   competition thresholds instead of silently relabeling them.
3. A claim that a label change improves the image model still needs a fixed
   training comparison and untouched held-out evaluation. These results are
   model agreement, **not 94.87% accuracy and not proof of an AUC gain**.

The review is complete; those subsequent steps were not performed by this run.

## Artifacts

- [Machine-readable comparison summary](../outputs/opus_blind_review_v1/comparison_summary.json)
- [All 480 comparison slots](../outputs/opus_blind_review_v1/all_comparisons.jsonl)
- [Adjudication queue](../outputs/opus_blind_review_v1/adjudication_queue.jsonl)
- [Old-extractor candidate review](../outputs/opus_blind_review_v1/baseline_candidate_review.jsonl)
- [Accepted Opus assertions](../outputs/opus_blind_review_v1/assertions.jsonl)
- [Frozen manifest](../outputs/opus_blind_review_v1/pilot.json)
- [Execution result](../outputs/opus_blind_review_v1/run_result.json)

Model specifications and pricing: [Anthropic Opus 5.5](https://platform.claude.com/docs/en/models/opus-5-5/overview).
