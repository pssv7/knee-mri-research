# Opus 5.5 blind report review

The user requested Opus 5.5 to review the completed Sonnet labeling pilot. This
is a second, blinded model annotation of the original reports. It can reveal
disagreements and corroborate candidates; it cannot establish a human reference
standard or prove improved image-model accuracy. The [completed measured result](opus_blind_review_result.md)
records 40 accepted reports, 94.87% paired state agreement and $4.110458 shared
accounted cost.

## Frozen method

- Same 40 seeded-random training reports, one per duplicate component; gold,
  inner-selection and outer-held-out studies remain excluded.
- `claude-opus-5-5`, adaptive thinking, medium effort, 6,000 maximum output
  tokens, global routing. Anthropic documents medium as this model's default
  effort and supports structured JSON output.
- Same research rubric and twelve-finding schema as Sonnet. The only changes
  in the inference request are the model and effort settings.
- Each request contains one original report and the rubric/schema. It contains
  no Sonnet answers, old extractor outputs, soft targets, gold labels, study IDs,
  disagreement categories or prior validation outcomes.
- All 40 request hashes were frozen before the first Opus call. The report
  Sonnet failed to quote exactly is included, without telling Opus it failed.
- The report-level validator checks schema, finding coverage and exact evidence
  quotations. It can restore original whitespace only when alignment is unique;
  raw responses and each restoration remain auditable. Invalid responses are
  paid but excluded, without an automatic retry.
- Comparison remains sealed until all 40 review responses finish. Rejected
  responses contribute unavailable slots, never agreements or negative labels.

The evidence-span check detects fabricated quotations; it does not prove that
the quoted text supports the assigned finding. Both models share a provider and
rubric, so agreement can reflect shared mistakes. The rubric's unresolved
competition thresholds, particularly PF OA, remain unresolved.

## Spending and reproducibility

The shared **$10** cap includes the previous Sonnet pilot and conservative
reservations for its two earlier failed requests: **$1.467430** before Opus.
Every new call is gated by a conservative input/output reservation. Completed
calls release unused reservation using recorded API token usage. No extra
budget or full-dataset extraction is authorized by this review.

Frozen prices are $4 per million input tokens and $20 per million output tokens
for Opus 5.5, checked September 29, 2026. Token-cost accounting is not a provider
invoice. Sources: [Opus 5.5 model specification](https://platform.claude.com/docs/en/models/opus-5-5/overview),
[structured output support](https://platform.claude.com/docs/en/build-with-claude/structured-outputs).

The frozen manifest, raw responses, budget ledger and execution receipt are in
`outputs/opus_blind_review_v1/`. Its manifest's `api_calls_made: 0` describes
preparation, not the later execution; use `run_result.json` for execution status.

```sh
# Resume the authorized cohort only; completed responses are reused.
.venv/bin/python tools/opus_blind_review.py run --max-cost-usd 10

# Offline comparison; refuses an unfinished cohort and never calls an API.
.venv/bin/python tools/compare_opus_blind_review.py

.venv/bin/python -m pytest tests/test_compare_opus_blind_review.py tests/test_opus_blind_review.py tests/test_claude_label_pilot.py tests/test_openai_label_pilot.py -q
```

The comparison exports all 480 finding slots, accepted Opus assertions, missing
responses, state confusion counts, per-finding agreement, both-present versus
either-present counts, certainty/severity disagreements, and the disposition of
the 55 Sonnet-positive/low-soft-target cases. Cases where the old extractor was
already positive are distinguished from old silent/negative calls.

## Decision boundary

Keep model agreement, report-reference accuracy and downstream image-model AUC
as different measurements. This review automatically changes none of the
training labels. Use its disagreement queue for report adjudication, and audit
agreements as well to detect shared errors. Severity/fusion changes require a
separate candidate; mild positive reports need not have binary targets of one.
An image-model accuracy claim still requires a fixed training comparison and
untouched held-out evaluation. Kaggle remains managed by the user.
