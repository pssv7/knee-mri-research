# Claude report-label pilot result — 2026-09-29

The 40-report pilot completed with **Claude Sonnet 5.5, low effort**. It established
that hosted extraction is operational at low cost. It did **not** establish better
report-label accuracy or higher image-model accuracy.

## Measured outcome

| Measure | Result |
|---|---:|
| Training reports processed | 40 |
| Reports passing local validation | 39 |
| Reports rejected from accepted labels | 1 |
| Accepted finding assertions | 468 |
| Original whitespace restored in evidence quotes | 2 |
| Token-usage cost at the frozen public rates | $1.091174 |
| Conservative reservation retained for two earlier HTTP errors | $0.376256 |
| Total accounted against the $10 cap | $1.467430 |

The accepted count is a schema/evidence-validation result, **not 97.5% extraction
accuracy**. Exact quotations do not prove correct anatomy, negation, severity,
or clinical interpretation. The API invoice is authoritative for charges.

The rejected report changed `normal` to `normales` inside evidence for three OA
findings. All twelve labels from that report remain excluded. The two accepted
whitespace repairs changed no words, case, or punctuation; each had exactly one
matching span in the original report. Raw API responses and repair audits remain
available. The rejected report remains in the blind-review sample and must count
as an extraction failure or abstention in any accuracy comparison.

## What the baseline comparison actually shows

Screening against the previous soft targets flagged 72 findings in 33 accepted
reports: 55 Claude-positive findings below baseline target 0.5, one reverse
direction case, and 16 uncertain/unattributable assertions. These are review
candidates, not 72 proven label errors.

The cached categorical and probability extractor dumps were checked against
their previously frozen hashes before investigating the 55 positive candidates:

| Old categorical extractor call | Cases |
|---|---:|
| Already positive | 39 |
| Silent | 15 |
| Negative | 1 |

For Effusion specifically, **20 of 23** flagged positives were already positive
in the old categorical extractor. A stronger LLM cannot be credited with newly
detecting those mentions. Existing targets combine severity, certainty,
probability-mode scores, lexicon fusion, and distribution mapping. A target below
0.5 is not equivalent to a negative extraction; mild findings can receive low
targets by design. The [rubric](report_review_rubric.md#3-certainty-and-severity-fields)
explicitly preserves this distinction.

The 16 old-silent/negative cases are the more direct extraction-comparison
candidates. Whether Claude is correct still needs independent, blinded review.
Neither agreement between models nor this assistant's inspection is that reference.

## Next comparison

1. Independently annotate **all 40 reports / 480 finding slots**, using the blank
   review packet before opening either model's outputs. Include agreements,
   silence, uncertainty, and the rejected report; do not score only flagged cases.
2. Compare extraction states with the old categorical calls. Separately review
   the 39 already-positive cases to assess severity and target conversion. Do not
   replace low targets with 1 merely because a finding is present.
3. If a larger fresh evaluation supports an extraction improvement, test a new
   label version while holding the image model, split, initialization, and training
   schedule fixed. Test changes to severity/target conversion separately so their
   effect is not attributed to the LLM switch. The 40-report operational sample is
   too small for reliable rare-finding accuracy claims.

No production labels or Kaggle jobs changed. Forty-one local tests passed, and a
completed-run replay with network sending disabled verified that no requests were
resent. Frozen input hashes, request hashes, raw responses, and baseline labels
were verified during the final audit.

## Artifacts

The canonical result directory is `outputs/claude_label_pilot_v4/`:

- `result_summary.json`: counts, token usage, costs, and per-finding state counts.
- `baseline_diagnostic_summary.json`: old-call analysis with source hashes.
- `blind_review_reports.jsonl` and `blind_review.jsonl`: original reports and blank
  480-slot review packet; these contain no candidate or baseline answers.
- `assertions.jsonl`: the 468 accepted assertions.
- `model_comparison.jsonl` and `baseline_diagnostic_cases.jsonl`: unblinded review
  aids; open only after independent annotations are saved.
- `rejected_reports.jsonl`, `evidence_normalizations.jsonl`, and `attempts/`:
  failures, repairs, and full response/usage provenance.

Earlier versions retain the schema-compilation failures and continuation history.
Responses were reused across versions without paid re-extraction. The request
body stayed fixed after the compact-array schema was introduced in v2.
