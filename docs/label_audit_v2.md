# Report audit v2: Lateral Meniscus, Contusion, PF OA

Generated 2026-09-23T22:31 UTC by `tools/audit_report.py`. Counts, rates and intervals only; no report text. One reviewer throughout: agreement is consistency, not independence.

## Decision (D5)

**GO** — PF OA (pass 2 ran without the 7-day delay; see the D6 limitation below)

Go needs 5+ target errors agreed by both passes among a finding's original random controls. Weight-only changes do not count.

Agreed errors among original controls, by class: Lateral Meniscus: severity_definition 1; PF OA: missed_mention 5, severity_definition 6.

## Random controls (population estimates)

| Finding | Reviewed | Label judged incorrect | Missed positive | False positive | No judgement from text | Agreed target errors | Awaiting pass 2 | Decidable |
|---|---|---|---|---|---|---|---|---|
| Lateral Meniscus | 50/50 | 2/50 = 4.0% [1.1%, 13.5%] | 1/50 = 2.0% [0.4%, 10.5%] | 1/50 = 2.0% [0.4%, 10.5%] | 5 | 1 | 0 | yes |
| Contusion | 50/50 | 1/50 = 2.0% [0.4%, 10.5%] | 1/50 = 2.0% [0.4%, 10.5%] | 0/50 = 0.0% [0.0%, 7.1%] | 24 | 0 | 0 | yes |
| PF OA | 50/50 | 11/50 = 22.0% [12.8%, 35.2%] | 11/50 = 22.0% [12.8%, 35.2%] | 0/50 = 0.0% [0.0%, 7.1%] | 10 | 11 | 0 | yes |

95% Wilson intervals. The first-pass "incorrect" rate counts every reading the rubric judges against the final label; only agreed target errors can become corrections.

### Enlarged controls (D4)

- PF OA: 23/100 = 23.0% [15.8%, 32.2%]; complete.

## Enriched strata (selected for looking wrong; never a rate)

| Finding and stratum | Reviewed | Label judged incorrect | Agreed target errors | From label_review_v1 |
|---|---|---|---|---|
| Lateral Meniscus · suspected_false_positive | 50 | 5 | 1 | 29 |
| Lateral Meniscus · suspected_missed_positive | 11 | 4 | 3 | 0 |
| Contusion · suspected_false_positive | 50 | 19 | 3 | 4 |
| Contusion · suspected_missed_positive | 23 | 8 | 2 | 0 |
| PF OA · suspected_false_positive | 50 | 2 | 0 | 12 |
| PF OA · suspected_missed_positive | 50 | 34 | 19 | 0 |

## Error classes, ranked by agreed target errors

- PF OA | missed_mention: 24 agreed, 25 in pass 1
- PF OA | severity_definition: 12 agreed, 18 in pass 1
- Lateral Meniscus | severity_definition: 5 agreed, 8 in pass 1
- Contusion | anatomy: 2 agreed, 6 in pass 1
- Contusion | missed_mention: 2 agreed, 2 in pass 1
- Contusion | negation: 1 agreed, 9 in pass 1
- Contusion | unresolved: 0 agreed, 36 in pass 1
- PF OA | negation: 0 agreed, 22 in pass 1
- Lateral Meniscus | negation: 0 agreed, 20 in pass 1
- PF OA | fusion_weighting: 0 agreed, 20 in pass 1
- Contusion | fusion_weighting: 0 agreed, 18 in pass 1
- PF OA | unresolved: 0 agreed, 18 in pass 1
- Lateral Meniscus | unresolved: 0 agreed, 12 in pass 1
- Contusion | severity_definition: 0 agreed, 9 in pass 1
- Lateral Meniscus | fusion_weighting: 0 agreed, 5 in pass 1

Missed mentions by language (pass 1): Contusion | el: 1, Contusion | en: 1, PF OA | bg: 1, PF OA | de: 1, PF OA | el: 2, PF OA | en: 7, PF OA | es: 1, PF OA | hr: 1, PF OA | tr: 1, PF OA | unknown: 6

## Assertion classes (pass 1)

- Contusion | control | negated: 20
- Contusion | control | present: 6
- Contusion | control | uncertain: 8
- Contusion | control | unmentioned: 16
- Contusion | enriched | negated: 20
- Contusion | enriched | present: 24
- Contusion | enriched | uncertain: 24
- Contusion | enriched | unmentioned: 5
- Lateral Meniscus | control | negated: 38
- Lateral Meniscus | control | present: 4
- Lateral Meniscus | control | present_below_threshold: 3
- Lateral Meniscus | control | uncertain: 1
- Lateral Meniscus | control | unmentioned: 4
- Lateral Meniscus | enriched | negated: 5
- Lateral Meniscus | enriched | present: 42
- Lateral Meniscus | enriched | present_below_threshold: 2
- Lateral Meniscus | enriched | uncertain: 7
- Lateral Meniscus | enriched | unmentioned: 5
- PF OA | control | negated: 15
- PF OA | control | present: 24
- PF OA | control | present_below_threshold: 1
- PF OA | control | uncertain: 4
- PF OA | control | unmentioned: 6
- PF OA | enriched | negated: 6
- PF OA | enriched | present: 74
- PF OA | enriched | uncertain: 10
- PF OA | enriched | unmentioned: 10
- PF OA | extra_control | negated: 18
- PF OA | extra_control | present: 19
- PF OA | extra_control | present_below_threshold: 1
- PF OA | extra_control | uncertain: 1
- PF OA | extra_control | unmentioned: 9

## Agreement between passes

- Random controls (150 pairs): reading 91.3%, κ 0.87; disposition 97.3%, κ 0.84.
- Extra controls, D4 (48 pairs): reading 95.8%, κ 0.94.
- Selectively re-reviewed proposed changes: 28/28 agreed.

The two lines are reported apart on purpose: controls are a random sample, proposed changes were re-reviewed because the first pass flagged them.

## Unresolved and not acted on

- unresolved 66, report–image disagreement 0, proposed in one pass but not agreed 7.

## Adaptive sampling

- Contusion: first assessment 1/50, width 0.101 (no extra batch).
- Lateral Meniscus: first assessment 2/50, width 0.124 (no extra batch).
- PF OA: first assessment 11/50, width 0.225 (extra batch due).
- PF OA: 50 extra controls drawn, 2 reusing an enriched review. The go/no-go stays on the original controls.

**Limitation (D6):** the 7-day delay before pass 2 was waived for 226 re-reviewed pairs. Their agreement partly reflects recall of the first reading, which inflates agreed corrections and biases the decision toward go.

Limits: the enriched strata are selected by model disagreement and describe mechanisms, not prevalence. Prior exposure is known only from label_review_v1 and the gold 58. Secondary workstream: 183 pending v1 checks, reported separately when done.

## Coverage of reviewed pairs

- language: en 154, es 55, unknown 43, hr 35, tr 35, el 25, bg 15, de 15, nl 6, fr 1
- scanner: Siemens Healthineers|MAGNETOM Vida|3.0 34, SIEMENS|Aera|1.5 31, Siemens Healthineers|MAGNETOM Avanto fit|1.5 24, GE MEDICAL SYSTEMS|SIGNA Artist|1.5 22, Philips Medical Systems|Ingenia|3.0 21, GE MEDICAL SYSTEMS|SIGNA EXCITE|1.5 18, SIEMENS|SonataVision|1.5 17, TOSHIBA|Vantage|nan 16, Philips Medical Systems|Achieva dStream|1.5 13, Philips|Ingenia Ambition X|1.5 12, Philips|Achieva dStream|3.0 11, GE MEDICAL SYSTEMS|SIGNA Explorer|1.5 11
- weight_bucket: 0.7 154, 1.0 141, 0.25 76, 0.4 13
- extractor_call: yes 242, unclear_or_missing 94, no 48
