# Report-label audit: initial findings, September 15

## Decision

There is a concrete label-extraction problem worth addressing before another
architecture experiment. The first targeted full-report checks establish
wrong-structure attribution, missed negation, and degeneration treated as tears.
They do not establish how much leaderboard improvement fixing these would buy.
No production labels were changed and no training was launched.

## Reproducible selection

Run `.venv/bin/python tools/build_label_review.py`.

Artifacts are in `outputs/label_review_v1/`: `summary.json`,
`review_queue.jsonl`, and `review.html` (full reports with extractor evidence).
The development population is 2,609 studies / 31,308 study–finding pairs from
folds 0, 2, and 4. Folds 1 and 3 were excluded from report review.

Both the saved equal 16/24-slice blend and protocol predictor must oppose the
target's >=0.5 classification with probability >=0.7. This identifies 1,088
pairs. Select the strongest 200 deterministically and 50 random controls from
the remaining population with seed 20260915. This is a review-priority score,
not a calibrated probability that a label is wrong.

The 200 selected pairs comprise 184 possible false positives and 16 possible
missed positives; 118 carry weight 1.0, 60 weight 0.7, 20 weight 0.4, and two
weight 0.25. This enrichment is not an estimate of dataset-wide error rates.

Protocol OOF hashes match the earlier comparison receipts on all five folds.
Checkpoint sidecars declare exactly the expected held-out study IDs and no
overlap with training IDs. Legacy individual 16/24 OOF files are absent locally;
their saved blend is used, with declared fold membership checked. Independent
legacy training exclusion cannot be re-proven. These are two correlated sources,
not three independent votes. No averaged predictions were reverse-engineered
and represented as recovered individual models.

Folds 1 and 3 have already informed earlier project decisions; they are reserved
from this text review, not pristine test sets. Existing OOF teachers for development
folds can have trained on folds 1/3. Their scores must not become a fitted rule
for changing training targets; any subsequent correction must rest on report
evidence, with fixed evaluation and explicit limitations.

## Review status and concrete findings

All 250 saved categorical evidence entries were screened. Full source reports
were checked for 23 prioritized pairs (19 disagreement cases and four controls).
The remaining 227 full reports are pending. No overall reviewed error rate or
comparison with the random-control error rate is claimed.

Seventeen of those targeted checks have a report-grounding problem in a positive
training target. These are text-extraction judgments under the repository's
finding definitions, not new image-derived clinical ground truth.

| IDs | Finding | Report-grounding problem |
| --- | --- | --- |
| R039, R088 | Lateral OA | Lateral patellofemoral cartilage disease assigned to lateral tibiofemoral OA. R088 explicitly says femorotibial cartilage is normal. |
| R049, R058, R149, R193 | Meniscus | Report describes degeneration / grade-II intrasubstance signal; extraction calls a tear. |
| R125 | Medial Meniscus | Extractor quotes the sentence stating no tear, then calls it positive. |
| R127, R139 | Medial Meniscus | Cartilage findings used as evidence for a meniscal tear; report says menisci have no tear / no categorical tear criteria. |
| R168 | Medial Meniscus | Medial degeneration and lateral tears are conflated. |
| R195 | ACL | PCL tear assigned to ACL; the report says ACL has no abnormal findings. |
| R163 | MCL | Medial patellar retinacular injury assigned to MCL; collateral ligaments reported without abnormalities. |
| R150 | Contusion | Soft-tissue contusion used as evidence of bone contusion despite unremarkable bone structures. |
| R073, R121 | Synovitis | Periligamentous edema used as synovitis evidence; report does not establish synovitis. |
| R177 | Baker's | Cyst above the lateral collateral ligament labeled Baker's without supporting report identification. |
| R199 | Medial Meniscus | Extrusion/contusion used as evidence of a medial tear; explicit tear is lateral. |

Six other full-report checks are not safe corrections:

- R075: lateral PF disease plus nonspecific knee OA; lateral tibiofemoral
  involvement is not settled by the report.
- R155: ACL mucoid degeneration; definition/clinical significance requires
  adjudication, not an automatic negative.
- R204 and R245 (random controls): explicit tear text with final soft target
  around 0.42; below-0.5 does not by itself prove a bug because targets encode
  severity/certainty and fusion transforms.
- R215 (random control): effusion with target around 0.42; amount/threshold
  cannot be resolved safely from that alone.
- R219: categorical extractor falsely calls lateral meniscal abnormality a
  tear, but the final target is already approximately 0.013. Do not change a
  correct final negative to repair an intermediate output.

Many of the other screened disagreements explicitly quote real tears, fractures,
or cartilage damage. Low model scores are not sufficient grounds to delete them.

## Source of the problem

The existing `tools/audit_labels.py` reproduction check passes: all 104,376
target/weight values reproduce across 4,349 studies. This rules out accidental
label-file drift in this local audit; it does not validate the original extraction.
That command also runs an existing gold diagnostic; no gold-based thresholds or
new rules were selected from its output.

## Next bounded implementation

Build a separate evidence-grounded extraction candidate requiring: exact report
quotation, correct anatomical structure, explicit assertion/negation, and a
separate uncertainty flag. Start with meniscal tears and compartment-specific OA,
where several distinct attribution errors are already demonstrated. Keep the
current targets for ambiguous or unsupported cases until reviewed; silence must
not be turned into a confident negative. Save proposed corrections separately.

Finish the remaining full-report reviews and a blinded random-control review
before estimating error rates. Test the candidate on an additional development
sample that was not used to write the corrections. Only then stage a matched
224-pixel protocol fold-1 experiment, with validation labels fixed. Repeat on
fold 3 if the new evidence supports it. Do not claim mined-label AUC alone
proves improved image truth: repaired labels may disagree with noisy validation.

## Corrections to earlier advice

The 384 runs extend protocol selection. The available historical protocol OOF
references are approximately 0.881118 (fold 1) and 0.877224 (fold 3), not the
weaker uniform-selection baseline. Rounded new epoch scores 0.877/0.873 do not
show an improvement over that reference. Exact matched OOF comparison remains
necessary because metric computation and runtime differ.

At 14.1 minutes/epoch, three additional 12-epoch folds would take approximately
8.46 GPU-hours for epochs, plus overhead—not the previously stated 42 hours.
