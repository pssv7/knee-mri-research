# Fixed 15-model protocol ensemble benchmark

**Subsequent result:** the user authorized one competition entry. Submission
56196386, notebook version 1, completed at **0.914**, verified September 13.
This is now the incumbent; see `submission_protocol.md`. Staging-only status
below records the earlier benchmark handoff.

The user authorized packaging and runtime benchmarking after all protocol folds
completed. The fixed recipe is five selected 16-slice models, five selected
24-slice models and five selected protocol models, with weight 1/15 each.
No EMA/SWA duplicates, classwise addition, test-time augmentation or weight tuning.

## Local package and verification

- Package: `outputs/ensemble_16_24_protocol/ensemble.json`.
- Private upload staging: `outputs/kaggle_protocol_ensemble_upload/`.
- Benchmark staging: `outputs/kaggle_protocol_ensemble_benchmark/`.
- Reproduce: `.venv/bin/python tools/prepare_protocol_benchmark.py`.
- All 15 model hashes bind to the full OOF comparison receipts. All 18 embedded
  notebook modules match workspace source. The scored incumbent is unchanged.
- Fifteen package/comparison/benchmark tests passed. The actual notebook's
  explicit-ensemble path ran end to end on the local DICOM fixture, including
  its intentional missing-imaging fallback. This is plumbing validation, not
  GPU runtime evidence.

## Benchmark design

Private T4 script, internet disabled, pinned Kaggle image, one-hour platform cap
and 3500-second internal stop. Attached inputs are the three full cache shards,
side classifier, labels, competition and the private ensemble dataset.

Before timing, reproduce saved OOF predictions for eight series-rich held-out
studies per model (120 member/study checks). Require each maximum absolute
probability difference to be below 0.01. This exercises the actual selection,
geometry and checkpoint loading used by the submission inference functions.

Time all 15 models on 100 raw-DICOM studies: the five largest studies plus 95
seeded random studies, identical to the incumbent benchmark sample. Require
finite probabilities, zero errors, all 100 studies completed and projection
for 1300 studies below 6.3 hours. Projection includes setup and reserves 30%
of the nine-hour submission cap. Hidden-test composition and I/O can differ.

The benchmark writes predictions and a report; it does not create a competition
entry. Current public-score incumbent remains 0.911.

## Launch

Dataset version 1 is ready and verified private (dataset ID 12000725):
https://www.kaggle.com/datasets/pssv27/knee-ensemble-protocol-private

Benchmark version 1 completed with a 3600-second cap:
https://www.kaggle.com/code/pssv27/knee-protocol-ensemble-benchmark

Source/metadata receipt: `outputs/kaggle_protocol_ensemble_benchmark_receipt/`.
Pulled source and all private/offline/image/input settings matched staging.
Script SHA-256: `cfe8d5bbf04141a17d7860f178cf2589413b5324dd551a0797c5844e52418311`.

## Completed GPU results

| Measurement | Result |
| --- | ---: |
| Selected models verified | 15/15 |
| Cached OOF reproduction | 8 held-out studies per model, all passed |
| Largest legacy probability difference | 0.000454187 |
| Protocol reproduction difference | 0 for each of five models |
| Raw studies completed / errors | 100 / 0 |
| Mean raw-study runtime | 5.283 s |
| 95th percentile / maximum | 8.121 s / 13.991 s |
| Peak GPU allocation / reserved | 3.223 GiB / 4.262 GiB |
| Setup time | 62.865 s |
| Total benchmark runtime | 630.164 s (10.50 min) |
| Projection for 1300 studies including setup | 1.925 h |

PASS against the predeclared 6.3-hour screening threshold. Sample IDs exactly
match the previous incumbent benchmark, which averaged 4.496 s/study. The new
ensemble adds about 0.787 s/study (17.5%). Setup differs across runs; the 1.925 h
projection is an estimate, not a guarantee for the hidden set.

Outputs: `outputs/kaggle_protocol_ensemble_benchmark_results/`. Downloaded
report, 100 unique finite prediction rows and sample identity independently
verified. Manifest SHA-256:
`dbb09326c7876930ddd6ae65a0dd2f678179c8260d8d02d08c4ac6e9a4cf0469`.

## Submission readiness

`tools/prepare_protocol_submission.py` generated
`outputs/kaggle_protocol_submission/submission.ipynb`, with the same inference
source and exact 15 model hashes/weights. It uses offline T4 execution and only
competition data, labels, the private package and the side classifier; cache
notebooks are removed. Eleven guard tests passed, covering incomplete runs,
wrong manifests/checkpoints, failed reference checks and excessive runtime.

The notebook is staged locally only. No submission kernel run or competition
entry was launched. This measured package is ready for a competition submission
decision; the verified public-score incumbent remains 0.911.
