# Private ensemble benchmark: 2026-09-09

One private T4 run completed; no retraining and no competition submission.
Kaggle runtime limit was 3,600 seconds, with a 3,500-second internal watchdog.
Both dataset and kernel visibility were verified private; internet was disabled.

- [Benchmark, version 1](https://www.kaggle.com/code/pssv27/knee-ensemble-private-benchmark)
- [Private ensemble dataset, version 1](https://www.kaggle.com/datasets/pssv27/knee-ensemble-16-24-private)
- Local outputs: `outputs/kaggle_benchmark_results/`
- Downloaded privacy/input receipts: `outputs/kaggle_upload_receipt/` and
  `outputs/kaggle_benchmark_receipt/`.

## Fold-1 identity

`experiment /oof_fold1.parquet` is a valid 871-study fold-1 file, but **not the
predictions of the selected 24-slice checkpoint at its intended input geometry**.
Against newly recovered predictions its maximum absolute probability difference
is 0.86979 and mean absolute difference is 0.09207. Its macro AUC is 0.87246.
Its actual source run is still unknown; the file was preserved unchanged.

The selected checkpoint is epoch 6, with recorded validation macro 0.876742.
Its recovered macro is **0.876751**, consistent with small numerical differences.
Eight existing fold-0 reference studies reproduced within maximum difference
0.000443, checking the cached-input path against that family's saved OOF.

All 871 fold-1 predictions were recovered and copied to:
`experiment /slice_24/fold1/oof_fold1.parquet`, with a `.parquet.json` sidecar.
The source checkpoint SHA-256 was verified against the local ensemble member:
`868475f12e7152a774396d87d4ba21bf94e2ab98314263c39dbbfcdd3987abcd`.
Legacy fold membership is reconstructed; this does not independently prove
that the original training excluded every held-out study.

## Full OOF comparison

All 4,349 studies, all five folds, same label matrix and fixed probability weights:

| Model | Pooled macro AUC |
|---|---:|
| 16 slices | 0.865320 |
| 24 slices | 0.865688 |
| Fixed 50/50 blend | **0.876676** |

The blend improves every fold and all twelve pooled finding AUCs. Its delta
versus 16 slices is +0.011356, paired 95% bootstrap CI [+0.009224, +0.013609]
using 1,000 within-fold study resamples. Detailed outputs are in
`data/meta/compare_24slice_complete/`.

These are report-mined-label proxy measurements, not a leaderboard score.
Bootstrap intervals condition on fitted models and do not cover label bias or
retraining variability. Each OOF study uses its held-out model from each family;
test inference averages all ten models, so the OOF delta is not a direct estimate
of the ten-model ensemble's leaderboard gain. No weights were tuned on gold.

## GPU runtime

| Measurement | Result |
|---|---:|
| End-to-end benchmark script | 973.70 s (16.23 min) |
| Setup | 277.35 s |
| Fold-1 recovery | 235.81 s |
| Raw-DICOM studies completed | 100 / 100 |
| Errors | 0 |
| Mean per raw study | 4.50 s |
| 95th percentile | 7.66 s |
| Maximum | 13.10 s |
| Peak PyTorch GPU allocation | 2.50 GiB |
| Projection for 1,300 studies, including setup | **1.70 h** |

Sampling: the five studies with the most series plus 95 seeded random studies.
All ten members, full DICOM decoding, side classification, and the submission's
explicit-ensemble prediction path were used. No synthetic-image timing and no
duplicated A/B inference. The projection is well below the 6.3-hour screening
threshold that reserves 30% of the nine-hour limit, but hidden-test composition,
CPU allocation and I/O contention can differ. Peak allocated memory is not
the same as total device usage or PyTorch reserved memory.

## Decision

The fixed blend passes the local diversity and runtime checks and is the next
competition-submission candidate. The user subsequently authorized one entry;
see [submission 56135074](submission_16_24.md), pending as of 2026-09-10 02:07 UTC.
The approved single private benchmark has finished. The gold holdout diagnostic
has not been rerun. No claim of 0.954, or any new leaderboard score, is justified.
