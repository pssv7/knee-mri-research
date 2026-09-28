# Protocol addition to the incumbent families

## Full five-fold result — September 12

All protocol training is complete, including fold 2 version 2 and fold 4
version 3. Fold 4 ran through 12 epochs and all exports in 2.57 hours after
removing the scheduled diagnostic stack dumps; no recurring dump appeared.

| Mined-label OOF macro | Incumbent-family proxy | With protocol | Delta |
| --- | ---: | ---: | ---: |
| Fold 0, 871 studies | 0.890112 | 0.895060 | +0.004947 |
| Fold 1, 871 studies | 0.884078 | 0.891892 | +0.007814 |
| Fold 2, 870 studies | 0.884560 | 0.890176 | +0.005616 |
| Fold 3, 869 studies | 0.883052 | 0.889416 | +0.006365 |
| Fold 4, 868 studies | 0.877970 | 0.883375 | +0.005405 |
| Pooled, 4349 studies | 0.876676 | 0.885228 | +0.008552 |

The predeclared equal 16/24/protocol family blend improves all five folds and
all twelve pooled finding AUCs. Paired within-fold bootstrap (1000 draws)
95% CI for the pooled delta: **[+0.006843, +0.010095]**. This holds fitted
models fixed and measures uncertainty against mined labels; it is not a
leaderboard-score forecast or proof of diagnostic improvement. Pooled AUC
also reflects cross-fold calibration and is not the mean of fold AUCs.

Verified all five selected protocol checkpoint checksums and 220 finite weight
tensors per model, exact saved folds and training exclusion, 4349 unique OOF
studies, finite probabilities, matching screened recipe/initialization and
incumbent checkpoint/source hashes. Legacy training exclusion retains the
provenance limitations below. Completion receipts and selected models for
folds 0/2/4 are in `outputs/kaggle_protocol_foldN_comparison_download/`.
Eleven focused comparison tests passed.

Reproduce:

```sh
.venv/bin/python tools/compare_protocol_incumbent.py --folds 0 1 2 3 4 --out data/meta/compare_protocol_incumbent_all_folds
```

That directory contains `summary.json` with hashes, `per_finding.csv` and both
aligned OOF prediction tables. The secondary 50/50 output is not a selected
recipe; no blend weights were tuned. Next is packaging and a runtime benchmark
of the fixed 15-model ensemble before a submission decision. No new submission
was made; the 0.911 incumbent remains unchanged.

## Earlier two-fold screening and execution history

The prior baseline/protocol pairs did not compare against the scored 16/24-slice
ensemble. The new reproducible check uses each study's own held-out fold model
from each family. It does not average predictions from other folds that trained
on that study. Legacy exclusion still relies on original training artifacts;
reconstructed fold IDs alone are not independent proof.

Command: `.venv/bin/python tools/compare_protocol_incumbent.py`.
Tool: `tools/compare_protocol_incumbent.py`; report:
`data/meta/compare_protocol_incumbent/`. Checkpoint/source hashes, prediction
hashes, exact fold membership and fixed 0.1 incumbent manifest weights were
verified and recorded. Recipe: equal 16/24-slice families versus equal
16/24-slice/protocol families. No weight search, no classwise addition.

| Mined-label OOF macro | Incumbent-family proxy | With protocol |
| --- | ---: | ---: |
| Fold 1, 871 studies | 0.884078 | 0.891892 |
| Fold 3, 869 studies | 0.883052 | 0.889416 |
| Pooled, 1,740 studies | 0.874456 | 0.885955 |

Pooled delta +0.011499, paired within-fold bootstrap 95% CI
[0.008596, 0.014766]. All twelve pooled finding AUCs improve, including PF OA
0.813950 to 0.825325 and synovitis 0.886092 to 0.894437. Pooled AUC is not an
average of fold AUCs; cross-fold score calibration can affect it. This is a
two-fold mined-label proxy, not a new public score or complete five-fold result.

## Remaining protocol folds

**September 12 update:** quota reset was verified at 21 hours. Fold 0 version 1
completed. Fold 2 version 1 passed smoke but its full subprocess timed out after
20 minutes and exited SIGSEGV during epoch 2, with no valid export. Fold 4
version 1 launched after fold 0 freed the second GPU slot and reports RUNNING;
its smoke/result is pending. Pulled source and private T4/input metadata match
staging. 38 focused tests passed. See `PROJECT.md` for the active handoff.
The fold-2 smoke exported finite checkpoints and the full run reached epoch 2;
logs contain no data, NaN, OOM or contract error. Treat it as a transient NCCL
distributed-runtime failure and retry the unchanged screened recipe once a GPU
slot is available.
The user then requested an immediate retry: fold 2 version 2 was pushed with
the unchanged screened script while fold 4 remained RUNNING, and Kaggle reports
the retry RUNNING. Its startup and smoke results are pending.
Fold 4 version 1 separately timed out in input discovery while recursively
walking its direct raw-DICOM competition mount. The staging finder now skips
roots containing the competition CSVs; fold 4 version 2 was relaunched with
that fix and reports RUNNING.

Stage with `.venv/bin/python tools/prepare_kaggle_protocol_completion.py`.
Folds 0, 2 and 4 are staged separately under
`outputs/kaggle_protocol_fold{0,2,4}_completion/`. Each job runs its own short
smoke test and one full protocol training, not another baseline pair. Jobs
have separate source/metadata/receipt directories. The original staging was
not a launch; current execution status is recorded above.

Staging refuses failed screening folds, changed labels, a changed generated
notebook or altered protocol preset. Runtime preflight matches the screened
contract and full initial weights, allowing only fold/step-count/warmup-step
differences. Each job verifies exact split IDs, finite OOF probabilities and
all three checkpoint sidecars. Core training modules remain unchanged.

After sufficient quota is available, complete these three models first. Then
run a full 4,349-study fixed-family OOF comparison and benchmark any proposed
15-model submission for inference time. No submission is pre-authorized by the
development request, and the 0.911 package must remain unchanged until promotion.
The separate V2 pair is staged as the next architecture test.

Verification before the smoke launch: 38 focused staging/comparison tests
passed. A broader test run was interrupted without its result being captured;
do not claim it passed from that invocation.
