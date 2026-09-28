# Finding-specific pooling: fold 1

**Replication update:** fold 3 is COMPLETE and failed both the standalone and
predeclared incremental-ensemble gates. Do not expand classwise to remaining
folds. See `classwise_fold3.md`; the fold-3 launch status below is historical.

**Latest status, September 11:** version 1 is COMPLETE. Both runs and exports
finished in 17,907.9 seconds (4.97 hours). The standalone screening gate FAILED:
macro delta +0.002474 is below the registered +0.003 requirement. The fixed
blend nevertheless improved every mined-label finding, motivating an explicitly
ensemble-focused fold-3 repeat rather than relabelling the standalone gate.

## Completed results

Artifacts: `outputs/kaggle_classwise_fold1_results/`; early small reports:
`outputs/kaggle_classwise_fold1_report/`. Local comparison:
`data/meta/compare_classwise_fold1/`. Six checkpoint hashes, matched contracts,
exact train/validation exclusion and all 871 OOF rows were checked. Local
recomputation reproduces the remote report. The baseline selected checkpoint
and OOF predictions are identical to the recovered protocol fold-1 baseline;
the changed launcher did not change that baseline's measured outputs.

Selected classwise SHA-256:
`1220e4f5f8577c1e75babff0e147bfe63394d56183864fe9352665d6ec172ba1`.

| Mined-label AUC | Baseline | Classwise | Fixed 50/50 blend |
| --- | ---: | ---: | ---: |
| Macro | 0.871575 | 0.874049 | 0.883031 |
| PF OA | 0.787674 | 0.785673 | 0.801474 |
| Synovitis | 0.898999 | 0.893979 | 0.905744 |
| Contusion | 0.820338 | 0.862910 | 0.854342 |

Standalone delta CI [-0.005712, 0.010830]; blend delta +0.011456,
CI [0.006676, 0.016151]. The blend improves all twelve findings. Neither
standalone PF OA nor standalone synovitis improves against mined labels.

Gold-58 diagnostic macro: baseline 0.856530, classwise 0.866667, blend 0.883020.
PF OA: 0.740026 / 0.763192 / 0.773488, candidate delta CI
[-0.085859, 0.130574]. Synovitis pooled: 0.685783 / 0.704898 / 0.706093.
Effusion-conditioned synovitis: baseline 0.606, classwise 0.652, paired delta
+0.045 with CI [-0.055, 0.149]. These are encouraging point estimates, not
established diagnostic gains. Gold did not select models or blend weights.

An exploratory equal-thirds baseline/protocol/classwise blend scored 0.890261,
versus 0.885457 for the fixed baseline/protocol blend: delta +0.004804,
CI [0.001123, 0.008469]. No weight search was performed. Report:
`data/meta/compare_classwise_incremental_fold1/`. The comparison tool also
prints a secondary half-blend of these ensembles; it is not a selected recipe.
These are single-fold proxies, not comparisons against the ten-model 0.911
incumbent. Bootstrap intervals hold fitted models fixed.

## Replication decision

Repeat baseline/classwise on fold 3 under the existing development authorization
because of complementary validation predictions, not because the standalone
gate passed (it did not), and not because of gold selection. Before launch,
freeze the follow-up readout: report the same standalone gate, the fixed 50/50
baseline/classwise blend, and the same equal-thirds blend against the fixed
baseline/protocol blend. For the incremental ensemble screen, require macro
delta >= 0.003 with no finding down >0.02 on fold 3; report paired uncertainty
regardless of the gate. No per-finding weights, no further blend sweep, and no
remaining-fold classwise expansion without reviewing this replication.

The staged fold-3 source is `outputs/kaggle_classwise_fold3/`, SHA-256:
`b4196e0a714ed79c6f2bd23c317120df434fe0acb8779ba367f2086a5dd8393d`.
Version 1 was pushed and verified RUNNING September 11 around 02:59 UTC:
https://www.kaggle.com/code/pssv27/knee-classwise-fold3-pair.
Pulled receipt `outputs/kaggle_classwise_fold3_receipt/` verifies byte-identical
source, private GPU execution, pinned image and all inputs. Thirteen staging
tests passed again. The remote job has a six-hour limit; the two completed
pairs took approximately five hours each. Quota at launch was 6.34 hours.
No new competition submission. Preserve the exact 0.911 incumbent.

## Execution receipt

The user authorized continued development without a GPU-hour optimization
constraint. Version 1 was pushed and verified RUNNING on September 10, 2026,
around 21:30 UTC: https://www.kaggle.com/code/pssv27/knee-classwise-fold1-pair.
No competition submission is part of this experiment. Preserve the 0.911 entry.

Stage command:

```bash
.venv/bin/python tools/prepare_kaggle_training_pair.py --candidate classwise --fold 1
```

Staged job: `outputs/kaggle_classwise_fold1/`. Uploaded source and metadata were
pulled to `outputs/kaggle_classwise_fold1_receipt/`. Private visibility, GPU,
pinned Docker image, labels, three cache shards and side predictions match.
Byte-identical source SHA-256:
`e9a493f4acfdcedae086cdd9f90ede18046be62415da16c46b2e49bab7c47d60`.

## Controlled comparison

The job first exercises the classwise model in a two-GPU end-of-fit/export smoke
test, then trains a fresh baseline and the classwise candidate sequentially.
Both use native script DDP, seed 0, 12 epochs, 16 slices, five series, uniform
series selection and the unchanged labels. There are 3,478 training studies,
871 held-out studies and 58 image-labelled diagnostic studies.

Only study-level pooling changes. The candidate gives each finding its own
series weights; slice attention remains shared. This does not combine protocol
selection with classwise pooling, and does not claim a measured PF OA or
synovitis improvement before the run finishes.

Whole-model initial hashes differ when architecture changes. Both runs instead
verify identical upstream weights, excluding `model.study_pool.*` and
`model.head.*`, whose initialization consumes different random draws. The
full initial hashes are still recorded. Protocol comparisons retain the stricter
whole-model equality check. All other contract fields and the one-variable
experiment specification must match. The embedded repo modules are unchanged.

Thirteen local staging/contract tests pass, including actual shared model-state
equality and rejection of parameter or initialization drift. The 210 focused
pipeline, model, notebook and synovitis-diagnostic tests also passed before launch.

## Readout

The job exports `baseline_fold1/`, `classwise_fold1/`, `comparison.json`,
`per_finding.csv` and `pair_progress.json`. Use all 871 matched OOF studies.
Candidate screening requires macro delta >= 0.003 and no finding down > 0.02;
the guard conservatively covers all twelve findings. Also report a fixed 50/50
blend. Repeat a promising result on fold 3 before remaining-fold training.

Inspect PF OA and lateral meniscus explicitly. For synovitis, report both pooled
AUC and effusion-conditioned image-label AUC with paired uncertainty. Gold-58
remains diagnostic: never select epochs, blend weights or per-finding members
using it. A null result would motivate another controlled test, not dropping a
required output or assuming an established performance ceiling.
