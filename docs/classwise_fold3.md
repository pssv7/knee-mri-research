# Classwise pooling: fold-3 replication

## Completion and verification

`pssv27/knee-classwise-fold3-pair`, version 1, is COMPLETE, verified September
11, 2026 around 13:30 UTC. Both training runs and exports completed in
18,148.5 seconds (5.04 hours), within the six-hour job limit.

Full artifacts: `outputs/kaggle_classwise_fold3_results/`. The small report
download in `outputs/kaggle_classwise_fold3_report/` matches corresponding full
output files. Source/upload receipt: `outputs/kaggle_classwise_fold3_receipt/`.
All six selected/EMA/SWA checkpoint hashes were verified. Training contracts,
shared initialization, identical split definitions, 3,480 training and 869
held-out IDs, train/validation exclusion and all prediction IDs were checked.
The baseline selected weights and OOF predictions are identical to those from
the protocol fold-3 pair. Local recomputation exactly reproduces the report.

Selected classwise SHA-256:
`eb51140b4c0863388faeecb2b29225db1a5dd25e7df584feb492a5f46ebe23bb`.
Selected baseline SHA-256:
`05c47bf9e50474b885ca4e965a86a2a6013faea573252ea59e3f664ebd743828`.

## Registered readout

Mined-label results on 869 matched held-out studies:

| AUC | Baseline | Classwise | Fixed 50/50 blend |
| --- | ---: | ---: | ---: |
| Macro | 0.871415 | 0.866763 | 0.882532 |
| ACL | 0.930029 | 0.874198 | 0.920038 |
| PF OA | 0.807131 | 0.790226 | 0.814640 |
| Synovitis | 0.881523 | 0.886796 | 0.893199 |

The standalone gate FAILED: macro delta -0.004651 and ACL delta -0.055831.
Its macro delta CI is [-0.013742, 0.005324]. The baseline/classwise blend gains
+0.011117, CI [0.005312, 0.017957], improving eleven findings but reducing ACL
by 0.009991. The all-twelve improvement seen on fold 1 did not repeat.
Full table and independent report: `data/meta/compare_classwise_fold3/`.

The separately predeclared ensemble replication also FAILED:

| Fixed ensemble | Macro AUC |
| --- | ---: |
| Baseline/protocol, equal halves | 0.886863 |
| Baseline/protocol/classwise, equal thirds | 0.889489 |

Incremental delta +0.002626 falls below the predeclared +0.003 requirement;
paired 95% CI [-0.000155, 0.006130] includes zero. No finding falls by more
than 0.02, but the macro condition is not met. The strongest added finding is
lateral meniscus (+0.015290); PF OA adds only +0.000412. Do not change the
threshold after seeing these numbers. Report:
`data/meta/compare_classwise_incremental_fold3/`.

The generic comparison tool also prints a secondary half-blend of the two
ensembles. It was not the registered recipe and is not a replacement decision
rule. No weight search or gold-based selection was performed. Bootstrap CIs
hold the fitted models fixed and exclude retraining variability.

## Image-labelled diagnostic

The same 58 image-labelled studies remain diagnostic only:

| AUC | Baseline | Classwise | Fixed 50/50 blend |
| --- | ---: | ---: | ---: |
| Macro | 0.887565 | 0.864273 | 0.891789 |
| PF OA | 0.817246 | 0.761905 | 0.808237 |
| Synovitis pooled | 0.720430 | 0.689367 | 0.713262 |

PF OA candidate delta -0.055341, paired CI [-0.143383, 0.021104].
Effusion-conditioned synovitis is 0.660 baseline versus 0.625 classwise,
paired delta -0.035, CI [-0.152, 0.074]. The earlier positive diagnostic point
estimates did not repeat. Neither finding has a demonstrated improvement.
The blend's slightly higher gold macro does not override the failed validation
gate or justify selecting a different blend.

## Decision

Archive classwise as a tested but unpromoted variant. Do not train its remaining
three folds or replace the current ensemble. The test found complementary
predictions, but not enough incremental gain to meet its replication criterion.

Prioritize the already replicated protocol-selection candidate for complete
OOF comparison against the actual 16/24-slice incumbent. ConvNeXt-V2 remains
the next planned controlled architecture experiment; keep pooling and input
settings fixed for that test. Neither a new training job nor a competition
submission was launched during this result check. The public score remains
0.911; the two- and three-model screening blends are not the ten-model incumbent.
