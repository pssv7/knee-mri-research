# Protocol selection: fold-3 replication

## Completion and verification

`pssv27/knee-protocol-fold3-pair`, version 1, is COMPLETE, verified September
10, 2026 around 21:48 UTC. The paired job took 17,658.8 seconds (4.91 hours),
including smoke test, both training runs and exports. Both runs used native
DDP and returned successfully. The earlier shutdown failure did not recur.

Full downloaded artifacts: `outputs/kaggle_protocol_fold3_results/`.
The smaller early report download in `outputs/kaggle_protocol_fold3_report/`
is byte-identical to corresponding full-output files. Source receipt remains
`outputs/kaggle_protocol_fold3_receipt/`, as recorded in `protocol_fold1.md`.

Verified all six selected/EMA/SWA model SHA-256 sidecars, matched training
contracts and full initialization hashes, identical folds across both runs
and fold 1, 3,480 training IDs, 869 held-out IDs and no train/validation overlap.
OOF IDs match the held-out split exactly. Local recomputation reproduces every
field of the remote comparison. Results: `data/meta/compare_protocol_fold3/`.

Selected baseline SHA-256:
`05c47bf9e50474b885ca4e965a86a2a6013faea573252ea59e3f664ebd743828`.
Selected protocol SHA-256:
`347bb18b82d5259d65e3e80574ad4e07af5b9de9a126f46a135deb6b9ddf185f`.

## Mined-label validation

| Finding | Baseline | Protocol | Fixed 50/50 blend |
| --- | ---: | ---: | ---: |
| Macro | 0.871415 | 0.877224 | 0.886863 |
| ACL | 0.93003 | 0.92221 | 0.93691 |
| MCL | 0.91977 | 0.90290 | 0.92879 |
| Medial meniscus | 0.88230 | 0.87911 | 0.88989 |
| Lateral meniscus | 0.81288 | 0.83439 | 0.83672 |
| Medial OA | 0.90313 | 0.90057 | 0.90945 |
| Lateral OA | 0.85709 | 0.85301 | 0.86569 |
| PF OA | 0.80713 | 0.81623 | 0.82384 |
| Effusion | 0.87948 | 0.90816 | 0.90246 |
| Synovitis | 0.88152 | 0.89605 | 0.89653 |
| Baker's | 0.89600 | 0.90009 | 0.90805 |
| Contusion | 0.85114 | 0.86265 | 0.87091 |
| Fracture | 0.83652 | 0.85133 | 0.87313 |

Protocol macro delta +0.005810, paired bootstrap 95% CI [-0.003935, 0.016420].
Blend delta +0.015448, CI [0.009178, 0.022626]. The registered macro >= +0.003
and no-finding-drop >0.02 screen passes. It is an allocation rule, not a
significance test. The blend improves all twelve findings, as on fold 1.

Combined OOF across the two disjoint screening folds (1,740 studies) is in
`data/meta/compare_protocol_folds1_3/`: macro 0.869578 / 0.877965 / 0.885019.
Candidate delta CI [0.001429, 0.014822]; blend delta CI [0.010998, 0.019803].
Pooled cross-fold AUC is not the average of individual fold AUCs. These CIs
hold models fixed and exclude retraining variance; fold 1 retains its launcher
confound. None of these results measures the public leaderboard.

## Image-labelled diagnostic

The same 58 studies are diagnostic, not checkpoint or blend selection data.

| AUC | Baseline | Protocol | Fixed 50/50 blend |
| --- | ---: | ---: | ---: |
| Macro | 0.887565 | 0.870440 | 0.889122 |
| PF OA | 0.817246 | 0.773488 | 0.803089 |
| Synovitis pooled | 0.720430 | 0.689367 | 0.702509 |

PF OA protocol delta -0.043758, paired 95% CI [-0.105263, 0.009216].
Synovitis effusion-conditioned AUC is 0.660 baseline versus 0.593 protocol;
paired delta -0.066, CI [-0.185, 0.046]. Its pooled delta is -0.031,
CI [-0.121, 0.058]. These findings remain uncertain, with no demonstrated
image-label benefit. In particular, mined-label PF OA improvement on both
folds must not be presented as replicated image-label improvement.

## Decision

Protocol selection passes both practical screening folds and merits further
ensemble evaluation. The strongest evidence is complementary predictions in
the fixed blend, not standalone replacement. Gold macro for the blend is
roughly level with the fold-3 baseline; the small increase is not proof.
Do not tune per-finding weights or choose EMA/SWA using gold.

Keep the scored 0.911 ensemble unchanged. The independent classwise fold-1
pair remains RUNNING. Remaining-fold protocol training is eligible under the
screening plan, but no new job was launched during this result check. Before
promoting a new submission package, evaluate complete matched OOF against the
actual 16/24-slice incumbent and benchmark its inference runtime. The current
two-model blends are not comparisons against that ten-model incumbent.
