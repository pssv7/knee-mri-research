# Fold-1 protocol experiment

The user authorized matched baseline/protocol training after submission
56135074 scored 0.911, and explicitly removed GPU-hour scarcity as a development
constraint. Preserve the exact incumbent ensemble. No new competition submission
is part of this experiment.

**Latest status, 2026-09-10:** recovery **version 2 completed** at
https://www.kaggle.com/code/pssv27/knee-protocol-fold1-recovery.
Its two-GPU smoke test completed and exported selected, EMA and SWA weights.
Baseline recovery then passed the OOF reproduction check (tolerance 0.001),
and the full protocol run completed on two T4s after both ranks passed the
matched-settings/initialization guard. The screening gate passed.
No further approval is needed for the authorized development experiments;
use their measured results to determine subsequent runs.

Uploaded code SHA-256:
`bd101dfaae4b2007dbbd880e4efc29cb29b93af832f26ead1c1a2e3876e9c2fd`.
Pulled metadata and source in `outputs/kaggle_training_recovery_v2_receipt/`
confirm private visibility, both required datasets, all three cache shards,
the side predictions, and byte-identical source. Dataset input order is
server-sorted and has no significance. Five staging/recovery tests passed.

## Completed results and replication

**Replication update:** fold 3 completed and passed its screening gate. Full
results and gold diagnostic limitations are in `protocol_fold3.md`. The launch
status below is the historical receipt, not the current job state.

Downloaded artifacts: `outputs/kaggle_protocol_fold1_results/`. All four full-run
checkpoint hashes passed sidecar verification. The pair took 8,835 seconds
(2.45 hours), excluding the original baseline training. Independent local
recomputation is in `data/meta/compare_protocol_fold1/`.

| Mined-label AUC, 871 matched held-out studies | Baseline | Protocol | Fixed 50/50 blend |
| --- | ---: | ---: | ---: |
| Macro, all 12 findings | 0.871575 | 0.881118 | 0.885457 |
| PF OA | 0.787674 | 0.816353 | 0.816073 |
| Synovitis | 0.898999 | 0.892042 | 0.899631 |

Protocol macro delta is +0.009543, paired bootstrap 95% CI [0.002531, 0.017191]
(1,000 draws). No finding falls by more than 0.02, so the registered screening
gate passes. The fixed blend delta is +0.013882, CI [0.010040, 0.018111], and
improves all twelve pooled finding AUCs. This is not a leaderboard measurement.

The 58 image-labelled studies are diagnostic only, not used for selection.
Baseline/protocol/blend macro is 0.856530 / 0.854347 / 0.868809. PF OA rises
0.740026 to 0.774775, but its paired delta CI [-0.068216, 0.139177] is wide.
Synovitis pooled AUC falls 0.685783 to 0.660096; effusion-conditioned AUC falls
0.606 to 0.562, paired delta CI [-0.161, 0.076]. No demonstrated synovitis gain.

Important limitation: the recovered baseline trained with the notebook fork
launcher, while protocol used native script DDP. Initial weights, settings and
versions matched, but a stochastic execution difference remains possible.
Do not attribute the entire observed gain solely to the selector.

Fold-3 replication version 1 was pushed and verified RUNNING on September 10,
approximately 16:51 UTC: https://www.kaggle.com/code/pssv27/knee-protocol-fold3-pair.
Both baseline and protocol now train under native DDP, with 3,480 training and
869 held-out studies. The generated notebook and embedded training modules
remain unchanged. Seven staging tests pass, including both folds' source
isolation, fold membership guards and invalid-recovery rejection.

Stage with `.venv/bin/python tools/prepare_kaggle_training_pair.py --fold 3`.
Staged source: `outputs/kaggle_protocol_fold3/`; pulled upload receipt:
`outputs/kaggle_protocol_fold3_receipt/`. Byte-identical source SHA-256:
`1d73116c46779375d8f6762f8cd68f9e7928645c044bdb155cb716dc53eeee6d`.
Private visibility, GPU, pinned image and all attached inputs were checked.
Review both screening folds before remaining-fold training or model promotion.
No new competition entry was submitted; the incumbent remains 0.911.

## Baseline completed; paired job failed

`pssv27/knee-protocol-fold1-pair`, version 1, trained the baseline for all twelve
epochs on two T4 GPUs. The log reached `max_epochs=12` after 8,707 seconds, then
stalled before exporting EMA/SWA or returning from `trainer.fit`. The outer
runner timed out at 10,800 seconds. The protocol experiment never started.
The exact blocking call was not captured, so a fork/thread deadlock is a
hypothesis, not an established traceback diagnosis.

The selected Lightning checkpoint survived and was downloaded to
`outputs/kaggle_training_pair_failure_v1/baseline_fold1/lit_fold1.ckpt`.
It is 464,313,245 bytes; every state tensor is finite. It records epoch 8
(ninth epoch), global step 3,915, and best validation macro **0.871666**.
Gold macro at that epoch was **0.856530**, diagnostic only. Full epoch metrics,
initial-weight hash, input configuration and the failure log are preserved
beside it. The baseline does not need retraining.

## Recovery changes

- Script jobs use native `ddp`, with only rank zero exporting and scoring.
  The regular generated notebook remains unchanged. This avoids carrying the
  notebook's fork launcher into a stand-alone script.
- A short two-GPU test must finish `fit` and write selected, EMA and SWA weights
  before a full experiment runs. It is an execution test, not an AUC result.
- Recover the baseline checkpoint by SHA-256, reproduce all 871 OOF predictions,
  and require macro AUC within 0.001 of its saved selected-epoch validation AUC.
- Protocol training must match the baseline's initial weights, seed, labels,
  optimizer schedule, effective batch, GPU count and software versions.
- Input discovery skips the raw-DICOM competition directory. Searching that
  entire tree for a label file delayed startup and would repeat on each DDP rank.
- Stack dumps diagnose a stalled process. Each full training subprocess has
  a five-hour timeout; the Kaggle job has a nine-hour upper limit.

Kaggle rejected the failed paired kernel as an attachable input source. The
checkpoint and metrics were therefore uploaded as the private dataset
`pssv27/knee-fold1-recovery-private` (ID 11972406); privacy and all three files
were verified through the API. Recovery version 1 was launched before that
dataset attachment fix and can only exercise its smoke test before failing
the missing-checkpoint guard. Its smoke test passed. The corrected version 2
attaches the private recovery dataset and has passed recovery as recorded above.

Staging command:

```bash
.venv/bin/python tools/prepare_kaggle_training_pair.py \
  --recover-baseline outputs/kaggle_training_pair_failure_v1
```

The staged job is `outputs/kaggle_training_recovery/`. It writes separate
`baseline_fold1` and `protocol_fold1` directories and a final `comparison.json`
and `per_finding.csv`. Promotion screening requires candidate macro +0.003 and
no finding down more than 0.02; the regression guard applies to all twelve
findings. Also report the fixed 50/50 blend. Gold does not select anything.
