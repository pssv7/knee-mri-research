# ConvNeXt-V2: execution readiness

## September 13 — full matched pair launched

The user authorized rebuilding and running this experiment after the protocol
ensemble scored 0.914. Restaged with the corrected adapter: scheduled
`faulthandler` dumps removed and direct competition mounts skipped during
input discovery. Both full arms and smoke retain the matched-backbone controls.
Training notebook and labels match the previously screened source hashes.

**Full pair version 1 is RUNNING:**
https://www.kaggle.com/code/pssv27/knee-convnextv2-fold1-pair

- Kernel ID 134222298, private T4, pinned image, all three full cache shards,
  labels and side-classifier inputs; internet enabled for pretrained weights.
- Nine-hour platform cap, 900-second smoke limit, 18000 seconds per full arm.
  Prior matched pairs took about five hours; V2 full runtime is unmeasured.
- 33 focused tests passed, including actual V1/V2 aggregation initialization.
  Pulled source bytes and metadata/input sets match staging exactly.
- Source SHA-256:
  `a91124550b1d5cd17ee09681babce7982354e8845666a581a3ac2d4aa2184557`.
  Receipt: `outputs/kaggle_convnextv2_fold1_receipt/`.
- Quota before launch: 34.83 GPU hours remaining. Startup smoke and full
  results are pending. This is a training experiment, not a leaderboard entry.

Both arms use uniform selection/shared pooling, 16 slices, five series,
12 epochs, seed 0 and effective batch 8. The baseline is freshly trained with
the same aggregation reseeding as V2. Gold remains diagnostic only. Evaluate
the same-fold standalone and fixed blend, then replicate on fold 3 if supported;
any later ensemble promotion must be compared with the **0.914** incumbent.
The staged-only/quota-blocked handoff below is historical.

## Smoke test passed

`pssv27/knee-convnextv2-fold1-smoke`, version 1, is COMPLETE, verified September
11, 2026. The log confirms pretrained V2 weights loaded, both T4 ranks started,
the short mixed-precision training returned, and selected/EMA/SWA weights were
exported. The job stopped after the smoke test; no full baseline or V2 training
ran. Eight train batches and eight validation batches over one epoch are an
execution check, not a meaningful AUC measurement or full-run runtime estimate.

Artifacts: `outputs/kaggle_convnextv2_fold1_smoke_results/`. Three checkpoint
sidecars/hashes and finite model tensors were checked. Uploaded source and
private GPU/image/input settings match the pulled receipt in
`outputs/kaggle_convnextv2_fold1_smoke_receipt/`. Source SHA-256:
`20c51d45f420d6b73bbd849c62ba28eda7a4ab4a7da6ae05efa5e100dbc00155`.

## Full comparison staged, not launched

The adapter supports the existing `convnextv2` preset. Both arms retain uniform
series selection, shared pooling, 16 slices, five series, 12 epochs, original
labels, seed 0 and native DDP. Only the backbone/pretraining variant changes.
Legacy ConvNeXt input normalization remains off and trunk LR scaling is 1.0;
normalization tuning would be a separate experiment.

The backbone is constructed before reseeding the aggregation layers in both
arms. Different trunk constructors otherwise consume different random draws.
The adapter requires identical initial weights outside `model.backbone.*`,
768-dimensional features and identical input defaults. A fresh matched baseline
is required; do not reuse an old baseline trained with the earlier RNG order.
Local tests exercise both actual V1/V2 backbone feature shapes and identical
seeded aggregation weights. The embedded notebook source is unchanged.

Stage command:

```bash
.venv/bin/python tools/prepare_kaggle_training_pair.py --candidate convnextv2 --fold 1
```

Full job: `outputs/kaggle_convnextv2_fold1/`, kernel name
`pssv27/knee-convnextv2-fold1-pair`, source SHA-256:
`3ef24cdfd51ef1df5b7ce3120d655297a1c41d38d2aee611d0aae9edc24bbab7`.
This is staged only. The smoke-test kernel is a different job and must never
be presented as the full experiment. Existing macro +0.003/all-finding -0.02
screening guards and fixed 50/50 reporting remain in force; gold is diagnostic.

## Runtime handoff

Kaggle reported 1.27 GPU hours remaining after the smoke test, refreshing at
2026-09-12 00:00 UTC (September 11, 20:00 America/New_York). A full matched pair
is expected to need roughly five hours based on previous pairs, with V2's full
runtime still unmeasured. Do not launch a job knowingly unable to finish.
The user has been asked to increase quota or supply another GPU environment.
No full job was launched and no future automatic start was scheduled.

Priority remains completing the already replicated protocol family; see
`protocol_incumbent.md`. The current public score is still 0.911.
