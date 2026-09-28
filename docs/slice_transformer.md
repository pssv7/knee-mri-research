# Slice-sequence transformer experiment

User authorized implementation and Kaggle folds 1 and 3, with parent timeouts
removed, on September 13, 2026. Both version-1 jobs were pushed successfully:

- https://www.kaggle.com/code/pssv27/knee-slice-transformer-fold1-pair
- https://www.kaggle.com/code/pssv27/knee-slice-transformer-fold3-pair

Both reported RUNNING after launch. Pulled source bytes and private T4/image/input
metadata match staging exactly. Kernel IDs are 134267899 (fold 1) and 134267910
(fold 3). Startup log requests returned no summaries yet; smoke is unverified.

## Additional fold 2 on Colab

User offered an additional Colab GPU and existing Drive shards. Prepared
`notebooks/16_slice_transformer_fold2_colab.ipynb` using the same matched recipe
with 870 held-out studies. It is **prepared, not running**: browser control is
unavailable in this session (`CUA_REPL_ENABLED_SURFACES is required`).

Open the notebook in Colab, select one GPU, set `DRIVE_CACHE` to the full-shard
folder (default `/content/drive/MyDrive/kagglehub`), then run cells in order and
authorize Drive. Three unchanged full cache shards and matching manifests are
copied to VM-local disk with SHA-256 copy checks, shape/dtype/series coverage
validation. Labels must retain the same checksum. Only missing small metadata
may download from Kaggle. Competition CSVs have a separate input mount so the
raw-DICOM exclusion guard cannot hide the image cache.

Outputs: `/content/drive/MyDrive/knee_runs/slice_transformer_fold2_v1/attempt_*`.
The notebook saves checkpoints, metrics and logs directly to Drive, plus source,
input and GPU/runtime receipts. Each rerun starts a separate attempt; no exact
mid-fold resume is claimed. Both training arms run on the same Colab GPU with
effective batch 8. Runtime differences from Kaggle DDP are recorded; compare
against the local matched control. Parent smoke/full timeouts remain None;
Colab session limits still apply. No additional Kaggle job was launched for fold 2.
The adapter/regression suite passed 44 tests; all five final Colab-specific
tests passed, including mounted-cache visibility and nested output relocation.

### Colab metadata staging repair

The first live attempt copied all three image shards successfully, then found
multiple Drive `train.csv` files. Updated staging collapses byte-identical
metadata copies by SHA-256; conflicting CSV/side-prediction copies use the
official Kaggle source instead of selecting an arbitrary path. Labels select
the copy matching the pinned hash, or download and verify the official matrix.
Existing local shard copies are content-verified and reused, including copies
from the original staging cell that had not yet written the final receipt.
Free-space checks account only for missing or differently sized copies.

Seven Colab regression tests passed. Regenerated the fold-2 notebook and
`notebooks/colab_slice_staging_fix.py`, a standalone definitions-only module
for uploading to the existing runtime. Execute that module, rerun the staging
call, then run the original training cell. Keep the runtime and copied shards.

The live runtime then rejected symlinks under read-only `/kaggle/input`.
Colab mounts now use `/content/knee_slice_mounts`; the outer runner, both
embedded arms and smoke source all relocate their input paths to that root.
Kaggle jobs are unaffected. Eight Colab tests pass, including real input-finder
execution through the relocated mounts. For an already completed staging copy,
upload/execute the refreshed repair module and call `mount_inputs(LOCAL_INPUTS)`
directly, then run the training cell. Recopying or rehashing shards is unnecessary
for this mount-only repair; `input_receipt.json` was written before the failure.

## Implementation

Both control and candidate use `convnext_tiny.fb_in22k_ft_in1k`, protocol
series selection, 16 sampled slices, five series, 224-pixel inputs, original
labels, seed 0, effective batch 8, and the existing 12-epoch schedule. The
control directory is named `baseline_foldN`, but its selection is explicitly
**protocol**, unlike the earlier uniform baseline arms.

The candidate inserts `SliceSequenceEncoder` before slice attention pooling:
feature LayerNorm, projection from 768 to 256, learned projection of normalized
slice positions [-1, 1], two independently initialized pre-norm transformer
layers (8 heads, feedforward width 512, dropout 0.2), and a residual projection
back to 768. The output projection starts at zero, reproducing the control's
initial evaluation predictions. Initial updates train that projection, after
which gradients reach the transformer. Positions describe sampled slice order,
not physical distance; no new DICOM geometry or higher-resolution data is used.

Existing modules initialize before the added branch, whose constructor
preserves CPU RNG state. The full CNN and all shared aggregation weights must
have identical initial hashes across arms. Run contracts, fold memberships,
label checksums, and checkpoint sidecars are validated before comparison.
Checkpoint reconstruction detects the encoder from its state keys; disabled
models gain no state keys and existing experiment dictionaries remain unchanged.

Each job runs one short GPU smoke (8 train/validation batches, one epoch),
then the fresh control and candidate sequentially. `--no-timeout` sets both
parent subprocess timeouts to `None`. Kaggle's nine-hour platform limit still
applies. No automatic retries or subsequent folds are scheduled. GPU quota
before these launches was 19.97 hours remaining; full runtime is unmeasured.

## Verification and provenance

213 tests passed across slice-transformer behavior, matched staging, network,
Lightning training, artifact/input pipeline, and protocol completion. Tests
cover shared initial weights, identity initialization, nonzero learned context,
position sensitivity, finite gradients, variable depths, masked series,
strict reload, input-policy drift rejection, and timeout wiring.

All six embedded training/smoke scripts parse and match current model source.
Notebook SHA-256:
`a22cf2da497bbc9fb2a2adf48e829712cf519062a3857ead8253392986f6b1ff`.
Labels SHA-256:
`e54e46531aec6cd16131d70aa283c990542dfa17daa890ac54da663cf74569b0`.

| Fold | Source SHA-256 |
| --- | --- |
| 1 | `6820b0cde207a143d15960eeeeb6298c5dbacb39b6a9bbefddec9b7bd83e0db7` |
| 3 | `c4a53c37144e6c96613dff0547a7a3c952572bc9014e344b927827d8fcdb71b2` |

Staging: `outputs/kaggle_slice_transformer_foldN/`.
Pulled receipts: `outputs/kaggle_slice_transformer_foldN_receipt/`.

## Evaluation

The original standalone screening gate is macro delta >= +0.003 and no
finding delta below -0.02, repeated on both folds. Report fixed 50/50 blending
separately. A successful blend with the control does not establish improvement
over the 0.914 incumbent: evaluate matched, held-out incumbent-family OOF
predictions before any expansion/promotion. Gold-58 is diagnostic only.
Training and smoke results remain pending until remote logs establish them.

## Correction to the preceding V2 handoff

V2 fold 1 ended with `TimeoutExpired` after 18000 seconds in the V2 arm,
during epoch 12. This is incomplete execution, not a completed failed metric
gate. Fold 3 completed: baseline 0.8703229314, V2 0.8697720418, fixed blend
0.8821408674; standalone gate failed, including ACL delta -0.04129.
The earlier pasted 0.866/0.856 retired-reference tables were baseline gold
diagnostics, not V2 validation scores. No V2 retry is included in this launch.
