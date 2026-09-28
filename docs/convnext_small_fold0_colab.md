# ConvNeXt-Small, 224 pixels, nested fold 0 — completed; screen failed

## Result verified 2026-09-25

The user completed the Colab run and supplied seven result artifacts in Downloads.
All six receipt-listed non-checkpoint files match their recorded hashes; the
correct training log is `metrics_09:25.csv`, not the unrelated `metrics.csv`.
The completion bundle matches the locally generated bundle, every frozen spec
field matches, and the declared training-source/input configuration hashes agree.
Smoke and full training share initialization and runtime. All 691 outer prediction
IDs match their roles, with no train/selection overlap. Recomputed all metrics
and 2,000-draw bootstrap results exactly from saved predictions and frozen labels.

| Outer report-proxy metric | Result |
| --- | ---: |
| Historical Tiny macro | 0.878838 |
| Small macro | 0.877015 |
| Small minus Tiny | -0.001823 |
| Paired sampling 95% interval | [-0.011790, +0.007877] |
| Contusion delta | -0.058151 |
| Lateral meniscus delta | -0.026041 |

The macro gain gate fails, its interval crosses zero, and two findings breach
the -0.020 point floor. **Stop this Small/224 configuration; no expansion,
confirmation, blending or promotion follows this screen.** This is not proof
that every Small recipe is inferior: the comparator is historical, hardware
and torch differ, outer0 was previously inspected, and labels are report-derived.

All 12 epochs completed, with 3,312 optimizer steps logged. The fitting/export
stage reported 56.69 minutes on A100 40 GB / torch 2.11 (Tiny used T4 / torch
2.10). Inner validation peaked at epoch 10 (zero-based 9), 0.880218, then fell
to 0.877168 at epoch 12 while training loss continued to fall. These curves
do not support extending the same run just because Small is larger.

The selected checkpoint was not supplied. Its declared hash agrees between
the run contract and completion receipt, but its bytes and checkpoint epoch
have not been independently inspected here. Remote input bytes were not reread;
the known local-path input configuration reproduces the declared hash.

Preserved downloads and verification script/receipt:
`outputs/colab_small_224_outer0_results/`. Training was not rerun.

## Historical preparation

The user requested one Small fold-0 Colab notebook, using the existing Drive
shards, and explicitly excluded 384 pixels. No remote training was launched.

Open `notebooks/22_convnext_small_224_fold0_colab.ipynb` in Colab, choose one GPU
(A100 if available), and run the cells in order. The generator is
`tools/prepare_colab_small.py`; orchestration is in `tools/colab_small_runtime.py`.
The notebook is self-contained: repository upload is not required.

## Drive inputs and outputs

The old 08, 16, 19 and 20 Colab notebooks consistently use
`/content/drive/MyDrive/kagglehub` for the original 224 cache. The new notebook
uses that same path and searches its subdirectories for the three original
arrays and corresponding manifests. These are full **24 × 224 × 224 uint8**
caches, totaling 29.35 GB; the loader selects 16 slices. Thinned 8-slice exports
and mismatched files are rejected. Allow roughly 35 GB free VM-local disk.

Images are copied from Drive to local storage with pinned byte counts and
SHA-256 verification. Existing verified local copies are reused. Original
Drive files are not modified. Labels, roles and series metadata are embedded;
`side_predictions.parquet` is found in the cache tree or at optional `SIDE_FILE`.
If missing, only that small file is downloaded from the private side-classifier
output, using a hidden Kaggle credential prompt if needed. Packages and original
pretrained weights also require internet.

Outputs save directly under
`/content/drive/MyDrive/knee_runs/convnext_small_224_outer0_v1`.
Best and last Lightning checkpoints, selected weights, logs, contracts, outer
predictions, source snapshots and comparison receipts remain on Drive.
Completed runs are verified and skipped. Partial runs stop for explicit stage
recovery; a repeated cell does not silently restart completed fitting.

## Frozen experiment

- Backbone: `convnext_small.fb_in22k_ft_in1k`, original pretraining.
- 224 pixels, 16 slices, five series, protocol selection, legacy sampler.
- Seed 0, 12 epochs, effective batch eight, original OneCycle and learning rate.
- Same nested outer0 roles/labels as the sampler control: 2,208 train,
  553 inner-selection and 691 outer-evaluation studies. This is not the old
  4,349-study five-fold configuration. Gold and review exclusions remain intact.
- Smoke: two transient optimizer steps on full-five-series training studies,
  finite loss/gradient/parameter checks and checkpoint round-trip. Full fitting
  then starts in a new process from original pretraining and the same seed.
- Only inner selection chooses the checkpoint. Outer predictions follow fitting.

The original runner is adapted with checked substitutions in the generator;
historical source files and their hashes remain unchanged. A separate source
manifest binds the new runner and runtime. The original strict matched-pair
evaluator is not weakened to permit a different backbone or runtime.

The notebook compares against the verified historical Tiny checkpoint's saved
predictions (macro 0.878838). It explicitly records hardware/runtime differences
and reused outer0 as limitations: this is an exploratory comparison. Gate:
macro delta ≥0.005, no finding below −0.020, positive lower paired sampling
bound. Passing can motivate a separately proposed matched confirmation;
it cannot trigger training, promotion, blending or submission automatically.
No independent image-accuracy claim follows from report-proxy scores.

## Local validation

32 targeted tests passed across the Small notebook and fallback validation
modules. Checks include notebook schema and deterministic regeneration, every
embedded source/hash, real miniature-file preflight with Small accepted/Tiny
rejected, wrong-cache rejection and copy reuse, completion artifact tampering,
historical comparison safeguards, and actual two-step/reload smoke orchestration
using a tiny CPU surrogate. CUDA execution and real Small memory use are not
locally verified; the notebook runs that gate before full training.

## Configuration-cell recovery

The staging cell initially depended on the first configuration cell having run.
A reported `NameError: DRIVE_CACHE` was fixed in the generator: staging and
training now restore missing path defaults while preserving custom values.
Missing code/bootstrap state raises an actionable section-order message.
Three targeted checks passed for missing settings, preserved custom settings,
and regenerated notebook consistency. Existing runtime state and local shard
copies can be reused; no restart or training retry is required for this error.
