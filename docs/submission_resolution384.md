# 384-pixel five-fold submission — submitted 2026-09-23 UTC

All five completed Colab runs are organized in `outputs/resolution384_downloads/fold{0..4}/`.
The original folder structure below each attempt is preserved (`full`, `smoke`, receipts).
Only selected checkpoints and small audit files were extracted because local disk space
was limited. EMA/SWA and Lightning checkpoint binaries remain in the original Downloads
ZIPs; those archives must be retained if those exports are needed later.

The earlier `fold3-20260922T163223Z-1-001.zip` contains an incomplete run and is excluded.
Fold 3 uses the completed `attempt_43974f31232c` from the later September-23 UTC archive.
The original successful fold 1 is `attempt_b5e73e547b5a`.

## Prepared artifacts

- `outputs/ensemble_resolution384/`: upload this folder as the private model dataset
  `pssv27/knee-resolution384-five-fold-private`. Five selected `.pt` files, their
  hash-bound `.pt.json` sidecars, `ensemble.json`, and `dataset-metadata.json`.
  Checkpoints are hard-linked to the organized runs to avoid duplicate storage.
- `outputs/kaggle_resolution384_submission/submission.ipynb`: standalone offline
  inference notebook, five folds at weight 0.2 each. No incumbent mixing, EMA/SWA
  selection, rank averaging, test-time flipping, or training.
- `outputs/kaggle_resolution384_submission/kernel-metadata.json`: staged private
  GPU kernel metadata. Attach the new model dataset, `pssv27/knee-labels`,
  `pssv27/knee-side-classifier` output, and the competition data.
- `outputs/resolution384_downloads/import_receipt.json`: archives, attempts,
  selected checkpoint hashes, and common training contract.
- `outputs/resolution384_downloads/oof_all_folds.parquet` and `oof_metrics.json`:
  combined held-out predictions and diagnostic mined-label AUCs.

The notebook embeds current project inference modules without changing the incumbent
notebook. It builds raw-DICOM volumes at 384 for the knee classifiers and separately
at 224 for SideNet preprocessing (which then downsamples to its 112-pixel input).
It checks the exact manifest and all five checkpoint hashes. The final validation
refuses a run containing fallback studies, invalid probabilities, or incorrect IDs/columns.

## Verification

- Complete, non-smoke execution receipts for all five runs.
- Extracted receipt hashes and selected checkpoint sidecar hashes match.
- Matching training contracts and original input receipts across folds.
- Exact train/validation partition membership, no overlap, 4,349 OOF studies covered
  exactly once; all checkpoint tensors and predictions finite.
- Local label file hash matches training provenance.
- Pooled mined-label OOF macro AUC: **0.873693**. Fold AUCs 0–4:
  0.890694, 0.877286, 0.877528, 0.873398, 0.873993. This is a diagnostic proxy,
  not a leaderboard score or evidence that this ensemble beats the incumbent.
- Actual staged notebook executed on CPU using all five real checkpoints and a
  synthetic single study constructed from local development DICOMs: zero fallbacks,
  valid CSV, exact 224 preprocessing, and learned side-classification path checked.
  Receipt: `outputs/kaggle_resolution384_submission/local_smoke_report.json`.
  This small test does not establish GPU performance or held-out prediction reproduction.
- Related pytest run: **24 passed, 3 failed**. The three failures are cache-building
  tests in `tests/test_resolution.py` that encountered the existing 2-GiB free-space
  reserve guard. No production cache was built or changed.

## Execution history (pre-launch handoff)

The initial local handoff, now completed, was to upload the model folder as a
**private** Kaggle dataset, then run the prepared notebook with the attached inputs
listed above. Review `submission_report.json`: five correct hashes, weights 0.2,
zero fallbacks, and the expected visible-test rows. The GPU benchmark below subsequently measured memory and representative raw-DICOM
runtime and reproduced the saved held-out predictions. Actual hidden-test runtime
remains unknown. The incumbent 0.914 artifacts remain unchanged.

Reproduce local preparation with `.venv/bin/python tools/prepare_resolution_submission.py`.
Run the actual-weight local smoke with `.venv/bin/python tools/check_resolution_submission.py`.
The smoke CSV uses synthetic IDs and is intentionally not packaged as a competition CSV.

## Authorized execution — 2026-09-22 evening

User authorized project cleanup, Kaggle GPU execution and submission. Removed only
two reproducible random smoke checkpoints under `outputs/spatial_smoke/` (about
303 MiB), retaining their metrics and sidecars. All 27 related tests now pass after
cleanup. Private model upload started. Prepared `outputs/kaggle_resolution384_benchmark/`
for 100 fixed studies (20 held out per fold), raw-DICOM runtime, and saved-OOF
prediction reproduction with an absolute-probability tolerance of 0.01.

Private dataset ID 12144825 is ready and independently verified private. Both
GPU jobs launched as version 1; pulled source cells and private/offline input
settings match staging exactly. Visible-test notebook completed on Tesla T4 in
53.77 seconds: three correct IDs, valid probabilities, five exact model hashes,
weights 0.2, zero fallbacks. CSV SHA-256:
`65b5542cb35ea18038a2c39b0ad8fcc4130a0f7684ccd179ff32c45ecbcc29a4`.
The 100-study benchmark is still running; competition submission is pending.

## Accepted competition entry

Submitted exactly one entry: **56480789**, September 23, 2026 at 03:00:21 UTC
(September 22 evening local time). Status at receipt: **PENDING**; score unknown.
Four daily entries remain. Kernel `pssv27/knee-resolution384-five-fold`, version 1,
ID 135458402. Dataset ID 12144825 is private.

The 100-study T4 benchmark completed with zero fallbacks. All held-out predictions
reproduced within 0.000574827 maximum absolute probability difference, below 0.01.
Raw inference took 656.91 seconds (6.57 seconds/study); projected 1,300-study runtime
is 2.372 hours. Peak GPU allocation was 2.771 GiB. This is a projection, not measured
hidden-test runtime. The report binds the exact submission notebook and model hashes.

All 27 related tests pass after cleanup. Removed about 303 MiB of random spatial
smoke weights only; all trained fold weights and incumbent packages were retained.

Receipts: `outputs/kaggle_resolution384_submission/competition_receipt.json`,
`outputs/kaggle_resolution384_benchmark_results/benchmark_report.json`, and
`outputs/kaggle_resolution384_submission_results/submission_report.json`.
