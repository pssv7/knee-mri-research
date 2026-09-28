# Protocol Completion on Colab

Prepared `notebooks/15_protocol_completion_colab.ipynb` for the remaining protocol
folds 0, 2 and 4. Folds 1 and 3 are already complete. No Colab training has been
launched or GPU-tested here, and no competition submission is made by this notebook.
The user paused work on September 11 and will return later. Local verification
finished with 44 focused tests passing; see the paused handoff in `PROJECT.md`.

## Run

1. Upload the notebook at https://colab.research.google.com/.
2. Select a single-GPU runtime, preferably A100 if available. Leave the recipe
   settings unchanged. `FOLDS = [0, 2, 4]` runs sequentially; use one fold per
   session to distribute them across separate runtimes without overlap.
3. Run the cells in order. Authorize Drive and use the Kaggle account with access
   to the private sources and accepted competition rules. A Colab secret named
   `KAGGLE_API_TOKEN` avoids repeatedly entering the hidden token prompt.
4. Send the completion receipts, metrics and predictions for review. Keep the
   weights and their JSON sidecars together on Drive for later packaging.

Default output: `MyDrive/knee_runs/protocol_completion_colab_v1/foldN/`.
Each attempt has a separate subdirectory. A completed fold is skipped only after
its recipe and every required artifact checksum have been verified. Do not run
the same fold against the same output folder concurrently.

## Preservation and Limits

The adapter reuses the screened notebook's embedded source unchanged and requires
the existing screening gates, label hash and notebook hash. It preserves the
protocol preset, 16 slices, 5 series, 224-pixel input, seed 0, 12 epochs, losses,
learning rate, scheduler and effective batch 8. The single GPU uses accumulation 8.
It runs the short fit/export smoke test before each full fold.

`timm==1.0.26` and `lightning==2.6.5` match the screened runs. Colab's installed
PyTorch is retained. The execution contract permits only GPU count, PyTorch
version and fold-dependent schedule totals to differ. Initial model weights must
still hash identically; any model, seed, dependency or recipe drift stops training.
`colab_execution.json` records actual GPU, CUDA and contract differences. Single-GPU
execution is not bitwise equivalent to two-T4 DDP, even with equal effective batch.

All three full cache shards are downloaded to local VM storage. The retired
8-slice transfer is not used. Only the two small competition metadata CSVs are
downloaded, not the raw DICOM archive. Cache staging checks exact shard counts;
training additionally checks 4,349 training studies, 58 gold studies and cache shape.

Training runs in a Drive-backed attempt directory. Best-validation checkpoints
and CSV metrics are saved during training; selected/EMA/SWA weights, sidecars,
fold assignments, OOF and gold predictions are saved on successful completion.
Completion requires checkpoint checksum/metadata, split and prediction checks.
Drive writes can be interrupted; saved artifacts are not treated as complete
without a verified receipt.

An interrupted fold restarts at epoch 0 in a new directory, leaving the partial
attempt untouched. The existing averaging callback does not serialize its running
EMA/SWA state, so claiming exact mid-fold resume would be incorrect. Completed
folds survive runtime replacement. Image caches must be staged again on a new VM.

After all remaining folds complete, evaluate the five-fold protocol family and
its fixed blend with the incumbent 16/24-slice families. Neither faster hardware
nor validation gains establish a public leaderboard improvement.

Generator: `.venv/bin/python tools/prepare_colab_protocol_completion.py`.
This also refreshes the existing unlaunched Kaggle completion scripts, using the
same screen gates; it never launches a remote job.
