# Campaign implementation and fold-1 handoff — 2026-09-16

The specification remains [multiplane_campaign_plan.md](multiplane_campaign_plan.md).

**Update 2026-09-16: spatial fold 1 FAILED** (macro delta -0.002862 vs +0.005,
MCL -0.0532); the spatial lever and the plane ablation are closed. See
`spatial_multiplane.md`. The next notebook to run is
`notebooks/20_resolution_pair_fold1_colab.ipynb`.
The campaign is prepared for its first GPU screen, not completed or promoted.
No GPU job, competition submission or change to final entries was launched.

## Notebook to run now

Upload `notebooks/19_spatial_fold1_colab.ipynb` to Colab and select one A100.
Set `DRIVE_CACHE` to the folder containing the **three original 224-pixel,
24-slice cache shards**. The user's six 384-pixel shards belong to the later
resolution experiment, not to this architecture screen. Do not resize them
into a replacement control.

The notebook copies verified inputs to local disk and saves outputs to
`MyDrive/knee_runs/spatial_fold1_v1/`. It runs a representative five-series,
24-slice, fp16 smoke with microbatch one and accumulation eight, then actual
training smokes for both arms, then the matched 12-epoch control/candidate pair.
The representative smoke checks finite forward/backward, parameter movement,
initial identity and export/reload agreement. Training must fit the proposed
36-GiB gate. Each run records training and validation timing/memory separately.
The actual-data smoke supplies an explicitly approximate full-run projection.
Rerunning preserves attempts and starts training at epoch zero.

Read `comparison.json` and `pair_progress.json` in the completed attempt.
Both use the same configured +0.005 standalone threshold and −0.020
per-finding floor. Fold 3 is not launched automatically. Historical protocol,
classwise, V2 and slice-transformer thresholds remain +0.003.

## Implemented preparation

- `tools/campaign_receipt.py` snapshots revision, exact dirty source hashes,
  labels, reconstructed splits, incumbent OOF/manifest and all 15 checkpoint
  hashes. Output: `outputs/multiplane_campaign/input_receipt.json`. Legacy
  training exclusion is explicitly reconstructed, not independently proven.
- Six-shard 384 adoption now requires original receipts, checks payload hashes,
  geometry, disjoint series, exact 24,371-series coverage and complete side
  predictions. It does not manufacture raw-DICOM provenance or require Drive
  hardlinks/symlinks. Verification of the user's files happens in Colab.
- `tools/compare_incumbent_addition.py` accepts explicit incumbent/candidate
  OOF, saved splits and checkpoint/run provenance. It rejects intersections,
  duplicate or incomplete predictions, invalid probabilities, changed weights
  and train/validation overlap. Standalone, mandatory 50/50 allocation, exact
  equal-family deployment and combined-family results are named separately.
  Its module docstring describes the provenance JSON. Candidate OOF must be
  primary seed 0, with matching labels and saved splits.
- `tools/prepare_campaign_completion.py` stages separate candidate-only fold
  0/2/4 seed-0 runs and fold-1 seed-1 repeats for spatial and resolution384.
  Output identities, scripts and receipts include seed; split generation stays
  fixed. `run_candidate.py` wraps smoke/full execution with completion and
  artifact-hash receipts. These scripts are staged, not authorized promotions.
- `tools/prepare_colab_resolution_pair.py` generates
  `notebooks/20_resolution_pair_fold1_colab.ipynb`: matched single-GPU 224/384
  arms, unchanged 16 selected slices, effective batch eight, 12 epochs and
  checkpointing in both arms. It requires both original cache sets (about
  120 GiB free local disk). This is the planned fallback if Kaggle's measured
  projection exceeds eight hours; use after the spatial sequence. Notebook
  17 remains a candidate-only diagnostic and now defaults to fold 1 only.
- Submission notebook 10's explicit integration check now includes spatial
  and plane-adapter checkpoints. Sidecar-bearing ensemble tests cover absent
  and unknown planes and agreement after checkpoint reload.

## Verification

The real ConvNeXt CPU smoke, at five-series/24-slice/224 geometry, passed for
both arms: finite training, parameters moved, initial maximum logit difference
0.0 and export/reload maximum difference 0.0. Saved in `outputs/spatial_smoke/`.
CPU timings are not used to claim GPU fit or campaign duration.

The historical equal-third protocol OOF was reproduced for all 4,349 studies;
maximum probability difference was 3.974e-8. Separate tests verify the 50/50
arithmetic. Full suite: 836 passed, one skipped, followed by targeted checks
of the final smoke-loop changes. The explicit submission integration check passed
with four members, finite predictions for all 12 findings and exactly one
fallback, for the intentionally missing study.

## Still gated by execution

The A100 smokes, spatial fold-1 AUC and all six real Drive payloads remain
unverified here. Follow the source plan's sequence and thresholds. A failed
spatial screen closes that lever; it does not authorize plane adapters.
The optional plane ablation is deferred and excluded from deployment.

Seed-rule acceptance is required before launching a repeat; report seed
variability against saved controls and incumbent separately, and never include
the repeat in the submission family. Deployment-rule acceptance and completed
screen/blend/seed gates precede expansion. No concurrency departure is assumed.

Mixed-resolution inference remains conditional on resolution promotion.
Exact-manifest packaging, generalized benchmarking, 100-study T4 measurements,
submission approval and final-entry selection remain outstanding downstream
steps. The current submission loader still rejects 384 checkpoints: these
training notebooks do not make them deployable. Expansion must start by
October 1, packaging finish by October 17, and the recorded submission deadline
is October 22 at 23:59 UTC. If both levers fail, stop expansion and reassess
supervision and validation as specified in the plan.
