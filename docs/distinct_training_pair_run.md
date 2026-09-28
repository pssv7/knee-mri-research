# Single matched sampler pair — completed; screen failed 2026-09-25

## Verified completion

Both arms completed 12 epochs in 2.91 h wall time. Outer report-proxy macro
AUC: control **0.878838**, candidate **0.879690**, delta **+0.000853**, paired
95% interval **[-0.007144, +0.008840]**. Fracture delta **-0.026898** also
breaches the -0.020 floor. The frozen screen failed: stop the sampler branch,
keep the default disabled, and do not launch confirmation, blending or submission.
This is report-proxy evidence, not independent image-accuracy evidence.

Source, input configuration, matched initialization/recipes, checkpoint and
prediction hashes, and all 12 epochs were verified. All 28 package hashes
match returned and local files; the downloaded launcher matches frozen staging.
Local evaluation exactly reproduces the full returned report using the
predeclared 2,000 bootstrap draws. Artifacts are preserved under
`outputs/kaggle_distinct_training_pair_v1_results/`; verification and reproduced
evaluation are in `outputs/kaggle_distinct_training_pair_v1_completion_receipt/`.

## Historical launch record

The user authorized the frozen 12-epoch control/candidate experiment after the
CPU integrity and GPU execution gates passed. Kaggle accepted private kernel
`pssv27/knee-distinct-sampling-pair-v1`, version 1, and reported RUNNING.
Executed source and metadata were pulled and verified against local staging.
No training result or score is available at this launch checkpoint.

Two arms run concurrently in separate processes, one Tesla T4 each. Both use
seed 0, original timm pretraining, identical frozen outer0 roles/labels, the
same runtime, 12 epochs, effective batch eight, and the same scheduler. Only
the sampling policy differs. They are standalone models, not an ensemble.
Both unmodified runners repeat full cache hashes and provenance preflights.

Each checkpoint is selected solely on the 553-study inner-selection set.
There are 2,208 training studies and 691 outer studies. Outer predictions are
produced after each arm selects its checkpoint; the comparison runs only after
both processes complete and their contracts pass the evaluator checks.

The predeclared exploratory point gate is macro delta ≥0.005 with no per-finding
point delta below -0.02. The launcher also requires a positive lower paired
bootstrap bound to mark a result eligible for separate confirmation; an
inconclusive interval does not advance. No automatic confirmation runs, label
changes, blends, or submissions are allowed. Proxy success cannot establish
an independent image-finding accuracy gain. The protected gold/review exclusions
remain enforced.

The job stops the other arm if either process fails; it does not silently
retry training. The launcher has a 12-hour resource cap, not an instruction
to use the entire allowance. The probe suggests roughly 2.4 active-device
compute hours per arm before validation/I/O; concurrency shares host resources,
so several hours of wall time are expected but not guaranteed.

Artifacts:

- Frozen protocol: `outputs/distinct_sampling_pair_v1/` (unchanged).
- Launcher: `outputs/kaggle_distinct_training_pair_v1/train_pair.py`.
- Preparation tool: `tools/prepare_distinct_training_pair.py`.
- Prelaunch checks: `outputs/kaggle_distinct_training_pair_v1/authorization_and_checks.json`.
- Executed source/settings and observed running status:
  `outputs/kaggle_distinct_training_pair_v1_receipt/launch_verification.json`.

On completion, retrieve `pair_receipt.json`, `screen_decision.json`,
`outer_proxy_evaluation.json`, both `run_contract.json` files, selected weights,
outer predictions and training logs. Verify matched initialization, source,
input and checkpoint hashes before interpreting the result. If export or
evaluation fails after fitting, preserve completed checkpoints and diagnose
the failed stage; do not rerun completed training to repair an artifact issue.

Status command:

```sh
.venv/bin/kaggle kernels status pssv27/knee-distinct-sampling-pair-v1
```
