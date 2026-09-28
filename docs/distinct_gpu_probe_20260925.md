# Bounded GPU probe — 2026-09-25

Both frozen sampler arms passed a bounded GPU execution check. This is not a
predictive-performance result. No full experiment, validation/outer scoring,
submission or retained model checkpoint was produced.

Private Kaggle kernel `pssv27/knee-distinct-gpu-probe-v1` (135756683), version 3.
Used one Tesla T4 of two visible devices, FP16 mixed precision, no gradient
checkpointing. PyTorch 2.10.0+cu128, timm 1.0.26, Lightning 2.6.5. Internet was
enabled for pinned dependencies and original pretrained weights. Original
pretrained initialization matched exactly between arms by state-dict SHA-256.

Each arm processed 16 five-series training studies: eight containing affected
series and eight unaffected, chosen by UID order and acquisition metadata.
No selection or outer studies were used. There were two transient optimizer
steps per arm, using accumulation eight and the original 12-epoch scheduler
configuration; trained weights were discarded. All checked losses, gradients
and final parameters were finite. Frozen pair data and sources remain unchanged.

| Measurement | Control | Candidate |
|---|---:|---:|
| Microbatches / optimizer steps | 16 / 2 | 16 / 2 |
| Peak allocated memory | 5.376 GiB | 5.376 GiB |
| Peak reserved memory | 5.609 GiB | 5.609 GiB |
| Mean batch compute time, excluding first | 0.327 s | 0.319 s |
| Total fit wall time | 20.05 s | 6.08 s |

Do not interpret the wall-time difference as a sampler speed gain. The control
ran first and its first batch took 13.88 seconds, versus 0.355 seconds for the
candidate after process/GPU warm-up. This small ordered probe cannot estimate
tiny throughput differences. It verifies that both fit in memory and execute.

At the measured warm batch times, 12 epochs over 2,208 five-series training
studies extrapolate to 2.40 and 2.35 active-T4 compute hours respectively,
approximately **4.75 hours for the pair before validation, input loading and
checkpoint overhead**. This is not a duration guarantee. Some real studies
have fewer series. Conversely, cache stalls and full evaluation are absent
from this estimate. Kaggle allocated two visible T4s while only one was used;
platform quota consumption must not be equated with active-device compute time.
The receipt's naive scaled whole-fit estimate repeats startup overhead every
16 batches; do not use it as a full-run forecast.

## Preserved attempts and reproducibility

- Version 1 failed before model construction: base image lacked `lightning`.
- Version 2 installed the pinned dependencies but was cancelled before the
  sample/model stage. No completion receipt exists. Recursive searches across
  the raw DICOM tree were a likely bottleneck; cancellation cause is not
  independently established. This is not evidence of GPU OOM or model failure.
- Version 3 uses exact verified mount paths and completed. Do not reintroduce
  unrestricted `rglob` over the competition image tree in staging code.

All attempts and logs are preserved. Version 3 source was pulled and checked
against the prepared script. Returned source hashes, train-only membership,
initialization equality, step counts and absence of retained checkpoints were
verified. Small input files were rehashed; full arrays were size-checked against
the prior full-cache SHA-256 receipt, not rehashed in this GPU probe.

Evidence:

- `outputs/kaggle_distinct_gpu_probe_v3_results/gpu_probe_receipt.json`
- `outputs/kaggle_distinct_gpu_probe_v3_results/control_probe.json`
- `outputs/kaggle_distinct_gpu_probe_v3_results/candidate_probe.json`
- `outputs/kaggle_distinct_gpu_probe_v3_results/probe_sample.json`
- `outputs/kaggle_distinct_gpu_probe_v3_receipt/verification.json`

The prepared staging source is `outputs/kaggle_distinct_gpu_probe_v3/probe.py`;
generator: `tools/prepare_distinct_gpu_probe.py`. The production runner itself
was not invoked for a complete fit. Checkpoint export/reload and full selection
evaluation remain unexercised in this probe.

The next scientific step is the **single frozen outer0 matched 12-epoch pair**,
with the existing inner checkpoint selection, proxy-only reporting, per-finding
safeguards and stop rules. The user subsequently authorized this pair and it is
now [launched, with results pending](distinct_training_pair_run.md). The sampler stays disabled
by default; no independent clinical labels are available and image-level gain
remains unknown. See `outputs/distinct_sampling_pair_v1/protocol.json` and
[the implementation](distinct_sampling_implementation.md).
