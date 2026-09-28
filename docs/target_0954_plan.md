# Plan for 0.954

Active queue superseded by [multiplane_reset_plan.md](multiplane_reset_plan.md)
after the reviewed-label candidate lost validation AUC on folds 1 and 3.
The results below remain historical evidence.

Prepared 2026-09-09. This is the active plan for the user's new target and
supersedes the experiment queue in `docs/plan.md`. Historical results remain
useful, but several older summaries there contradict later measurements.

## Outcome and constraints

**September 13 update:** protocol submission 56196386 completed at **0.914**,
confirmed by the user and authenticated API. The fixed 15-model 16/24/protocol
blend is the new incumbent, +0.003 over 0.911; target gap **0.040**. Preserve
its exact version-1 notebook and model dataset. Full OOF comparison and GPU
benchmark passed. See `submission_protocol.md`. Next planned experiment is
the independent matched ConvNeXt-V2 fold-1 pair; restage it first to remove the
diagnostic timer retained in its old staged source. No new launch during this
score check. Historical 0.911 handoffs below are superseded by this update.

Target: **0.954 public macro ROC AUC across all twelve findings**. The verified
incumbent is now **0.911**, a gap of **0.043**. Submission 56135074 is COMPLETE,
confirmed by the user and authenticated Kaggle API on September 10. The user
authorized the proposed experiments and said GPU hours should not constrain
development. A target is not an achievable guarantee.

Authenticated Kaggle API audit on September 9 confirms:

- Five leading entries display 0.954; twentieth displays 0.949.
- This account has two completed 0.906 submissions: 55908661 and 55947876.
- Deadline: October 22, 2026, 23:59 UTC; entry deadline October 15.
- Five submissions per day. Notebook submissions, internet disabled, at most
  nine hours. Publicly available external data and pretrained models allowed.
- Approximately 1,300 hidden test studies. The three visible studies exercise
  inference plumbing; they cannot estimate the competition score.
- Reports are training-only. Findings are twelve independent binary outputs,
  not one mutually exclusive class. Keep sigmoid and exact submission columns.
- Fluid sensitivity and fat suppression must remain separate inputs even
  though their values coincide on every series in the current training CSV.

Local receipts: `data/meta/kaggle_audit/summary.json` and the downloaded
competition pages beside it. Regenerate with `tools/kaggle_audit.py`.

## Evidence collected

| Measurement | Result | Implication |
|---|---|---|
| Training inventory | 4,407 studies, 24,371 series | Audit by study, not by image |
| Training supervision | 4,349 mined studies, 58 image-labelled studies | Validation labels are a noisy proxy |
| Label reproduction | All 104,376 target/weight cells identical | No evidence of an accidentally changed label file |
| Label extremes / mean weight / silence | 74.75% / 0.6252 / 34.52% | Current target distribution reproduced |
| Label misses against gold at 0.5 | 45.8% of positives; 7.2% false alarms on negatives | Report omissions remain substantial |
| Current cap loses an available plane | 228 studies (5.17%) | Some studies cannot show all available planes |
| Protocol selector loses an available plane | 0 studies at the same cap of five | Coverage improvement established, AUC improvement untested |
| Omitted distinct protocol combinations | 1,577 uniform; 566 protocol | Less redundancy without more CNN forwards |
| Matched OOF studies | 4,349; all five folds | Fold 1 recovered on GPU |
| Baseline / candidate OOF macro | 0.86532 / 0.86569 | Candidate alone is not a clear improvement |
| Fixed 50/50 OOF blend | 0.87668 | Stronger local evidence for diversity than replacement |
| Paired candidate delta 95% CI | [-0.00347, +0.00428] | Consistent with no gain |
| Paired blend delta 95% CI | [+0.00922, +0.01361] | Promising against mined labels, not proof of LB gain |
| T4 raw-DICOM benchmark | 100/100 passed; 4.50 s/study | Estimated 1.70 h for 1,300 studies |

OOF confidence intervals use 1,000 study bootstrap draws, resampled within
folds and shared across all twelve findings and both models. They condition
on the fitted models, exclude training-seed variation, and inherit label bias.
Pooled OOF AUC also mixes models' calibration across folds; compare fold-level
results before making a promotion decision. Saved fold columns agree with the
legacy splits, but a fold column alone cannot prove training exclusion.

The user confirmed that the 16-slice and 24-slice runs each scored 0.906.
The fixed blend improves macro AUC on all five folds and all twelve pooled
finding AUCs. [The completed private benchmark](benchmark_16_24_results.md)
records recovery, numerical reproduction checks and runtime measurements.
Slice count is absent from the legacy checkpoint hyperparameters; exports
record the user's confirmation, and five series is taken from the project
configuration. Split membership is reconstructed, not verified from training
logs. Do not substitute another file merely because it says fold 1.

## Changes implemented

1. **Input classification and assembly:** finite parsing of sequence flags,
   no mirroring of unknown/invalid side values, duplicate-key checks on joins,
   and optional selection by plane and protocol. Training resampling remains
   available; validation and inference are deterministic. A reduced cache is
   sampled using its actual depth rather than hard-coded 24-slice indices.
2. **Inference laterality:** use available DICOM side evidence, propagate a
   consistent side through the study, and retain per-series sides when tags
   conflict. Use the image classifier for missing evidence. Read the full
   study before voting; the five-series cap must not silently change the vote.
   The geometry cross-check remains diagnostic with the existing CNN policy.
3. **Model option:** independent gated attention over series for each finding.
   ACL, PF OA and effusion no longer have to share the same series weights in
   this variant. The CNN and slice pool remain shared. This is a limited,
   testable extension, not a claim to implement the full Query2Label model.
4. **Experiment controls:** `baseline`, `resample`, `protocol`, `classwise`,
   and `convnextv2` presets in generated notebook 08. The existing uncommitted
   resampling run remains its default; choose `baseline` for the control.
5. **Training correctness:** exact optimizer-step count; EMA/SWA read weights
   after optimizer updates, including the final step. Missing grouped-split
   metadata and incomplete full-cache coverage now stop the run.
6. **Checkpoint provenance and mixed ensembles:** `.pt.json` sidecars bind a
   SHA-256 digest, labels, backbone, geometry, input policy and training/validation
   UIDs to each exported checkpoint. An explicit JSON ensemble can combine
   different backbones, pooling variants and slice counts with fixed global
   weights. Legacy globs remain for old homogeneous ensembles, with sidecar
   checks to reject an incompatible input policy.
7. **Measurements:** reproducible input audit, Kaggle audit and paired OOF
   comparison tools. No label values were overwritten or fitted on gold.

## Experiment order and gates

### Resume after the user returns with the result

Current handoff: **training authorized; protocol passed folds 1 and 3;
classwise completed both folds and failed its replication gates**.
Use the measured results to determine further
experiments. GPU-hour scarcity is not a user constraint. Retain the scored
0.911 ensemble unchanged. The historical result branches below explain the
decision; the wait-for-result requirement is superseded. No new competition
entry is included in the paired training job.

Latest preparation: fixed protocol addition improves the incumbent-family
OOF proxy on both screening folds; see `protocol_incumbent.md`. Remaining
protocol folds 0/2/4 are staged as separate single-run jobs. Full matched V2
fold 1 is also staged, with identical aggregation initialization and unchanged
pooling/selection. Its two-T4 smoke test passed, but full training has NOT run.
At the latest check Kaggle had only 1.27 hours remaining; new quota or another
GPU environment is needed before full jobs can finish. See `convnextv2_fold1.md`
for exact readiness, source hashes and the quota refresh time. Nothing is
scheduled to start automatically.

**Resumed September 11 evening (September 12 UTC):** reset verified at 21 GPU
hours. Protocol folds 0 and 2 version 1 launched and report RUNNING; source and
metadata receipts verified. Fold 4 was rejected by Kaggle's two-concurrent-job
limit and remains staged for the next free slot. GPU smoke logs/results are
pending. The V2 pair remains staged. This supersedes the quota block above;
see `PROJECT.md` for the current handoff.

September 10 execution update: the baseline completed (selected mined macro
0.871666) but the original forked script stalled at training shutdown. Its
checkpoint was recovered without retraining. Recovery job version 2 passed
a native-DDP GPU smoke test and baseline OOF reproduction, then completed.
Matched mined-label macro is 0.871575 baseline, 0.881118 protocol and 0.885457
fixed blend. Fold-3 version 1 completed with native DDP in both arms: macro
0.871415 / 0.877224 / 0.886863, passing the same gate. The fixed blend improves
all twelve mined-label findings in both folds. On gold-58, fold-3 macro is
0.887565 / 0.870440 / 0.889122; PF OA and synovitis do not show a demonstrated
image-label gain. Protocol is an ensemble candidate, not an established
standalone replacement. See `protocol_fold3.md` for exact results and caveats.
Remaining-fold training is eligible, but a full incumbent comparison and
runtime benchmark are still required before promotion. The public score is 0.911.

1. Confirm the reported result belongs to submission **56135074**, notebook
   `pssv27/knee-16-24-fixed-blend`, version 1. Record its final status, public
   score and runtime/error information when available in `PROJECT.md`,
   `docs/submission_16_24.md` and the worklog. A status request is read-only;
   never repeat the submit command to check progress.
2. If the score exceeds **0.906**, retain the fixed blend as the provisional
   best entry and preserve its exact artifacts. A small rounded leaderboard
   gain is not statistical proof, and does not establish that 0.954 is reachable.
   The entry includes the benchmarked input safeguards, so do not attribute
   every change in score solely to averaging.
3. If the score ties or falls below **0.906**, keep the previously scored
   16-slice five-fold model as the lower-compute control. Do not chase the
   public leaderboard with per-finding weights or repeated blend sweeps.
   If scoring fails, diagnose the failure and logs before proposing a retry;
   no replacement submission is pre-authorized.
4. Next proposed training allocation, after agreeing compute: baseline versus
   protocol selection on fold 1, followed by fold 3 only if the screening gate
   below passes. Then test classwise pooling and ConvNeXt-V2 as separate
   experiments. Do not launch all five folds of an unmeasured variant.
5. Gold-58 and scanner/language-stratified diagnostics remain incomplete.
   Gold must not select epochs, blend weights or per-finding members. Labels
   were not changed; any labeling experiment must preserve the current matrix.

### 0. Confirm the existing ensemble opportunity

Priority over a new backbone training run. Geometry is now user-confirmed and
the selected checkpoints are packaged. Fold-1 recovery, full OOF comparison,
per-finding and fold-level checks, and the 100-study GPU benchmark are complete.
The fixed blend passes these checks. The user subsequently authorized one
entry: submission **56135074**, notebook version 1, pending at 2026-09-10
02:07 UTC. See [the submission receipt](submission_16_24.md).
Scanner/language-stratified evaluation and the
complete gold diagnostic below remain unperformed.

Evaluate the complete image-labelled holdout once as a diagnostic. Compare
the same UIDs, not a historical macro. Do not select twelve blend weights,
epochs or members on the 58 gold cases. Export a manifest only when geometry
and checkpoint identity are known; test it through the actual submission code.

Before submission, benchmark 100 diverse studies with full decoding and no
duplicated A/B inference, including the largest studies. Project to the
official approximate test size with at least 30% runtime reserve. Ten models
can fit in theory, but the project's old timing estimates are not a benchmark
of this implementation. If memory or runtime fails, compare smaller fixed
ensembles on OOF; never average unrelated weights to avoid the forward cost.

### 1. Repair the input selection bottleneck

Run notebook 08 with `EXPERIMENT = "baseline"`, fold 1, then `"protocol"`,
same seed, 16 slices, five series, 12 epochs, same label file and split.
The control includes this revision's correctness fixes; compare both against
the saved baseline as additional context. Keep auxiliary objectives off.

Pre-register the practical screening gate: mined OOF macro delta at least
+0.003 on fold 1, no common finding down more than 0.02, and no invalid
predictions. This gate is a compute allocation rule, not a significance claim.
Screening passes must repeat on fold 3 before buying all five folds. An
inconclusive result can still be useful as an ensemble member if a fixed blend
improves matched OOF consistently. Never promote on input coverage alone.

### 2. Test finding-specific pooling

Execution update, September 11: fold-1 version 1 completed. Standalone macro
delta +0.002474 fails the original +0.003 gate; do not call it a pass. The fixed
blend improves all twelve validation findings and an exploratory equal-thirds
baseline/protocol/classwise blend adds +0.004804 versus baseline/protocol.
Version 1 of `pssv27/knee-classwise-fold3-pair` completed: standalone macro
0.866763 versus baseline 0.871415, with ACL down 0.055831. The equal-thirds
blend adds +0.002626 versus baseline/protocol, below the predeclared +0.003
incremental gate; its CI includes zero. Both gates FAILED. Stop classwise
expansion and do not rescue it with blend-weight tuning. Gold remains diagnostic
and PF OA/synovitis gains did not repeat. See `classwise_fold3.md` for results.
Prioritize the protocol candidate's full incumbent comparison and proceed to
the separately planned ConvNeXt-V2 test, not a protocol/classwise architecture.

Run `EXPERIMENT = "classwise"` against `"baseline"` on the same folds and
use the same gates. Only study-level pooling changes. Explicitly inspect PF OA
and lateral meniscus, without sacrificing the other ten findings. The new
pool retains shared slice attention; if it helps but focal lesions still fail,
class-conditioned slice attention or a slice sequence encoder is a separate
future experiment. Do not bundle both before the first measurement.

### 3. Challenge the backbone with a matched test

Run `EXPERIMENT = "convnextv2"`: ConvNeXt-V2-Tiny FCMAE initialization at
the same resolution, geometry and schedule. Start with Tiny to keep compute
comparable, not Base merely because it is larger. This controlled probe keeps
the legacy input normalization; a follow-up using the pretrained model's
recommended statistics is a separate ablation if transfer is weak.

FCMAE/GRN and public weights justify a trial, not a knee-MRI performance claim.
There is no local proof V2 beats V1. Retain it only if the same-fold comparison
passes, or its fixed blend improves the incumbent. DINOv2-S and the retired
frozen/tuned blend already lost here; no repeat without a new, distinct reason.

### 4. Spend on missing information if the earlier stages plateau

Rebuild raw-DICOM cache at 320 or 384 only after the cheap experiments. Train
a matched model and check PF OA/meniscus changes, per-study runtime and memory.
Upscaling the 224 cache cannot recover discarded detail. Maintain a versioned
cache and separate side-classifier input geometry. The current artifact loader
deliberately rejects a different source-cache geometry; update its contract
and regression tests before this stage, rather than silently mixing caches.

Expand image-labelled evaluation with participating qualified readers and
permitted data if available. Aim for several hundred studies spanning vendors
and report languages, with a prospectively fixed split and sufficient positive
and negative examples per finding. This cannot be supplied by code or by an
LLM inventing image labels. Treat the repeatedly inspected gold 58 as a reused
diagnostic set, not an untouched final estimate.

Lexicon language coverage deserves a separate reviewed labeling experiment:
Bulgarian/Croatian meniscus phrases are frequently missed, and that affects
sample weights even when targets use mostly the LLM. Add language-specific
fixtures and inspect negation/localization on sampled reports before rebuilding
a candidate label matrix. Keep the original matrix versioned. Do not equate
more lexical matches with more correct diagnoses.

## Local verification

- Full suite: 650 passed; after adding packaging and side-whitespace tests,
  the expanded focused pipeline suite passed all 25 tests.
- Both notebook-10 CPU smoke tests passed: legacy homogeneous inference and
  explicit heterogeneous inference with unequal weights. Each emitted the
  exact twelve probability columns and used fallback only for the deliberately
  missing study. These use small synthetic DICOM fixtures, not score validation.
- All 114 code cells in 16 generated notebooks parsed after excluding IPython
  directives. `git diff --check` passed.
- All ten real selected checkpoints loaded strictly and contained finite
  weights. The 1.4 GB local package is complete, with checksummed sidecars.
  A transient write failure was recovered with `--resume`, which verifies
  completed exports tensor-for-tensor against their sources before reuse.
- A subsequent private T4 run completed in 16.23 minutes, recovering all 871
  fold-1 predictions and benchmarking all ten models on 100 raw studies with
  no errors. See `benchmark_16_24_results.md` for limitations and receipts.

## Commands and handoff

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python tools/audit_inputs.py
.venv/bin/python tools/audit_labels.py
.venv/bin/python tools/compare_oof.py \
  --baseline models_train/fold*/oof_fold*.parquet \
  --candidate 'experiment /slice_24/'fold*/oof_fold*.parquet \
  --draws 1000 --out data/meta/compare_24slice_complete
.venv/bin/python tools/make_notebook.py
.venv/bin/python -m pytest tests -q
.venv/bin/python tools/check_submission.py 10
.venv/bin/python tools/check_submission.py 10 --explicit
.venv/bin/python tools/package_16_24_ensemble.py
```

Run the regenerated notebook 08 on Kaggle/Colab with all three cache shards,
side predictions and the exact label matrix. In its Experiment cell set
`EXPERIMENT = "protocol"` (or the required control). Keep each run in a separate
output directory/dataset so matching filenames from different runs cannot
overwrite one another. Preserve every exported `.pt.json` beside its `.pt`.

In notebook 10 set `KNEE_ENSEMBLE_JSON` before the setup cell, or edit
`ENSEMBLE_JSON` to the attached manifest path. Its paths are relative to itself:

```json
{
  "members": [
    {"checkpoint": "baseline/convnext_fold0.pt", "weight": 1.0},
    {"checkpoint": "candidate/convnext_fold0.pt", "weight": 1.0}
  ]
}
```

This is a format example. The actual prepared candidate is
`outputs/ensemble_16_24/ensemble.json`, ten selected checkpoints with total
weight 0.5 per family. Its `provenance.json` records source hashes and which
saved OOF memberships were checked. Sidecar training/validation UID lists are
intentionally empty for these legacy exports; reconstructed folds are separate
and must not be presented as verified training records. The packaging command
refuses to overwrite an existing directory. A private dataset and private GPU
benchmark have now been created. One subsequently authorized competition
submission is pending; no new leaderboard score has been returned.

For a 50/50 family blend,
give each family half the total weight even if their member counts differ.
Do not list EMA and selected weights as separate votes for the same experiment
and fold. Old weights need explicitly established geometry and new sidecars.

## Sources and limits

- [Competition evaluation](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation),
  [data](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data),
  and [leaderboard](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard):
  verified by authenticated API, September 9, 2026.
- [Attention-based MIL](https://proceedings.mlr.press/v80/ilse18a.html):
  supports trainable gated instance pooling; does not establish a score here.
- [Query2Label](https://arxiv.org/abs/2107.10834): supports class-conditioned
  feature pooling in multi-label classification; natural-image evidence,
  not a measured knee-MRI transfer result.
- [ConvNeXt-V2-Tiny model card](https://huggingface.co/timm/convnextv2_tiny.fcmae_ft_in22k_in1k):
  public timm model, pretraining and usage. No competition-specific proof.

Full DICOM/cache training is unavailable locally: the environment reports no
CUDA or MPS accelerator, and the full image cache is absent. The authorized
private GPU inference run used the existing Kaggle cache and competition data.
Only further measured experiments and scored submissions can establish progress
to 0.954; no retraining or competition submission was authorized in that run.
