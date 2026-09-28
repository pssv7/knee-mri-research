# RSNA Knee Abnormality Detection — Project State

Read this first when picking the work back up cold. It covers what exists, what
the numbers are, what has been tried and failed, and which traps have already
been paid for.

## Completed 2026-09-27 — silence-masking screen failed

Silent label cells (weight 0.25) of the three OA findings, Synovitis, Baker's and
Contusion weighted 0 in training rows; concurrent Kaggle T4 pair; the control
again reproduced the historical control exactly. Declared primary (explicit-label
AUC, mean of the six): **−0.0001 [−0.0160, +0.0166]**, fail. Standard macro
−0.0068. PF OA +0.036 [+0.000, +0.074] (edge; one finding) and Contusion −0.037
[−0.074, −0.007]. Unmasked ligament/meniscus findings rose (ACL +0.022) as an
undeclared side effect. See [the screen](docs/silence_screen.md).

## Completed 2026-09-26 — mirror-augmentation screen failed

User-authorised training (a departure from the reset's no-training rule). A
concurrent Kaggle T4 pair on outer0, with the only change mirroring each training
study (p 0.5, labels kept). Macro **−0.0008 [−0.0084, +0.0072]**; MCL −0.022
breaches the floor; ACL/menisci unchanged. Stop. **The concurrent control
reproduced the historical control bit for bit** (checkpoint `38520640…`): the
recipe is deterministic on Kaggle T4/torch 2.10, but that is not a seed-variance
estimate. See [the screen](docs/mirror_screen.md). A silence-masking screen for
OA/Synovitis/Baker's/Contusion is running (`pssv27/knee-silence-screen-v1`).

## Completed 2026-09-25 — series-selection check (inference only)

Same Tiny control and outer0 studies, five declared selection policies on a
Kaggle T4; the reference reproduced the saved predictions exactly. Best arms:
`quality5` (keep the thinnest-slice repeat) +0.0035 [−0.0001, +0.0077] and `all`
series +0.0033 [+0.0002, +0.0066]. Forcing a sagittal fluid series: +0.0004
(26 studies, null). All are below the +0.005 gate; the OOF −0.035 association was
mostly confounding. A small lead, not a fix. See
[the check](docs/series_selection_check.md).

## Completed 2026-09-25 — perturbation variants scored against written labels

Each probe variant got labels written from its study's outer0 mined labels
(inherited; the mirror both swapped and kept). Unmodified arm 0.8788 = the saved
screen. Keeping medial/lateral labels on a mirrored knee beats the repo's swap by
**+0.028 macro [+0.020, +0.037]** (Medial Meniscus +0.19), so the swap convention
is wrong for this model. Even correctly labelled, a mirror costs ACL −0.087,
Lateral Meniscus −0.080, MCL −0.058: the model-driven findings depend on
canonical orientation. 8 slices cost −0.018 macro (Lateral Meniscus −0.062).
Synthetic noise σ 5 costs only −0.002, so noise cannot explain TOSHIBA's −0.053.
Brightness and the side covariate cost nothing. See
[the probe, labelled follow-up](docs/perturbation_probe.md).

## Completed 2026-09-25 — perturbation probe of the Tiny control

Inference-only sensitivity probe on the 691 outer0 studies: 14 declared input
variants versus the unmodified pass; no labels read, nothing trained or
submitted. Private Kaggle T4 kernel `pssv27/knee-perturbation-probe-v1`; the
unmodified pass reproduced the saved predictions with max |Δp| 0.0. Intensity
shifts up to ±20 levels and the `is_right` covariate barely matter (Spearman
≥0.982). Noise σ 5–10 shifts every finding directionally (Fracture −0.31 logit
at σ 10) but keeps the ranking (≥0.944). Slice density matters: 8 alternate
slices give Spearman 0.79–0.95, while dropping two edge slices at each end changes
almost nothing. **Mirror result:** the model locates the medial meniscus by
anatomy, not image position. Its flipped raw "Medial Meniscus" output tracks
the original medial meniscus (Spearman 0.862, versus 0.285 with the lateral).
The repo's flip-with-label-swap convention (`lateral_swap_order`, `FLIP_TTA`,
`laterality_flip`) is contradicted for this model. All three are off; keep them
off. See [the probe](docs/perturbation_probe.md).

## Completed 2026-09-25 — Small/224 screen failed

User-run Colab Small completed all 12 epochs. Downloaded prediction artifacts
and receipts verified; metrics and 2,000-draw bootstrap reproduced exactly.
Outer proxy macro **0.877015** versus historical Tiny **0.878838**, delta
**-0.001823**, 95% interval **[-0.011790, +0.007877]**. Contusion **-0.058151**
and lateral meniscus **-0.026041** breach the -0.020 floor. Stop this configuration;
no expansion, blend or promotion. Inner validation peaked at epoch 10 then
declined through epoch 12. A100/torch2.11 differs from Tiny's T4/torch2.10;
this remains exploratory report-proxy evidence. Selected checkpoint bytes
were not supplied; contract/completion hashes agree but are not a local
checkpoint verification. Artifacts: `outputs/colab_small_224_outer0_results/`.
See [full verification and limits](docs/convnext_small_fold0_colab.md).

## Prepared 2026-09-25 — Small at 224, nested fold 0 on Colab

After the sampler screen failed, the user requested one ConvNeXt-Small/224
fold-0 Colab notebook using existing Drive shards, explicitly excluding 384.
`notebooks/22_convnext_small_224_fold0_colab.ipynb` is prepared and self-contained.
It reuses `/content/drive/MyDrive/kagglehub`, as recorded in older notebooks,
and the current nested outer0 roles/labels (2,208/553/691 train/selection/outer).
Original sampling, 16 slices × five series, seed 0 and the 12-epoch recipe stay
fixed. Smoke precedes one Small fit; artifacts save to Drive. Historical Tiny
comparison is explicitly exploratory because Colab hardware/runtime may differ.
32 targeted tests passed, including CPU surrogate smoke/reload; CUDA remains
to be checked in Colab. No remote run or additional experiment was launched.
See [notebook instructions and limits](docs/convnext_small_fold0_colab.md).

## Active update: 2026-09-24 — validation credibility and one-model reset

**Score clarification:** the user reports **0.917** for the September 23
combined 20-model submission **56505208**, a **+0.003** public-score improvement
over 0.914. Record it as an ensemble gain, not a failed score or a standalone
gain. Source: `outputs/kaggle_combined20_submission/score_update_20260924.json`.
The original submission receipt's null score is preserved as history; the new
score has not been independently rechecked with Kaggle. The earlier 384-only
submission (56480789) still has no available score. Preserve the **0.917** ensemble
as the latest reference and the older **0.914** artifacts for comparison.
The user's reset toward credible validation and standalone performance stands.

The active protocol is [the September 24 reset](docs/validation_reset_20260924.md).
It supersedes the earlier training/ensemble queue and conflicting statistical
claims in historical plans, playbook and code comments. No training campaign,
blending rescue or public-score tuning loop is authorized by this reset.

Completed local read-only audits, preserving existing work:

- Rebuilt all **104,376** mined target/weight cells exactly; 58 organizer labels
  match both local gold copies; no gold IDs in mined training. This establishes
  provenance, not image-level accuracy.
- Verified ten selected protocol224/384 checkpoint hashes and declared split
  membership; reproduced protocol prediction references exactly. Same-fold
  observations selected the checkpoint AND scored its exported predictions.
- Gold has repeatedly influenced extraction, architecture and experiment
  selection. It is historical development evidence, not an untouched test.
- Available index: 819,078 instances, 24,371 series, no duplicated UIDs.
  **54 normalized-report duplicate groups / 204 studies**, 48 groups cross-fold,
  one connects gold and mined studies. These are identity-review leads, not
  proven duplicated images. Patient IDs/dates/site IDs and full local pixels
  are unavailable; repeat examinations and hidden-test overlap remain unverified.
- **4,340/4,349** held-out studies use scanner proxies also present in training;
  the split scarcely measures unseen-scanner transfer.
- The September 23 solo report audit found PF OA omissions but stopped before
  any repair/training. Same-reviewer same-day agreement is not independent
  adjudication (`docs/label_quality_result.md`). Its universal 0.01 leaderboard
  resolution claim is unsupported by a recoverable power analysis.

Keep 224-pixel protocol ConvNeXt-Tiny V1 (16 slices × five series, no auxiliaries)
as the reproducible single-model control recipe. The 384 family's pooled proxy
gain over it is only **+0.000813**, with hardware confounding and losses on
screening folds 1/3. This does not settle image-level ranking. Neither ensemble
score (0.914 or 0.917) is the single model's score. Two conditional supervision experiments
are specified in the reset, both requiring standalone image-reference evidence.

**Implementation update:** the user authorized implementation and confirmed
**no expert readers available**. The no-expert fallback is implemented in
`outputs/validation_fallback_v2/`: 3,452 eligible studies, 955 exclusions,
component-safe nested splits, train-only label transformation, an isolated
single-control runner and a provenance-checked proxy comparator. **87 targeted
tests passed**, plus real split/hash checks. No GPU job or production-label
change occurred. The subsequent **full-cache CPU preflight passed on Kaggle**:
three arrays (29.35 GB), 3,452 eligible studies / 18,955 series. Executed source
and private CPU/no-internet settings verified; CUDA training remains untested.
Receipt: `outputs/kaggle_nested_proxy_preflight_v1_results/preflight_receipt.json`.

The narrower PF proposal rule failed its existing-review design screen
(2/5 explicit-miss agreement); it is closed before opening the protected
reserve or training. This is not an independent precision estimate.
See [implementation and commands](docs/validation_fallback_implementation.md).
**Input-readiness gate completed:** actual cache files are now hash-bound and
the no-training preflight passed on the cache host. The subsequent bounded
[input audit](docs/baseline_input_audit_20260924.md) found avoidable repeated
slices in 2,479 retained series / 810 eligible studies. A 120-series CPU pixel
sample confirmed all predicted distinct-slice counts; no blank or flat sample
volumes. The axis chosen for mirroring matches geometry throughout retained
series. Shape distortion and omitted sequences are scanner/language-confounded
hypotheses, not proven causes of error. Four new synthetic checks passed.

**Sampler implemented, disabled by default:** isolated duplicate-free 16-slice
sampling for raw lengths 17–23, with reversal-aware cache/inference parity and
raw-count provenance checks. **450 distinct targeted tests passed**. One matched
outer0 pair is frozen at `outputs/distinct_sampling_pair_v1/`, with unchanged
v2 splits/labels and new runtime hashes. See
[implementation and limits](docs/distinct_sampling_implementation.md).
Both updated full-cache preflights and the **109-series real-cache smoke passed**:
61 affected series yield 16 distinct images; 48 unaffected series remain
unchanged; study assembly passed for 16 studies / 78 series. Private CPU-only
execution and returned source hashes verified. Receipt:
`outputs/kaggle_distinct_sampling_smoke_v1_results/smoke_receipt.json`.
The [bounded GPU probe](docs/distinct_gpu_probe_20260925.md) also passed on one
T4: two transient optimizer steps per arm, matched pretrained initialization,
finite losses/gradients, 5.38 GiB peak allocated, ~0.32 s per warmed microbatch.
No probe checkpoint retained or evaluation performed. The user subsequently
authorized the single frozen matched 12-epoch pair; it is **launched and Kaggle
reported RUNNING**, private kernel `pssv27/knee-distinct-sampling-pair-v1`,
version 1. One T4 per arm, no automatic follow-ups or submissions.
**Completion verified September 25:** Both arms completed 12 epochs in 2.91 h wall time. Outer report-proxy macro
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
See
[run status and retrieval instructions](docs/distinct_training_pair_run.md).
The short probe is not evidence of image gain, and its weights are not
candidates for deployment.
The failed PF-label-repair branch stays closed.
The 120-study image-review handoff remains prepared in
`outputs/validation_reset_20260924/`; qualified review of its first 40 studies
is conditional on reader access. No locked final image set has been created.
Trustworthy small-gain image-model ranking remains unsupported.

Reproduce the audit with `.venv/bin/python tools/audit_validation_reset.py`.
No old labels, weights, submission receipts or experiments were replaced.

## Previous update: 2026-09-22 evening — 384 five-fold submitted

User authorized cleanup, Kaggle GPU execution and submission. Exactly one competition
entry accepted: **56480789**, notebook `pssv27/knee-resolution384-five-fold` version 1
(kernel 135458402), submitted 2026-09-23 03:00:21 UTC. **PENDING**, score unknown.
Private model dataset ID 12144825. Incumbent remains 0.914 pending the new score.

Both uploaded notebook sources/settings verified against staging. Visible-test run:
three correct rows, zero fallbacks, 53.77 seconds on T4. The 100-study benchmark passed:
zero fallbacks, maximum held-out prediction difference 0.000574827, 656.91 seconds,
projected 1,300-study runtime 2.372 hours, peak GPU allocation 2.771 GiB.
All 27 related tests pass. Removed only ~303 MiB of reproducible random smoke weights.
Details: `docs/submission_resolution384.md`; accepted-entry receipt:
`outputs/kaggle_resolution384_submission/competition_receipt.json`.

## Previous update: 2026-09-22 — 384 five-fold submission prepared locally

Imported completed resolution384 folds 0–4 from Downloads, preserving fold 1 and
excluding the earlier incomplete fold-3 archive. Organized artifacts and receipts:
`outputs/resolution384_downloads/`. Selected-weight package (five equal 0.2 votes):
`outputs/ensemble_resolution384/`. Offline notebook and Kaggle metadata:
`outputs/kaggle_resolution384_submission/`. No upload, GPU launch or leaderboard entry.

All selected checkpoint hashes, training contracts, input receipts and split membership
verified; 4,349 unique OOF studies. Pooled mined-label macro 0.873693 is diagnostic,
not a leaderboard result. The staged notebook passed a local real-weight DICOM smoke
with zero fallbacks and separately verified 224-pixel SideNet preprocessing.
GPU runtime and held-out reproduction remain pending. Related tests: 24 passed;
three cache-building tests failed on the existing 2-GiB disk-reserve guard.
EMA/SWA and Lightning weights remain in original ZIPs due to limited disk space.
Details and remaining execution: `docs/submission_resolution384.md`.

## Previous update: 2026-09-16

**The spatial volume encoder FAILED its fold-1 screen; the lever is closed.**
Colab A100, matched 24-slice protocol control versus the same model with the
residual 3D blocks, 12 epochs each. Mined-label OOF on 871 studies: control
**0.880463**, spatial **0.877601**, delta **-0.002862** against +0.005, paired
bootstrap [-0.010086, +0.004138]. MCL -0.0532 breaches the -0.020 floor.
The fixed 50/50 blend's +0.0075 is against its own control and is not
evidence. Per `docs/multiplane_campaign_plan.md`: no fold 3, no seed repeat,
no plane adapters. Full table: `docs/spatial_multiplane.md`.

A100 cost measured: about 80 min per 12-epoch arm at 24 slices × five series.
The 24-slice control (0.8805) did not beat the 16-slice protocol arm (0.8811)
on fold 1.

First attempt failed at the probe with `No module named 'src'` (import finder
cache on a Drive working directory); fixed with `importlib.invalidate_caches()`
in `tools/prepare_colab_spatial_pair.py`, with a regression test.

**Next: the 384-pixel matched pair** on the A100,
`notebooks/20_resolution_pair_fold1_colab.ipynb` (regenerated, unchanged; 48
resolution/campaign tests pass). The six 384 shards are verified on Colab when
the notebook adopts them. Same gate. If it fails, the plan's stop rule applies:
no further model families, the 0.914 incumbent stands, and remaining effort
goes to supervision and validation. Incumbent LB remains **0.914**.

## Previous update: 2026-09-15

Implemented `docs/multiplane_reset_plan.md` steps 1-3 and answered step 6.
Nothing was launched, uploaded or submitted; no GPU hour was spent.

**The fold-1 reviewed-label comparison is recovered, without retraining.** The
Kaggle job's comparison failed on arm output discovery after both 12-epoch arms
had finished: the arms wrote to `reviewed_{arm}_fold1`, the comparison read
`{arm}_fold1`. `tools/recover_reviewed_comparison.py` reruns the same checks on
the downloaded artifacts and `tools/compare_oof.find_arm_file` now locates arm
outputs by content for both the local tool and the on-Kaggle runner. Verified
identical run contracts, split membership (3,478 train / 871 val), folds and
exported checkpoint hashes. Mined-label OOF macro: control **0.881118**,
candidate **0.878032**, delta **-0.003086** against a +0.003 screen, paired
bootstrap **[-0.010543, +0.004496]** — the screen fails and the interval
contains zero. Contusion -0.0328 and MCL -0.0283 breach the -0.020 floor. The
50/50 blend's +0.007393 is two near-identical runs averaging, not label
evidence. Output: `outputs/reviewed_labels_fold1_comparison/`.

**The spatial volume encoder and plane adapters are built and tested.** Residual
3D convolution blocks run between the trunk's feature maps and the trunk's own
pooling head, so the candidate is the same computation as its control until the
zero-initialised residual moves — verified on the real ConvNeXt at 24 slices
and five series, maximum logit difference 0.0, and identical again after export
and reload. They add 1,968,000 parameters to 38,669,422 (+5.1%). Plane adapters
pool sagittal, coronal, axial and an unknown-plane fallback separately and fuse
them with missing planes masked rather than zero-filled; they are a separate
ablation, refused inside the screening pair. New presets `spatial_control`,
`spatial`, `spatial_planes`; existing checkpoints and contracts gain no keys.

**Slice spacing was audited rather than assumed** (`tools/audit_geometry.py`,
`outputs/geometry_audit/`). All 24,371 series carry full geometry. Within a
series spacing is even — largest gap coefficient of variation 5.1e-4. Between
series it is not: after 24-position sampling, a depth-3 kernel spans a median
8.90 mm and 6.89-12.00 mm at the 5th-95th percentiles (ratio 1.74), and 4,247
series (17.4%) are shorter than the cache, so their volumes repeat slices. The
kernel therefore assumes ordering, not spacing.

**Fold-1 screen staged, not launched**: `notebooks/19_spatial_fold1_colab.ipynb`
(A100) and `outputs/kaggle_spatial_fold1/`, gate pre-declared at +0.005 macro
with no finding below -0.020. Run `tools/spatial_smoke.py` on the GPU first —
the A100 step time and peak memory are unmeasured and the campaign estimate
depends on them.

**384 is untested, not tested-and-flat**: shard 0 of 6 built and verified,
training staged and never run, and only a T4 memory/runtime probe (2.74x time,
2.30x memory against 224). It stays a separate resolution ablation.

Full record and numbers: `docs/spatial_multiplane.md`. 803 tests pass, 1 skipped.

## Previous update: 2026-09-14

User reports the latest slice overlay also scored **0.914**; submission ID
not independently verified in this audit. Local inference receipt confirms
18 equal-weight members, three of which are slice transformers. Recomputed
matched folds 1/3: protocol control macro **0.877965**, slice transformer
**0.870383**, fixed 50/50 blend **0.883779** on 1,740 mined-label studies.
The candidate is weaker on both folds; its blend gain is against its paired
control, not the full incumbent. Results: `data/meta/compare_slice_plateau/`.

Next proposed substantial experiment is a separate **384-pixel raw-DICOM
cache and matched protocol ConvNeXt run**, specified in
`docs/resolution_384.md`. Existing 224 cache cannot supply the missing detail.
Local raw sample is 800x800 but covers only one series; full geometry audit
remains required. No new GPU training or submission launched by this audit.

## Previous update: 2026-09-13

### Slice transformer implemented and launched on folds 1 and 3

Post-launch check: both RUNNING, uploaded source and metadata verified exactly;
kernel IDs 134267899 / 134267910. User additionally requested Colab compute
with unchanged Drive shards. Fold-2 notebook prepared at
`notebooks/16_slice_transformer_fold2_colab.ipynb`; **not started** because this
session has no enabled browser control. Select a single Colab GPU, configure
the existing Drive cache folder, and run the notebook. Drive outputs use
`knee_runs/slice_transformer_fold2_v1/attempt_*`. See `docs/slice_transformer.md`.

User authorized both folds and removal of parent timeouts. Version 1 of
`pssv27/knee-slice-transformer-fold1-pair` and
`pssv27/knee-slice-transformer-fold3-pair` pushed successfully. Both compare
fresh **protocol-selection** ConvNeXt-V1 controls to the same model with a
two-layer 256-wide slice transformer before pooling. Both arms use 12 epochs,
16 slices, five series, seed 0 and effective batch 8; common initialization
is checked. All 213 relevant tests passed. Parent smoke/training timeouts are
None; Kaggle's nine-hour platform limit remains. Before launch, GPU quota was
19.97 hours. Results pending; user will report failures. Full implementation,
source hashes and receipts: `docs/slice_transformer.md`. Incumbent is 0.914.

V2 correction: fold 1 timed out in epoch 12 after the 18000-second arm cap,
so it has no completed comparison. Fold 3 failed its standalone gate:
0.870323 control / 0.869772 V2, ACL -0.04129; fixed blend 0.882141 is only a
control-blend result. Earlier retired-reference tables were **gold** metrics.

### ConvNeXt-V2 matched fold-1 experiment launched

User authorized building and running the planned V2 experiment after the
0.914 result. Rebuilt `tools/prepare_kaggle_training_pair.py --candidate
convnextv2 --fold 1` with the corrected input finder and no scheduled diagnostic
stack dumps. Thirty-three focused tests passed. Launched
`pssv27/knee-convnextv2-fold1-pair` **version 1**, kernel ID 134222298, with a
nine-hour cap; status RUNNING. Quota before launch: 34.83 GPU hours remaining.

Private T4 image/input metadata and pulled source bytes match staging; all
three cache shards attached. Source SHA-256:
`a91124550b1d5cd17ee09681babce7982354e8845666a581a3ac2d4aa2184557`.
Receipt: `outputs/kaggle_convnextv2_fold1_receipt/`. Runs a short smoke test,
then fresh baseline and V2 arms, each 12 epochs with identical aggregation
initialization and unchanged uniform inputs. Smoke/full results pending.
Prior matched pairs took about five hours; V2 runtime remains unmeasured.
See `docs/convnextv2_fold1.md`. No new leaderboard entry; incumbent 0.914.

### ConvNeXt-V2 matched fold-3 experiment launched

The same matched baseline-versus-ConvNeXt-V2 pair was staged and launched for
fold 3 after the fold-1 launch. Kernel `pssv27/knee-convnextv2-fold3-pair`,
version 1, is private and uses the same pinned T4 image, three cache shards,
labels and side-classifier inputs. Staging and focused tests passed; source
SHA-256 is `c21148a3c9841456276581daed12780df94f1cb14bfeae606a51ce99ca663f0e`.
Both arms remain fresh 12-epoch runs with uniform selection, shared pooling,
16 slices, five series, seed 0, effective batch 8 and matched aggregation
initialization. Results are pending and are not a leaderboard submission.

### New incumbent: protocol ensemble scored 0.914

Submission **56196386** is COMPLETE at **0.914**, confirmed by the user and
authenticated Kaggle submission listing on September 13. This improves the
previous 0.911 blend by +0.003; the gap to the 0.954 target is now 0.040.

- Exact scored notebook: `pssv27/knee-16-24-protocol-fixed-blend`, version 1,
  kernel ID 134134153. Submitted September 13 at 01:31:11.647 UTC after the
  user explicitly authorized a leaderboard entry. Exactly one entry made.
- Preserve `outputs/ensemble_16_24_protocol/ensemble.json`, the private model
  dataset version 1, and the pulled notebook in
  `outputs/kaggle_protocol_submission_receipt/` as the new incumbent. Keep the
  former 0.911 artifacts intact as the comparison baseline.
- Visible-test execution passed all three rows with zero fallbacks, identical
  model hashes/weights, correct columns/IDs and finite probabilities. Hidden
  test runtime was not supplied by the score listing. The 1.925-hour runtime
  remains a benchmark projection, not a measured scoring-run time.
- Full receipt: `docs/submission_protocol.md` and
  `outputs/kaggle_protocol_submission/competition_receipt.json`.
- Next planned experiment is the independent matched ConvNeXt-V2 fold-1 pair.
  Restage it with the corrected adapter before any future launch: the old
  staged files predate removal of the diagnostic stack-dump timer. No new
  training or submission launched while recording this score.

### Fixed 15-model ensemble passed GPU benchmark; submission staged

User authorized packaging and benchmarking the full protocol addition. Dataset
`pssv27/knee-ensemble-protocol-private` version 1 is READY, verified private
(dataset ID 12000725). Package: `outputs/ensemble_16_24_protocol/ensemble.json`;
15 selected models, weight 1/15 each, exactly bound to the full OOF comparison.

Benchmark `pssv27/knee-protocol-ensemble-benchmark` version 1 is COMPLETE with
internet disabled, pinned T4 image, all three cache shards and a 3600-second cap.
Pulled source bytes and input/visibility metadata match staging. All 15 cached
OOF reproduction checks passed: eight held-out studies per model, maximum
legacy difference 0.000454; protocol checks all exactly match. All 100 raw-DICOM
studies passed with zero errors: mean 5.283 seconds, p95 8.121 seconds, peak
GPU allocation 3.223 GiB (reserved 4.262 GiB), 1300-study projection 1.925 hours.
Sample IDs exactly match the incumbent benchmark (4.496 seconds/study).
Run completed in 630.16 seconds; setup 62.86 seconds.

Fifteen focused package/comparison/benchmark tests, eleven submission-guard
tests and the explicit-ensemble local notebook smoke passed. Submission is
staged locally in `outputs/kaggle_protocol_submission/`, generated by
`tools/prepare_protocol_submission.py`. It binds exact benchmark manifest and
checkpoint hashes, requires completed reference/runtime checks, preserves
inference source, drops cache attachments and uses offline T4 execution.
The user subsequently authorized submission; version 1 scored 0.914, as
recorded above. Details: `docs/benchmark_protocol_results.md` and
`docs/submission_protocol.md`.

### All protocol folds complete; full incumbent comparison passed

Protocol folds 0/1/2/3/4 have completed. Fold 2's version-2 retry completed;
fold 4 version 3 completed all 12 epochs and final exports in 9258.7 seconds
(2.57 hours), with no recurring stack-dump timeout messages. The fixed run
completed after removal of the diagnostic timer implicated in its prior crash.

- Compared all 4349 held-out studies against the exact incumbent's same-fold
  16/24-slice members, using the predeclared equal-three-family addition.
  Pooled mined-label macro **0.876676 -> 0.885228**, delta **+0.008552**,
  paired within-fold bootstrap 95% CI **[+0.006843, +0.010095]**.
- All five folds improve: deltas 0/1/2/3/4 are +0.004947, +0.007814,
  +0.005616, +0.006365 and +0.005405. All twelve pooled finding AUCs improve.
  These are OOF proxy scores, not a new leaderboard result.
- All five selected protocol checkpoint hashes and 220 finite tensors per
  checkpoint verified. Exact train/validation splits, full OOF coverage,
  screened settings/initialization, incumbent hashes and receipts checked.
  Eleven focused comparison tests passed.
- Reproduce with `tools/compare_protocol_incumbent.py --folds 0 1 2 3 4
  --out data/meta/compare_protocol_incumbent_all_folds` using `.venv/bin/python`.
  Readout: `docs/protocol_incumbent.md`; report and prediction artifacts:
  `data/meta/compare_protocol_incumbent_all_folds/`.
- Next at that check: package and benchmark the fixed 15-model ensemble; this
  has now passed as recorded above. No submission launched. The
  public-score 0.911 incumbent is unchanged; independent V2 remains staged.

### Fold 4 diagnostic-timer fix and version 3 relaunch

- Latest failure logs confirm fold 4 **version 2** passed its smoke test,
  completed eight epochs (epoch-8 mined macro 0.866), then SIGSEGV during
  epoch 9 at 6128.9 seconds. The input-discovery fix worked.
- The likely trigger is the adapter's recurring `faulthandler` stack dump:
  rank zero dumped at 1328.442, 2528.444, 3728.446, 4928.452 and 6128.454
  seconds, with the final dump truncated mid-frame immediately before the
  SIGSEGV. Fold 2's saved failure likewise died immediately during its first
  scheduled dump. The earlier description of a 20-minute NCCL/internal
  training timeout was incorrect: this was a diagnostic timer, while the
  actual full-training subprocess limit is five hours. NCCL errors followed
  the rank-zero death. No native core dump is available to prove causality.
- Removed scheduled stack dumps from the staging adapter; retained fatal-signal
  diagnostics and parent smoke/full time limits. Both generated training ASTs
  match version 2 exactly after excluding the removed timer setup/cleanup.
  Screened notebook, embedded modules, inputs and training recipe are unchanged.
- **39 focused tests passed.** Fold 4 **version 3** was pushed with the same
  private T4 image, three cache shards and six-hour job cap; status RUNNING.
  Pulled source SHA-256:
  `4a67152da65bb50c55eb6ce8afc97c4adaf8108d75f03590bf9c52a28d0ac635`.
  Receipt: `outputs/kaggle_protocol_fold4_completion_v3_receipt/`.
  Latest failed source/logs: `outputs/kaggle_protocol_fold4_failure_latest/`.
- Restarts at epoch zero because the screened EMA/SWA callbacks do not support
  exact resume. Version-3 smoke and full completion remain pending. No fold-2
  relaunch or competition submission in this fix; incumbent remains 0.911.

### Resumed: protocol completion on Kaggle

The user returned September 11 evening (September 12 UTC) and asked to continue
because GPU limits reset. This supersedes the paused handoff below. The live
Kaggle API confirmed **21 GPU hours available**, zero used/reserved before
launch, with the next reset September 19 at 00:00 UTC.

- Protocol fold **0, version 1, completed**; fold 2 version 1 errored after
  about 20 minutes in its full run. Kernels: `pssv27/knee-protocol-fold0-completion`
  and `pssv27/knee-protocol-fold2-completion`. Fold 4 version 1 is now RUNNING.
  Each has a six-hour cap and runs the staged smoke test before full training.
- Pulled source bytes and private T4/image/all-three-shard input metadata
  match staging exactly. Receipts are in
  `outputs/kaggle_protocol_fold{0,2}_completion_receipt/`.
- **38 focused tests passed** before launch. Fold 4 is confirmed RUNNING;
  its smoke result is pending. Fold 2's logs show the smoke completed, then
  the full subprocess hit its 20-minute internal timeout and SIGSEGV during
  epoch 2; the NCCL messages are aftermath, not a completed export.
  The smoke exported finite checkpoints and full training reached epoch 2;
  no data, NaN, OOM or contract error appeared. Treat this as a transient
  distributed runtime failure. At the user's request, fold 2 **version 2** was
  relaunched with the exact screened recipe while fold 4 remains running; it
  currently reports RUNNING. Change the runtime only if this retry repeats the
  same fault.
- Fold 4 version 1 failed before training because its direct competition mount
  triggered a recursive raw-DICOM scan in input discovery. The staging adapter
  now skips any input root containing `train.csv`/`train_series.csv`; fold 4
  version 2 was relaunched with this infrastructure fix and reports RUNNING.
- Fold 4 first push was rejected by **Maximum batch GPU session count of 2**;
  after fold 0 completed, version 1 launched successfully and reports RUNNING.
- Next: verify fold 0 artifacts, monitor fold 4, then diagnose and rerun fold 2
  with a targeted DDP/runtime fix if needed. Compare all five protocol folds
  against the incumbent. The independent V2 pair remains staged for later.
  No automatic scheduler was installed. Incumbent **0.911** is unchanged;
  no competition submission was made.

### Previous paused handoff: Colab preparation (superseded)

Previously, the user offered Google Colab with faster GPUs, then explicitly asked to save
the stopping point and continue later. **Work is paused. Do not launch, submit,
poll or schedule anything until the user returns.** No Colab training or new
Kaggle job was launched during this preparation.

- Prepared `notebooks/15_protocol_completion_colab.ipynb`, generated by
  `tools/prepare_colab_protocol_completion.py`; orchestration lives in
  `tools/colab_protocol_runtime.py`. Instructions: `docs/protocol_colab.md`.
- Default queue is protocol folds `[0, 2, 4]`, sequentially on one GPU. Folds
  1 and 3 are already finished; do not retrain them. An A100 is the intended
  runtime if available. Separate sessions can each use one distinct fold.
- Images use the three full local cache shards; no retired 8-slice transfer.
  The screened recipe and embedded model code remain unchanged: 16 slices,
  5 series, 224 pixels, 12 epochs, seed 0, effective batch 8 (accumulation 8).
  Pins timm 1.0.26 / Lightning 2.6.5; retains Colab PyTorch. Contract checks
  allow only GPU count, PyTorch version and fold-specific schedule totals
  to differ, while requiring identical initial model weights. Hardware and
  runtime differences are recorded in `colab_execution.json`.
- Default output is `MyDrive/knee_runs/protocol_completion_colab_v1/foldN/`.
  Best checkpoints and metrics are written to Drive during training; final
  weights, sidecars, OOF/gold predictions and provenance stay there. Each
  attempt is isolated. Completed folds are checksum-verified before skipping.
  **Interrupted folds restart from epoch 0**, preserving partial attempts:
  screened EMA/SWA callbacks do not serialize the state needed for exact resume.
- Verification completed: **44 focused tests passed** (Colab completion,
  Kaggle completion, training-pair staging and incumbent comparison), including
  embedded-source preservation, contract failures, notebook syntax and artifact
  validation against downloaded protocol fold 3. **Not tested on a Colab GPU.**
- Next on return: hand off notebook 15 for upload to Colab, select one GPU,
  authorize Drive and the Kaggle token, then run the input and smoke cells.
  Do not bypass a dependency, initialization or data-integrity failure; inspect
  it before full training. User has not supplied a connected Colab runtime.
  After folds 0/2/4 complete, retrieve receipts/results and evaluate the full
  five-fold protocol addition to the incumbent before any submission decision.

The earlier Kaggle quota snapshot below is historical and must be rechecked
before any future Kaggle launch. The independent V2 pair is still staged only;
its separate GPU smoke passed. Preserve the public-score **0.911** incumbent.

Submission **56135074 completed at 0.911**, confirmed by the user and the
authenticated Kaggle API. The exact fixed 16/24-slice ensemble is now the
incumbent, +0.005 over either family alone; the gap to 0.954 is 0.043.
The user authorized the proposed training experiments and explicitly said GPU
hours should not constrain development. The baseline/protocol pairs passed
screening on folds 1 and 3. Classwise completed both folds but failed its
standalone and incremental-ensemble replication gates; do not expand it.
Its staging tool is `tools/prepare_kaggle_training_pair.py`; outputs are isolated
under `baseline_fold1` and `protocol_fold1`. Preserve the incumbent artifacts.
This authorization supersedes the previous wait-for-result handoff below.
New competition entries are not part of this training job.

**Latest handoff:** protocol addition was compared against the incumbent's
same-fold 16/24-slice OOF proxy. Macro improves 0.884078 to 0.891892 on fold 1
and 0.883052 to 0.889416 on fold 3. Pooled delta +0.011499, CI
[0.008596, 0.014766], with all twelve pooled findings improved. This is not a
new public score. See `docs/protocol_incumbent.md`.

Protocol completion jobs for folds 0/2/4 and the full matched V2 fold-1 pair
are staged but NOT launched. The separate V2 GPU smoke test passed and its
three checkpoint exports were verified; see `docs/convnextv2_fold1.md`.
Kaggle has 1.27 GPU hours remaining as of September 11 around 14:48 UTC,
refreshing at September 12 00:00 UTC (September 11, 20:00 New York). The user
was asked for more quota or another GPU environment. No automatic future job
start was scheduled. Continue with protocol completion once sufficient runtime
is available, then the independent V2 pair. Preserve the scored 0.911 ensemble.

The first paired job failed after completing baseline training: `trainer.fit`
stalled after epoch 12 and the runner timed out. The selected checkpoint is
recovered locally, validation macro 0.871666, and does not require retraining.
Recovery version 2 completed. On 871 matched held-out studies, mined-label macro
AUC is **0.871575 baseline / 0.881118 protocol / 0.885457 fixed 50/50 blend**.
Protocol delta is +0.009543 (paired bootstrap 95% CI +0.002531 to +0.017191),
with no finding down more than 0.02. PF OA improves 0.787674 to 0.816353;
synovitis has no demonstrated improvement, including on the gold diagnostic.
These are local proxy results, not a new leaderboard score.

**Fold-3 replication is COMPLETE**, version 1, verified September 10 around
21:48 UTC: `pssv27/knee-protocol-fold3-pair`. On 869 matched OOF studies,
baseline/protocol/fixed-blend macro is **0.871415 / 0.877224 / 0.886863**.
The screening gate passes, and the blend improves all twelve mined-label AUCs
on both folds. Protocol delta CI on fold 3 includes zero; blend CI does not.
Both runs use native DDP, removing the launcher difference introduced during
fold-1 recovery. Source and private GPU/input metadata were verified after pull.
All six exported checkpoint hashes and OOF split exclusion were verified locally.
Gold-58 macro is 0.887565 / 0.870440 / 0.889122; PF OA and synovitis decrease
on gold despite improving against mined labels. Treat protocol as an ensemble
candidate, not a proven diagnostic gain or a replacement for the incumbent.
See `docs/protocol_fold3.md` for the full readout and next-step criteria.

**Classwise fold 1 is COMPLETE.** Mined macro baseline/classwise/fixed blend:
0.871575 / 0.874049 / 0.883031. Standalone gate FAILED (+0.002474 < +0.003),
but the fixed blend improves all twelve findings. Gold diagnostic macro is
0.856530 / 0.866667 / 0.883020; PF OA and conditioned synovitis point estimates
increase, with wide paired CIs including zero. No proven diagnostic gain.
An exploratory equal-thirds baseline/protocol/classwise blend reaches 0.890261,
versus 0.885457 without classwise. No blend weight search was performed.

**Classwise fold 3 is COMPLETE**, version 1, verified September 11 around
13:30 UTC: `pssv27/knee-classwise-fold3-pair`. Mined macro is
0.871415 / 0.866763 / 0.882532 for baseline/classwise/fixed blend. Standalone
screen FAILED, including an ACL drop of 0.055831. The predeclared equal-thirds
blend reaches 0.889489 versus 0.886863 for baseline/protocol, but delta +0.002626
misses the +0.003 incremental gate and its CI includes zero. Gold PF OA and
conditioned synovitis gains did not repeat. All six model hashes, contracts,
split exclusion and local comparison were verified. See `docs/classwise_fold3.md`.

Do not expand classwise to remaining folds or adjust weights to rescue its gate.
Prioritize the protocol candidate's full incumbent comparison; ConvNeXt-V2 is
the next planned independent architecture test. No new job or submission was
launched during this result check. The public-score incumbent remains 0.911.

### Previous submission handoff (superseded)

**Handoff: waiting for the user to report the result of submission 56135074.**
Do not start another GPU job, tune weights, resubmit, or poll automatically.
Last verified status was Pending at 2026-09-10 02:07 UTC; it is not a current
status check. The user will return with the result. Resume using the
result-dependent steps in `docs/target_0954_plan.md`.

The user confirmed both 16-slice and 24-slice ensembles score **0.906** and
set a new target of **0.954**. Follow [the new evidence-based plan](docs/target_0954_plan.md),
which supersedes the older queue below and in `docs/plan.md`.
After recovering 24-slice fold-1 OOF, all 4,349 studies give a fixed 50/50
blend macro of **0.87668**, versus **0.86532 / 0.86569** separately. These are
mined-label proxy scores, not leaderboard scores. The ten-model package is
`outputs/ensemble_16_24/ensemble.json`. One private T4 benchmark passed all
100 raw-DICOM studies with no errors: 4.50 s/study, 2.50 GiB peak allocation,
and an estimated 1.70 h for 1,300 studies. See
[the measured results](docs/benchmark_16_24_results.md).
Input safeguards, protocol selection, optional finding-specific pooling,
controlled ConvNeXt-V2 experiments, and explicit mixed-geometry inference are
implemented. Existing pending resampling edits were retained. GPU inference
completed; no new training was performed. The user subsequently authorized
one competition entry: **56135074**, notebook `pssv27/knee-16-24-fixed-blend`
version 1, submitted 2026-09-10 02:06 UTC. Status at 02:07 UTC: **Pending**,
no new score yet. See [the submission receipt](docs/submission_16_24.md).
Latest verification: **655 local tests passed**, three visible Kaggle test
studies passed with zero fallbacks, and all ten checkpoint hashes matched.
The submitted notebook is a fixed artifact; experimental presets are
implemented but have not been trained. Preserve the dirty worktree and the
user's pre-existing changes. No commit was made.

## Historical state

Last updated: 2026-09-02. **Synovitis is descoped, the leaderboard is 0.906,
not 0.891, and step 1 is now verified against the shipped cache rather than
assumed.**

**Step 1 (ingest) is verified end to end, artifact as well as code
(2026-09-02).** The 29 GB that trained the 0.906 ensemble reproduces byte for
byte from raw DICOM: 500/500 rebuilt series exact, 0 blank or degenerate
volumes across all 24,371, 0 build errors, no study split across shards, and
`md5(study_uid) % 3` reproducing at 100.00%. Re-run `notebooks/12_verify_cache.ipynb`
(CPU kernel, no GPU hours) after any change under `src/data/`. The mirroring
concern that opened the session is closed: the geometry ruler and the
`Anatomical_Plane` string agree on **24,371/24,371** series, so collapsing the
two implementations is a tidy-up, not a fix. See worklog 2026-09-02.

**Step 3 (labels) is verified, and the label tributary has *not* diverged
(2026-09-02).** `train_labels.parquet` rebuilds from `fuse.build()` with all
104,376 cells identical, and the three numbers the docs quote about it -- 74.8%
extreme targets, mean weight 0.625, 34.5% at `WEIGHT_SILENT` -- are the
artifact's own. Re-run `tools/audit_labels.py` after any change under
`src/labels/`; it takes ten seconds and decodes nothing.

**The labels are precise and incomplete, and that is the ceiling rather than a
bug (2026-09-03).** Against the radiologist image labels they miss **45.8%** of
true findings and false-alarm on **7.2%** of true negatives; 25 of the 210
silent slots are truly positive and are written as zero by construction. The
sample weights do not predict the error -- the full-weight bucket still misses
28.8%. Worst findings: Synovitis 74.1% missed (independent corroboration of its
descoping), Fracture 66.7%, Effusion 60.0%; best: ACL 20.8%. This is why macro
AUC sits at 0.891 while half the positives are missed -- AUC is a ranking
statistic. `notebooks/14_verify_labels.ipynb` runs the whole audit on Kaggle
(CPU, no GPU hours) and **has not been run yet**; its Parts B/D/E/F need the
four extractor dumps added to `knee-labels`.

**The exposure on the label side is the lexicon's reach, and it is a reach
problem rather than a defect.** The lexicon is one of the two signals whose
informativeness sets every sample weight, and it is 3.2x less informative in
Bulgarian than in English *at matched report length* (decile 5: 43.3% against
15.4%), while the LLM's own gap is 1.16x. The cause is named: 15 of the 16
Medial Meniscus anatomy stems are multi-word and require adjacency, so
Bulgarian's definite-article suffix (`медиалния менискус`) and Croatian's
coordination (`medijalni i lateralni menisk`) never match -- 99.1% of Bulgarian
reports contain `мениск` and 0.0% match the stem list. Two consequences, and
only one has a knob: the **target** half shrinks non-English positives (0.855
against 0.904 on equally confident calls) and `LEXICON_WEIGHT = 0` removes it
for a measured 0.002 on gold, i.e. nothing; the **weight** half is identical at
every `LEXICON_WEIGHT`, and gives English 46.0% of the training mass on 40.9%
of the studies. Nothing was changed -- the gold 58 cannot test either half, and
rebuilding the matrix costs five fold retrains. See worklog 2026-09-02 (c).

**The remaining exposure in the image tributary is laterality, not the cache.**
9,968 series (40.9%) carry no side in the metadata and rest entirely on the
side classifier's 98.97% study accuracy -- roughly **45 of 4,407 studies
mirrored backwards**, which transposes four of the twelve findings on each and
is baked into every measurement including the 0.906. Twenty-six further studies
have series that contradict each other on the side and are left unresolved by
design. This is a measured limitation, not a defect; it belongs to notebook 04.

**Synovitis (closed).** Judge it on **effusion-stratified conditional AUC**,
never on macro and never on pooled alone -- pooled 0.701 is reproducible from
the gold effusion label alone (0.698), and the model's own effusion head scores
0.722 as a synovitis classifier. Conditional stands at 0.630, 95% CI
[0.46, 0.79]. Four measurements closed it: contrast is effectively absent
(3 studies of 4,407, none misrouted, none in the gold 58); the frozen path is
*more* gated, not less (0.620, paired -0.011, worse inside the effusion
strata); a stratified ranking loss diverged on a property of the labels; and
the residual parameterisation measured null (paired -0.024 on gold, +0.011 on
871 out-of-fold studies). Anatomical targeting was withdrawn rather than run --
the published result it rested on is subtype classification among patients who
all have synovitis, not detection. `docs/synovitis.html` §08 is the write-up,
with the ceiling arguments and the citations; `plan.md` in the repo root is the
plan it answers. **Only two things reopen this, and neither is compute:** ~150
more gold labels, or contrast-enhanced series.

**The leaderboard is 0.906 — the five-fold ensemble** (submission 55908661,
2026-08-31 06:05, confirmed). **Its five members are the checkpoints in
`models_train/fold{0..4}/convnext_fold*.pt`**, confirmed 2026-09-01, at
selected-epoch val / gold of 0.8814/0.8683, 0.8726/0.8747, 0.8675/0.8659,
0.8736/0.8915 and 0.8697/0.8753. All aux-free, 12 epochs, 16 x 5. Fold 3 is
the 2026-08-31 run, *not* the retired 2026-08-26 one the notebook's bookkeeping
used to name — a fold-3 retrain on 2026-09-01 came in below it on both val and
gold and was not swapped in. Against the 16x5 single fold's 0.889 that is
+0.017, above what the ensemble was expected to be worth. Every comparison
below that treats 0.891 as the ceiling predates it. **Notebook 10 is done.**

**The remaining headroom is PF OA and Lateral Meniscus**, measured out of fold
over 4,349 studies rather than on the gold 58: PF OA 0.773 on 925 positives is
now the weakest finding in the model, Lateral Meniscus 0.825 on 476. The
medial/lateral gap that runs through both compartment pairs is **not**
laterality (identical among studies whose side came from the DICOM tag) and
**not** label quality (the lateral labels are the better ones on gold) -- it is
positives: on Medial Meniscus the model beats its own labels by 0.123, on
Lateral Meniscus it lands on them exactly. Next run is `POS_WEIGHT`, which
`KneeLitModule` has always accepted and no run has ever set; pre-registered in
worklog 2026-08-31 (h).

Last major shape change: 2026-08-28. **The plan changed shape.** `docs/plan.md` no
longer runs Phases A-E; it is a model-selection and training-constraint plan,
and the phase structure it replaced had closed its own levers (A and B done, C
flat inside the gold set's resolution, C2 down 0.009 on the leaderboard).

What is new, and what it rests on:

- **The DINOv2 axis is closed.** Fine-tuned DINOv2-S measured gold 0.820
  against fold 1's 0.861 gate, and at 24 slices converged level with ConvNeXt
  while losing to its own frozen self on three findings. Both notebooks are
  back on `convnext_tiny.fb_in22k_ft_in1k`.
- **Four correctness bugs are fixed** (plan P0). Normalisation now travels in
  the checkpoint; the submission picks series with `np.linspace` the way
  training does, which was wrong for 32.5% of studies; checkpoint recovery no
  longer resolves alphabetically; `MAX_SERIES` is 6.
- **Checkpoint selection is the leading suspect.** Mined val rose to 0.860 at
  epoch 9 and fell to 0.843 by epoch 20 while gold held flat at 0.864-0.869
  over eleven epochs — the early-learning-then-memorisation signature of
  training on noisy labels, which makes mined val a *biased* selector rather
  than a noisy one. EMA and SWA now accumulate alongside the monitored
  checkpoint, so a run ships three candidates and selects on none of them.
- **58 gold studies cannot resolve 0.015.** Hanley-McNeil puts the per-finding
  SE at 0.045-0.08 and the macro SE at 0.015-0.025. The run-to-run sd of 0.004
  measures optimiser noise, not the gold estimate's error. Decisions inside
  0.02 on gold are not decisions.

**Nothing here has been measured yet.** The code changes are in; the 20-epoch
fold-1 run that reads P1 and P2 has not been made. See `docs/plan.md` for the
queue and the gates, and worklog 2026-08-28.

---

## The competition in one paragraph

Predict twelve binary findings per knee MRI study, scored by macro-averaged ROC
AUC. 4,407 training studies, but **only 58 carry radiologist labels** — the
other 4,349 have a free-text radiology report and nothing else. This is a
weak-supervision problem wearing an imaging problem's clothes, and the
leaderboard is won by whoever converts reports into usable labels. Reports exist
in train but **not** in test, so text is a training-time signal only.

**Deadlines:** entry and team merger 2026-10-15, final submission 2026-10-22.

---

## Where things stand

| | public LB |
|---|---|
| Frozen DINOv2-S + aggregation | 0.787 |
| Frozen + fine-tuned, 50/50 blend | 0.839 |
| Fine-tuned ConvNeXt-T, categorical labels | 0.845 |
| Fine-tuned ConvNeXt-T, dual-mode soft labels | 0.827 |
| Fine-tuned ConvNeXt-T, dual-mode sharpened | 0.843 |
| ConvNeXt-T, 16 slices, 3 series | 0.869 |
| ConvNeXt-T, 16 slices, 5 series (fold 1) | 0.889 |
| ConvNeXt-T, 24 slices, 5 series (fold 1) | 0.891 |
| 2026-08-31 06:05, submission 55908661 (**run not yet verified**) | **0.906 ← best** |
| ConvNeXt-T, 16 slices, 5 series + report loss 0.5 (fold 1) | 0.880 |
| ConvNeXt-T, 16 slices, 5 series (**fold 3**) | 0.877 |
| ConvNeXt-T, 16 x 5, 20 epochs + fixed warmup + P0 fixes (fold 1) | 0.888 |
| Leaderboard top | 0.952 |
| Leaderboard 20th | 0.941 |

**We are ~0.05 behind a very tight cluster.** Twenty teams sit within 0.011 of
each other at 0.94+, which means a known recipe exists and the gap is structural
rather than a matter of tuning.

**Two identical configurations land 0.012 apart on the LB (2026-08-26).** Folds
1 and 3, same geometry, same labels, aux-free: 0.889 and 0.877. Nothing about
the models differs but the split. Two consequences, both live: any single-fold
LB comparison of ≲0.012 across *different* folds is uninterpretable (same-fold
comparisons, like C2's 0.880 vs 0.889, are unaffected); and calling 0.891 "the
best config" is partly a statement about which fold drew the friendlier public
split, which is why the final entry should be the fold average rather than the
luckiest member. Fold 3 also carried the *higher* gold (0.867 vs 0.861) with
the lower LB.

### The leading hypothesis was wrong, and the replacement is measured

The original reading: fused labels scored 0.854 against the gold 58, the model
trained on them scored 0.845, so the model sat at its label ceiling rather than
its own — and 0.854 is low for what is essentially reading comprehension.

**That prediction was tested three times and did not hold.** Labels improved to
0.891 (+0.037 on gold) and the leaderboard moved −0.002 across three label
configurations. Two delivery bugs were found and fixed along the way
(target-shape dilution, worth +0.016; a silently retired silence
down-weighting), and neither closed the gap.

**Capacity was the binding constraint.** Doubling slices per series from 8 to
16 — nothing else changed — lifted honest gold from 0.817 to **0.840**, the
largest single gain measured on this project and well outside the ±0.02 noise
floor of the gold 58. Honest gold by configuration:

| configuration | honest gold |
|---|---|
| 8 slices, sharpened labels | 0.827 |
| 8 slices, + silence weights | 0.817 |
| 16 slices, 3 series | 0.840 |
| **16 slices, 5 series** (fold 1) | **0.861** |

The reading now: at 8 of 24 slices the model could not see enough of the knee
to use the label quality it already had. Label work is not exhausted, but it
is no longer the first lever. `docs/plan.md` holds the remaining ladder —
series count (3 of ~5.5) is the next rung, and its per-finding case is
recorded there.

**The label ceiling is the reports themselves (measured 2026-08-26).** Across
the gold 58, only **73% of positive labels are mentioned in their report at
all** — the radiologist marked the rest from the images and the text never
discusses the anatomy. Report coverage predicts extraction AUC at r = +0.65.
Synovitis is the extreme: **37% coverage, AUC 0.706**, its weakness since the
beginning. Of the 14B extractor's 37 missed positives, 25 were never written
down and a further 5 are the report explicitly contradicting the gold
("Lateral menisküs normal." against a positive label); only ~7 are genuinely
fixable misses. Its dominant error is over-assertion — 97 false positives,
all hedged at minimal severity. Inspect any of it with
`python3 tools/inspect_extraction_errors.py [finding]`.

**The mechanism was diagnosed and is concrete.** The categorical extractor ties
**369 of 696 label slots** on a single value:

| finding | studies tied on one value (of 58) |
|---|---|
| Baker's | 41 |
| Fracture | 41 |
| MCL | 39 |
| Synovitis | 37 |

AUC is a ranking metric. It cannot order studies inside a tie block, so more
than half the training signal carries no ordering information at all.

---

## Verified constraints

From the competition's own pages. `docs/rules_notes.md` has the detail and a
snippet to regenerate it through the Kaggle SDK.

| Constraint | Value |
|---|---|
| Submission | Notebooks only |
| Runtime | ≤ 9 hours |
| Internet at submission | **Disabled** |
| Pretrained models | **Allowed** |
| External data | Allowed if freely and publicly available |
| Metric | Macro AUC over 12 findings |
| Output | `submission.csv` |

**A second prize track worth $18,000.** The Efficiency Prize scores
`AUC + RuntimeSeconds / 32400`, so inference cost is a *scored quantity*, not
just a ceiling. The full 9 hours carries a 1.0 penalty; 30 minutes carries
0.056.

**Report text may not be sent to a hosted API.** The rules forbid making
Competition Data available to non-participants, and reports are Competition
Data. The extractor runs locally on a Colab or Kaggle GPU.

---

## Data and assets

| asset | what | where |
|---|---|---|
| Image cache | 24,371 series, 24×224×224 uint8, ~29 GB, 3 shards | Kaggle notebook outputs |
| Test cache | 15 public test series | Kaggle |
| Frozen embeddings | DINOv2-S, 768 dims/slice, 0.9 GB | `knee_embeddings` |
| Grid-4 embeddings | 6,528 dims/slice, 7.6 GB — **worse, see below** | `knee_embeddings_v2` |
| Side predictions | L/R for all 24,371 series | `knee_side_classifier` |
| Fused labels | 4,349 studies × 12 soft targets + weights | `knee-labels` |
| Frozen fold weights | 5 folds | `knee-train` |
| ConvNeXt weights | 1 fold, 12 epochs, all shards | notebook 08 output |

Kaggle limits: **25 GPU-hours per week, one session at a time.**

---

## Pipeline

```
raw DICOM (570 GB, Kaggle only)
  └─ 01 scan        → index_train.parquet          (~820k headers)
      └─ 02 cache   → cache_*.npy + manifests      (~29 GB, 24x224x224 uint8)
          ├─ 04 side classifier → side_predictions.parquet   (98.97% study acc)
          ├─ 05 embeddings      → embeddings_*.npy (frozen path)
          │   └─ 06 train       → knee_fold*.pt
          │       └─ 07 submit  → submission.csv
          └─ 08 finetune        → convnext_fold0.pt / lit_fold0.ckpt
              └─ 10 submit_ensemble → submission.csv  (BLEND=1.0 for tuned only)

train.csv reports
  └─ 03 extract (Colab/Kaggle GPU) → llm_probs.parquet
      └─ fuse (local)              → train_labels.parquet

09 export_for_colab   thins a shard to 8 slices (3.2 GB) so it can transfer
```

Notebooks are **generated** by `tools/make_notebook.py`, which embeds the real
module source so Kaggle and Colab run the same code as the repo. **Edit `src/`
and regenerate; never edit a notebook directly.** A test fails if a notebook is
stale.

---

## Repo layout

```
src/
  config.py              labels, cache geometry, path detection
  data/
    dicom_io.py          header reads, geometric slice sort, laterality
    cache.py             series -> uint8 volume (shared with submission)
    index.py             threaded header scan -> parquet
  labels/
    lexicon.py           multilingual rule extractor
    llm_extract.py       prompts + parsing, categorical AND probability modes
    mode_fusion.py       combine the two LLM modes; four global strategies
    eval_fusion.py       score strategies on gold (LOO, jackknife) locally
    fuse.py              blend extractors into soft targets
    calibrate.py         score an extractor against the gold 58
  model/
    embed.py             frozen DINOv2-S encoder, POOL_GRID variants
    dataset.py           study assembly, mirroring, collation
    net.py               attention pool -> FiLM -> series transformer -> 12 logits
    conditional.py       effusion-stratified conditional AUC, paired bootstrap
    averaging.py         EMA, SWA, uniform and greedy model soups
    splits.py            study-grouped folds, gold-leakage guard
    side_classifier.py   L/R prediction
    lightning.py         DDP training wrapper, DDP-safe metrics and export
tools/
  make_notebook.py           generates every notebook from src/
  synovitis_gate.py          conditional AUC, the Stage 0 gate, paired tests
  series_audit.py            contrast inventory and series-composition check
  error_analysis.py          per-finding table, subgroups, worst studies
  check_notebook.py          executes the Colab notebook's cells locally
  check_kaggle_notebooks.py  runs 01/02 against a fake mini-dataset
  check_submission.py        runs 07 or 10 end to end on a fake test set
docs/
  plan.md                the forward plan, with the gate at every step
  agent_playbook.md      how decisions get made here -- read before acting
  worklog.md             dated session records, with numbers
  attempts/              parked sessions: what was measured, what is open
  rules_notes.md         verified competition constraints
  dev_data.md            how to get a local dev series
tests/                   597 tests
```

---

## What has been tried

### Worked

**Fine-tuned ConvNeXt-T end to end** — gold 0.825, LB 0.845. All three shards,
12 epochs, 3 series x 8 slices per study. Best model so far. Only **one fold**
has been trained; five is the standard next gain.

**Probability-valued label extraction** — breaks the tie problem where it
applies. Nine of twelve findings improved sharply: Contusion 0.662→0.874,
Lateral OA 0.680→0.870, Baker's 0.818→0.957, Synovitis 0.619→0.777, ACL
0.855→0.983.

**Side classifier** — 98.97% study accuracy. The DICOM side tag is blank on 41%
of series and descriptions are anonymised, so this is how laterality is
recovered at all.

### Failed, with the reason

**Spatial patch embeddings (`POOL_GRID = 4`).** Hypothesis: the focal/regional
split (0.695 vs 0.865) is what a mean over 256 patch tokens predicts. Result:
gold 0.770 → **0.747**, eleven of twelve findings worse. 8.5x more input
dimensions against the same 4,349 studies made optimisation harder without
adding usable signal. *Confound recorded:* grid 1 uses an Identity projection
while grid 4 adds 3.3M new parameters, so features and projection changed
together. Control not run — both answers lead to the same next step.

**Frozen + tuned ensemble.** Per-finding gold comparisons suggested strong
complementarity. The leaderboard disagreed: frozen 0.787, blend 0.839, tuned
alone **0.845**. The blend lands between the two single models, which is what
averaging a weak ranker with a strong one does. A caution about how little 58
studies can settle.

**Sharpened meniscus/MCL definitions.** Reasoned from radiology — a tear
requires signal reaching an articular surface — rather than tested. Macro 0.830
→ 0.810, MCL 0.857 → 0.705 with 51 of 58 tied. Reverted.

**Per-finding fusion weights fitted on gold.** Scored 0.860 in-sample, 0.812
under leave-one-out, worse than the better single source. Replaced with a fixed
0.30/0.70 blend chosen from a plateau. **This lesson recurs: 58 studies cannot
support per-finding fitting.**

**The whole of Phase C — extraction v3 (levers 1-4).** Prompt variants, more
self-consistency samples, prompt ensembling, and a 32B categorical
re-extraction: none beat the shipped labels' 0.891 fused gold. The 32B was
*worse* at categorical extraction than the 14B (0.824 vs 0.838 cat-only).
**The lasting result is about the instrument, not the levers:** an
independent replication of the sweep put the run-to-run sd of fused gold
macro at ~0.004, with identical configurations differing by 0.005 while the
entire spread between different configurations was 0.007. The 58-study gold
set cannot resolve a label change smaller than ~0.01; the gate asked for
0.005. Every ranking inside the sweep was noise, and the delete-8 jackknife
could not see it because it resamples studies while holding the extraction
fixed.

**Prompt iteration, more samples, and prompt ensembling (Phase C, levers
1/2/4), in detail.** Three one-idea prompt variants, a 3→5 sample bump, and four fixed
ensembles, all scored on the gold 58 through the full fusion: nothing beat
the shipped 0.891 by the required 0.005. Best variant `granular` 0.888, best
ensemble `phrasings3` 0.893 (+0.002 at 3× the extraction cost). Two lessons
worth more than the runs: **raw extractor macro ranks candidates wrongly** —
`reason_first` led raw by +0.017 and fused last — because fusion needs the
probability mode for only six of twelve findings; and the other six are held
by the **14B** categorical extractor, which is where the remaining room is.

**Report-embedding auxiliary loss (Phase C2).** Best fold-1 val (0.881) and
gold (0.867) ever measured at 16 slices, and LB **0.880** against the aux-free
0.889 on the identical setup. Val labels are mined from the same reports the
loss embeds, so the val gain was partly the model learning to predict reports
rather than findings; gold's +0.006 was inside the 58-study floor. Rank loss
was worse still — it collapsed to 0.026 by memorising its own queue. Both
lambdas ship at 0.0; the code remains as a capacity-phase regulariser
candidate (its reduced-overfitting signature was real).

---

## The open lead: combine both extraction modes

**Status: measured and frozen (2026-08-23).** The 32B gold checkpoint ran and
`python -m src.labels.eval_fusion` settled it:

| scores on gold | macro |
|---|---|
| categorical LLM alone (14B) | 0.838 |
| probability LLM alone (**32B** — was 0.810 on 14B) | 0.860 |
| per-finding oracle of the two | 0.885 |
| **score_average of the two** | **0.889** |
| with 0.10 lexicon blend (the training targets) | **0.891** |

Averaging the two modes beats *choosing* between them per finding — the modes
err in different places, so the mean outranks the oracle. `score_average` and
`rank_average` tied (jackknife 0.54/0.47); the simpler, probability-scaled one
was frozen as `mode_fusion.DEFAULT_STRATEGY`. `LEXICON_WEIGHT` moved
0.30 → 0.10: with the stronger LLM the old weight cost ~0.010, and 0.00–0.15
is a plateau. MCL holds 0.955 and Medial Meniscus 0.951 (the two findings
probability mode used to collapse); nothing sits below Synovitis's 0.795.

Labels went 0.854 → 0.891 against gold. The next artefact is the ~1 h 32B
corpus run producing `llm_probs.parquet`; `fuse.build()` then emits the new
`train_labels.parquet` automatically. The table below records the 14B
per-finding numbers that motivated all this.

The categorical and probability extractors are complementary per finding:

| finding | categorical | probability |
|---|---|---|
| Medial Meniscus | **0.944** | 0.576 |
| MCL | **0.959** | 0.705 |
| PF OA | **0.826** | 0.777 |
| Fracture | 0.752 | **0.885** |
| Contusion | 0.760 | **0.807** |
| Lateral OA | 0.828 | **0.885** |
| Effusion | 0.757 | **0.809** |
| Synovitis | 0.706 | **0.759** |

Categorical wins where reports state a finding crisply. Probability wins where
reports are usually silent and the value is in ranking the studies nobody
mentioned it for. **Per-finding best-of-both scores ~0.869**, against 0.854 for
the labels currently in use.

Medial Meniscus and MCL stay collapsed under probability extraction — 12 and 8
distinct values, 45 and 51 tied — so those two should probably come from the
categorical extractor.

**The corpus run is done (2026-08-23, 32B).** `llm_probs.parquet` lives in
`data/meta/` and `python -m src.labels.fuse` rebuilt `train_labels.parquet`:
macro **0.891** on gold, weakest finding Synovitis 0.792, all twelve above the
0.70 kill threshold. Targets are continuous now — 300–975 distinct values per
finding against the tie-collapsed categorical labels — and target means track
gold prevalence where reports mention a finding.

**Immediate next steps (user, Kaggle):**
1. Upload the new `data/meta/train_labels.parquet` as a new version of the
   `knee-labels` dataset.
2. Update the Kaggle copies of notebooks 08 and 10 from the repo — both
   changed (validation-fold selection + `FOLD` constant; `BLEND = 1.0`).
3. Run notebook 08 with `FOLD = 0` first (~2 GPU-h). Gate: its monitor-only
   gold macro should land ≥ ~0.82 at the selected epoch (the old labels
   produced 0.825 *with* the selection leak; the new labels are 0.037 better,
   so materially less than that means something is wrong — stop and compare
   epoch curves before spending the other four sessions).
4. Folds 1–4 as four more sessions, editing `FOLD` each time. Save & Run All,
   never Quick Save. Upload the five `.pt` files as the `knee-convnext`
   dataset, then submit notebook 10.

**Beware the recurring trap:** per-finding selection on 58 studies is the same
overfit that sank the fusion weights. A robust blend capturing most of the
oracle beats twelve fitted choices.

---

## Decisions made deliberately

**Slice ordering by geometry, not `InstanceNumber`.** Projected
`ImagePositionPatient` onto the slice normal. On the first real series tested,
`InstanceNumber` ran 30→1 — exactly backwards. Sorting by it reverses the volume
with no error raised.

**Mirroring is plane-aware.** For coronal and axial the medial-lateral axis is
in-plane, so mirroring flips columns. For **sagittal it is the slice axis**, so
mirroring reverses slice order. A blanket column flip on a sagittal series does
not swap medial with lateral at all — it reflects anterior-posterior, putting
the patella behind the knee.

**Volumes are cached as acquired, not mirrored.** This is what saved the 29 GB
cache when the sagittal mirroring bug was found later.

**Laterality reaches the frozen model as a covariate** (`is_right`,
`side_known`), because precomputed embeddings cannot be flipped. The end-to-end
path mirrors geometrically instead. Which is better is untested.

**Silence is weighted down, not imputed.** `P(positive | unmentioned)` ranges
from 0.05 (MCL) to 0.58 (Effusion), so silence is informative — but using it as
a *target* was worse than zero on every variant tested.

**Splits are by study, never by series.** A study's ~5.5 series image the same
knee in one session.

**The gold 58 never enter training.** `splits.assert_no_gold_leakage` is called
by the entrypoint. The one known leak — notebook 08's best-epoch checkpointing
selected on gold macro, likely inflating that model's gold figure by 0.01–0.03
— **is fixed:** selection now happens on the validation fold's mined labels
binarised at 0.5, gold is printed as monitoring only, and `FOLD` is a constant
edited per session so five sessions produce `convnext_fold{0..4}.pt`. The
historical 0.825 gold figure for the existing fold-0 checkpoint still carries
that bias. Notebook 10 is now `BLEND = 1.0` (tuned-only, the leaderboard's
verdict) and averages across every fold checkpoint it finds.

---

## Traps already paid for

All guarded by tests now.

- **`.gitignore` `data/` also matched `src/data/`** — anchor with `/data/`.
- **nbformat `source` lines need trailing newlines**, or every cell collapses to
  one line. Validating by joining with `\n` hides it; join with `""`.
- **Writing `src/*.py` does not reload an already-imported module.** The
  bootstrap evicts `sys.modules` entries first. A cache build ran 15 minutes
  against stale code before this was found.
- **A docstring inside `nb.code("""...""")` terminates the string** and breaks
  the generator. Use comments in cell bodies. This happened twice.
- **Kaggle mounts competition data at `/kaggle/input/competitions/<slug>/`** and
  attached notebook output at `/kaggle/input/notebooks/<user>/<notebook>/`.
  Search by content, recursively, skipping the competition tree (820k files).
- **Kaggle slugs replace non-alphanumerics with hyphens** — a notebook titled
  `knee-cache-224-01_shard-1` has slug `knee-cache-224-01-shard-1`.
- **`Quick Save` does not capture `/kaggle/working`.** Only `Save & Run All`
  produces a version with output files. A shard was lost to this.
- **`/kaggle/working` caps at 20 GB** and a shard is ~9.7 GB. One shard per
  notebook.
- **P100 (sm_60) is unsupported** by Kaggle's PyTorch build. Use T4.
- **A process-pool worker defined in a notebook cell cannot be pickled.** It
  lives in `src/data/cache.py` as `build_series_record`.
- **After a forked `fit`, the parent process knows almost nothing.**
  Lightning's launcher transfers `best_model_path` back and leaves
  `best_model_score` at `None` (its source says "TODO: pass also best
  score"), and `CSVLogger` resolves its version in the child, so the parent
  asking afterwards points one version too far. Read the checkpoint from the
  filesystem and the metrics from the CSV; trust nothing the callbacks hold.
  Guarded by `tests/test_lightning.py::TestParentProcessKnowsNothing`.
- **Lightning runs callback hooks before the module hook of the same name.**
  A `Callback.on_validation_epoch_end` reading `callback_metrics` fires
  *before* `LightningModule.on_validation_epoch_end` has logged anything, so
  it reads an empty dict and reports `nan` for every epoch of a perfectly
  healthy run. Read metrics in `on_validation_end` instead, which is where
  `ModelCheckpoint` selects — which is also why selection was never affected.
  Guarded by a real one-epoch Trainer in
  `tests/test_lightning.py::TestEpochReportSeesTheMetrics`.
- **`torch.cuda.is_available()` poisons a fork.** Lightning's `ddp_notebook`
  forks, and a child inheriting a CUDA context dies with "Cannot
  re-initialize CUDA in forked subprocess" — raised from inside Lightning's
  own matmul-precision probe, so the traceback points away from the cause.
  torch's docstring says it plainly: `is_available()` avoids poisoning
  *only* when `PYTORCH_NVML_BASED_CUDA_CHECK=1`, otherwise it calls
  `cuInit`. Use `device_count()` (NVML first), never `get_device_name` or
  `get_device_properties` (both reach `_lazy_init`), and keep any `.to(cuda)`
  after `Trainer.fit`. Guarded by `tests/test_lightning.py::TestForkSafety`.
- **`Sag` in these series descriptions means sagittal**, not Turkish `sağ`
  ("right"). Adding that token to the side patterns would mislabel a large
  fraction of the dataset.
- **vLLM breaks when Colab's torch shifts** — first a torchaudio CUDA mismatch,
  then a missing `libcudart.so.13`. Notebook 03 falls back to `transformers`.
- **Downloading ~10 GB from Kaggle to Colab fails** at ~5.5 GB with
  `IncompleteRead`, consistently. Notebook 09 exports a 3.2 GB thinned shard
  instead.
- **A `persistent=False` buffer does not travel, and a strict load says
  nothing about it.** `KneeNet` registered its input normalisation that way on
  the argument that it is a backbone constant rather than learned state. Every
  construction site then had to supply it again, two scoring sites did not,
  and the strict load restored every weight while leaving normalisation off —
  an ImageNet-normalised trunk scored on raw [0,1]. Measured symptom: gold
  0.864 logged in training, 0.823 from the same weights reloaded. ConvNeXt hid
  it completely, because its statistics are `None`. The buffers are persistent
  now and `_load_from_state_dict` reconciles files written either side of the
  change. Guarded by `tests/test_backbone_normalisation.py`.
- **`sorted(glob(...))[0]` is alphabetical, and "-" sorts before "."** (45 <
  46 < 95). So `lit_fold1-v1.ckpt` beats `lit_fold1.ckpt` beats
  `lit_fold1_old.ckpt`, and Lightning writes those `-vN` suffixes on every
  name collision — a session resumed after a crash silently recovers the
  earlier attempt's weights. Prefer the exact name, then `os.path.getmtime`.
- **Training and submission disagreed on which series to keep.** Training
  spreads its pick with `np.linspace` (`StudyIndex.series_for`); the
  submission took `volumes[:MAX_SERIES]`, a head truncation. For a seven-series
  study that is positions [0, 2, 3, 4, 6] against [0, 1, 2, 3, 4], and 32.5% of
  studies carry more than five series. Shapes agree, nothing raises, and the
  prediction is simply built from the wrong images. Guarded by
  `tests/test_notebook.py::TestTrainingAndSubmissionAgree`.
- **A glob that averages checkpoints will average a fold with itself.**
  Notebook 08 writes three weights per fold (best-val, EMA, SWA), and notebook
  10 averages every `convnext_fold*.pt` it can see. The averages are therefore
  named `avg_fold{N}_ema.pt` / `_swa.pt`, outside that glob, and notebook 10
  refuses to load two families at once.

---

## Method notes for whoever continues

**Verify against the real artefact, not a proxy.** Several bugs here survived a
check that examined the wrong thing: a notebook validated by joining source
lines the same wrong way the generator wrote them; an import "confirmed" by
finding the module's own filename in the text; a test that skipped any cell
starting with a shell magic when the magic sat on line 3. When a check passes,
ask what exactly it looked at.

**The gold 58 are the only honest signal, and they are small.** Gold predicted
the leaderboard within 0.017 for the frozen model, which is why it is trusted —
but it also endorsed an ensemble the leaderboard rejected. Treat per-finding
differences on 58 studies as weak evidence.

**Cheap loops before expensive ones.** The gold checkpoint runs 58 reports in
seconds; the corpus takes an hour; a training run longer still. Two prompt
changes reasoned from domain knowledge both measured worse than what they
replaced.

**Conventions:** all project text is written in a neutral voice with no tooling
attribution, including commit messages and any notebooks pushed publicly. Tests
accompany every non-trivial module.
