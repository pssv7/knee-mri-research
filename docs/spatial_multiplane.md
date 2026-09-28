# Spatial multi-plane modelling — implementation record

Implements `docs/multiplane_reset_plan.md`. Steps 1 and 2 are done and
measured; step 3 is staged and needs a GPU session; steps 4, 5 and 7 depend on
results that do not exist yet; step 6 is answered below. Nothing here has been
launched, uploaded or submitted.

## Step 1 — the reviewed-label comparison, recovered without retraining

Both fold-1 arms trained to completion on Kaggle. The comparison that should
have followed them died on `FileNotFoundError:
/kaggle/working/control_fold1/convnext_fold1.pt.json` after 17,328 seconds:
`tools/prepare_reviewed_labels.stage_source` redirects each arm's outputs to
`/kaggle/working/reviewed_{arm}_fold{fold}`, while the launcher's comparison
block read the directory it had created to *run* the script in. Both
directories existed, which is why the failure was a missing file rather than
anything wrong with the runs.

`tools/recover_reviewed_comparison.py` runs the same comparison over the
downloaded artifacts. Discovery is now by content —
`tools/compare_oof.find_arm_file` locates the unique file of a given name under
a directory naming that arm and raises on ambiguity — and the same helper now
backs the on-Kaggle runner, so a relaunch cannot repeat the failure.

Verified before comparing: both arms' run contracts identical (including
`shared_initial_weights_sha256`), identical `train_uids` (3,478) and `val_uids`
(871), identical `folds.parquet`, both checkpoints matching the sha256 they
were exported with, the same corrections ledger hash, and both contracts
declaring validation labels and folds unchanged.

Mined-label OOF on the 871 fold-1 studies, scored against the **original**
labels for both arms:

| macro AUC | control | candidate | fixed 50/50 |
| --- | --- | --- | --- |
| fold 1 | 0.881118 | 0.878032 | 0.888511 |

Candidate minus control is **-0.003086**, against a pre-declared screen of
+0.003. The 1,000-draw paired bootstrap interval for that delta is
**[-0.010543, +0.004496]** — it contains zero, so this measurement does not
establish that the eleven corrections hurt either. Four findings fall, two of
them past the -0.020 floor: Contusion -0.0328, MCL -0.0283, PF OA -0.0227,
Fracture -0.0162. Gains: Baker's +0.0177, Medial Meniscus +0.0130.

The 50/50 blend is +0.007393 over the control with a bootstrap interval of
[+0.004034, +0.011040]. That is two near-identical runs averaging, which is
what averaging does; it is not evidence about label quality, and the blend is
against its own control rather than the incumbent.

The plan quotes 0.8811523318 / 0.8779833317 (delta -0.003169) for these arms.
The recovered figures, computed by the project's own `compare_oof` path,
are 0.8811182047 / 0.8780323593. The difference is under 1e-4 and the verdict
is identical on both: the screen fails and the interval spans zero. Output:
`outputs/reviewed_labels_fold1_comparison/`.

The plan's instruction stands unchanged — stop expanding these eleven
corrections — and this does not show corpus-wide supervision repair to be
ineffective, because eleven cells of 4,349 x 12 could not show that either way.

## Step 2 — the spatial model and its matched control

### What was built

`net.SpatialVolumeEncoder` inserts residual 3D convolution blocks *between* the
trunk's feature maps and the trunk's own pooling head:

    (N*D, 3, H, W) --trunk.forward_features--> (N*D, C, h, w)
      regroup to (N, D, C, h, w), 2 residual blocks of Conv3d(3x3x3) at width 128
      --trunk.forward_head--> (N*D, F) --> existing slice pool, FiLM, series
      transformer, study pool, twelve logits

Using the trunk's own head to finish means the two arms are the *same*
computation when the residual is zero, not merely similar ones. The output
convolution is zero-initialised, so an untrained spatial model reproduces its
control exactly — verified against the real ConvNeXt-Tiny at 24 slices, five
series, 224 pixels: maximum logit difference **0.0**.

`net.PlaneAdapterPool` replaces the study-level attention pool with four
groups — sagittal, coronal, axial, and a fallback for series whose plane tag is
missing or unrecognised. Each group has a zero-initialised residual adapter and
its own attention pool; the group vectors carry learned plane embeddings and
are fused by a second attention pool whose mask excludes planes that were not
acquired. A missing plane is masked, never zero-filled: a zero vector is a
value the fusion would learn to read as "the axial series says nothing is
there". This is a separate ablation and is not in the screening arm.

Presets (`src/model/experiments.py`), each differing from the previous one in
exactly one field, which the tests assert:

| preset | slices | selection | spatial blocks | plane adapters |
| --- | --- | --- | --- | --- |
| `spatial_control` | 24 | protocol | no | no |
| `spatial` | 24 | protocol | yes | no |
| `spatial_planes` | 24 | protocol | yes | yes |

The control is the 24-slice protocol model rather than the 16-slice incumbent,
so the screen varies the 3D path alone. Checkpoints written before this change
gain no state-dict keys and no contract keys; `net.aggregation_defaults`
recovers both variants from the weights, so a scoring site rebuilds the model
the checkpoint was trained as.

### Measured, locally, on the real backbone

`tools/spatial_smoke.py` runs one optimiser update, a validation pass, export
to a bare `KneeNet` state dict, reload and inference, per arm, and asserts
finite losses, that the optimiser moved parameters, and that reloaded weights
score identically. One study, five series, 24 slices, 224 pixels, CPU:

| | control | spatial |
| --- | --- | --- |
| trainable parameters | 38,669,422 | 40,637,422 |
| added by the 3D blocks | — | 1,968,000 (+5.1%) |
| reload max logit gap | 0.0 | 0.0 |

The two arms' logits at initialisation differ by 0.0 (maximum over the batch).

The CPU step times (163 s control, 140 s spatial) are noise — the candidate
cannot really be faster than the control it contains — and are recorded only
to say that they are not a measurement. **The A100 step time and peak memory
are unknown until the smoke runs there**, and the campaign estimate depends on
them. Note that *both* arms cost more than the 58-minute, 12-epoch A100
baseline the plan quotes: at 24 slices over five series each study is 120
backbone passes rather than the incumbent's 80, before the 3D blocks are
counted at all. Run `tools/spatial_smoke.py` on the GPU first; it writes
`outputs/spatial_smoke/smoke.json`.

Local tests: `tests/test_spatial.py` (15) covers gradient flow into every new
module, the zero-init identity, plane membership, missing planes, padded
series, a study with no usable series, checkpoint reload through
`aggregation_defaults`, and the shared-initialisation property the matched pair
depends on. `tests/test_spatial_pair.py` (8) covers the staging.

### The spacing limitation, measured rather than assumed

The plan requires that 3D processing not assume uniform physical spacing.
`tools/audit_geometry.py` measures it from the 819,078-row local DICOM header
index, ordering each series by projecting `ImagePositionPatient` onto its plane
normal exactly as `dicom_io.sort_slices` does. All 24,371 series carry full
geometry. Results (`outputs/geometry_audit/`):

- **Within a series, spacing is even.** The largest coefficient of variation of
  neighbour gaps across the whole corpus is 5.1e-4, and the largest
  max/min gap ratio is 1.002. The concern the plan raises does not apply
  *inside* a series.
- **Between series it varies.** Acquired gap median 3.50 mm (p05 2.20,
  p95 5.01). After the cache's 24-position sampling, one step of the depth axis
  is a median 4.45 mm (p05 3.44, p95 6.00), so a kernel of three spans a median
  8.90 mm and ranges 6.89-12.00 mm across the 5th-95th percentiles, a ratio of
  1.74.
- **4,247 series (17.4%) are shorter than 24 slices**, so their cached volumes
  repeat slices and some depth neighbours are the same image twice.
- Declared `SpacingBetweenSlices`/`SliceThickness` disagrees with the measured
  gap by more than 0.5 mm in 253 series (1.0%).
- In-plane spacing runs 0.167-0.563 mm/pixel (p05-p95) before the cache's
  resize, which is a pre-existing property of the 224 pipeline and not
  introduced here.

So the depth kernel assumes **ordering**, which the cache guarantees, and not
**spacing**, which it does not. Resampling to physical spacing would be a
separate experiment; it is not part of this one and is not a correction the
audit applies.

## Step 3 — staged, not launched

`tools/prepare_colab_spatial_pair.py --fold 1` writes
`notebooks/19_spatial_fold1_colab.ipynb` (A100, one GPU) and the staged pair
under `outputs/kaggle_spatial_fold1/`, reusing the slice-transformer Drive
staging path unchanged because the inputs are the same 224-pixel, 24-slice
cache. `--smoke-only` stages the smoke alone. Fold 3 is `--fold 3`; folds 0, 2
and 4 are refused, as the plan screens on 1 and 3.

Both arms: 12 epochs, seed 0, effective batch 8, five series, protocol
selection, 24 slices, no auxiliary losses. The remote body checks the two run
contracts before comparing and rejects anything but `spatial_encoder` differing
— including a plane-adapter arm, which must be screened separately.

Pre-declared gate, unchanged from the plan and not to be lowered afterwards:
**candidate macro at least +0.005 over its matched control on each screening
fold, with no finding below -0.020.** Gold remains diagnostic and selects
nothing.

## Step 6 — the 384 results, verified

There are none. The state on disk is: shard 0 of 6 of the 384 cache built from
raw DICOM and verified (4,064 series, receipt
`outputs/resolution_cache0_latest/resolution_receipt_00.json`); the fold-1
training staged at `outputs/kaggle_resolution384_fold1/` and never run; and a
T4 memory/runtime probe, which is not an AUC. The probe measured 384 at
1.132 s per microbatch and 4.25 GiB peak against 224's 0.414 s and 1.85 GiB —
2.74x the time and 2.30x the memory, with gradient checkpointing on both.

So 384 is untested, not tested-and-flat, and it remains a separate resolution
ablation behind five more cache shards. Nothing here upscales 224.

## Steps 4, 5 and 7

Not startable yet. Step 4's expansion decision needs the fold-1 screen; step
5's blend needs a candidate worth blending and incumbent members that excluded
each study from training; step 7's raw-DICOM inference benchmark and submission
follow both. No submission has been prepared.

## What a GPU session should do next, in order

1. `python tools/spatial_smoke.py --out outputs/spatial_smoke` on the A100.
   Read the step time and peak memory; they set the campaign estimate.
2. Run `notebooks/19_spatial_fold1_colab.ipynb` — smoke, then both arms, then
   the comparison it writes to `comparison.json` on Drive.
3. Read the gate. If it fails, the plan's instruction is to stop expanding
   models and reassess supervision and validation rather than to try the plane
   adapters anyway.

## Step 3 result — fold 1 FAILED, spatial lever closed (2026-09-16)

Ran `notebooks/19_spatial_fold1_colab.ipynb` on a Colab A100-SXM4-40GB,
attempt `MyDrive/knee_runs/spatial_fold1_v1/attempt_676986918b1b`. The first
attempt failed at the representative probe with `No module named 'src'`: the
probe wrote its sources under a Drive working directory whose cached import
finder never saw `src/` appear (Drive does not bump directory mtime). Fixed in
`tools/prepare_colab_spatial_pair.py` with `importlib.invalidate_caches()`;
`tests/test_spatial_pair.py` reproduces the failure with a frozen directory
mtime. The training arms were unaffected (their bundle already invalidated).

Both actual-data smokes passed and both 12-epoch arms completed with the
matched-run preflight passing. Mined-label OOF on the 871 fold-1 studies:

| macro AUC | control (24-slice protocol) | spatial | fixed 50/50 |
| --- | --- | --- | --- |
| fold 1 | 0.880463 | 0.877601 | 0.887933 |

`screening_gate.passed = false`: macro delta **-0.002862** against the
required +0.005, paired bootstrap **[-0.010086, +0.004138]**. MCL **-0.0532**
breaches the -0.020 floor; Contusion -0.0183. Gains: Lateral Meniscus
+0.0145, Medial Meniscus +0.0112, ACL +0.0096, Baker's +0.0070.

| finding | control | spatial | delta |
| --- | --- | --- | --- |
| ACL | 0.92252 | 0.93214 | +0.00962 |
| MCL | 0.91221 | 0.85901 | -0.05320 |
| Medial Meniscus | 0.86735 | 0.87860 | +0.01124 |
| Lateral Meniscus | 0.82839 | 0.84292 | +0.01453 |
| Medial OA | 0.87724 | 0.88108 | +0.00384 |
| Lateral OA | 0.83224 | 0.83560 | +0.00336 |
| PF OA | 0.78536 | 0.78461 | -0.00075 |
| Effusion | 0.88364 | 0.87992 | -0.00372 |
| Synovitis | 0.91057 | 0.90850 | -0.00207 |
| Baker's | 0.93032 | 0.93728 | +0.00696 |
| Contusion | 0.86230 | 0.84400 | -0.01830 |
| Fracture | 0.95341 | 0.94756 | -0.00585 |

The 50/50 blend's +0.007470 (CI [+0.003895, +0.011050]) is against this run's
own control, not the incumbent — two near-identical models averaging, the same
pattern as the reviewed-label and slice-transformer pairs. It is not evidence
for the 3D path. Gold (diagnostic only): control 0.877 selected / EMA 0.876,
spatial 0.870 selected / EMA 0.878, all inside the 58-study noise floor.

Decision, per the plan: **stop the spatial lever.** No fold 3, no seed repeat,
no plane-adapter ablation, no spatial members in any submission recipe.

Also measured:
- **24 slices did not beat 16 on fold 1.** The 24-slice protocol control's
  0.880463 sits level with the 16-slice protocol arm's 0.881118 on the same
  871 studies (different runs; not a matched pair).
- **A100 cost:** 6.5 min/epoch control, 6.7 min/epoch spatial, about 80 min
  per 12-epoch arm at 24 slices, five series, microbatch 1 × accumulation 8.
  Selected epochs 7 (control) and 9 (spatial).

Next lever: the 384-pixel resolution pair, `notebooks/20_resolution_pair_fold1_colab.ipynb`,
same +0.005 / -0.020 gate. If it also fails, the plan's stop rule applies.
