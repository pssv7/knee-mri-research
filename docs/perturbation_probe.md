# Perturbation probe — what the Tiny control reacts to (2026-09-25)

## Question and limits

Which controlled input changes move the historical Tiny control's predictions?
Each of the 691 nested outer0 studies was scored unmodified and under 14
declared variants of its own model input. Predictions were compared with the
unmodified pass. **No label was read.** Variants are not labels or medical truth:
a moved prediction shows what the model is sensitive to, not which prediction is
correct. No training, relabelling, submission or TTA change followed.

- Model: `convnext_tiny.fb_in22k_ft_in1k`, 224 px, 16 slices × five series. Checkpoint
  SHA-256 `38520640…`, selected by hash from `pssv27/knee-distinct-sampling-pair-v1`.
- Run: private Kaggle kernel `pssv27/knee-perturbation-probe-v1`, Tesla T4, torch
  2.10.0+cu128, fp16 autocast. 24.0 minutes for 10,365 forward passes; 34 minutes
  of kernel time in total.
- Code: `src/model/perturb.py` (grid and summary), `tools/perturbation_probe.py`
  (runner), `tools/prepare_kaggle_perturbation.py` (kernel). A Colab variant is
  `notebooks/23_perturbation_probe_colab.ipynb`; it was not used.
- Artifacts: `outputs/perturbation_probe_tiny_outer0_v1_results/`.

## Verification

- **Reproduction:** the unmodified pass matches the saved control predictions with
  max |Δp| **0.0** on all 691 studies (same GPU model and torch version). The
  declared 0.01 stop condition was never approached. The noise floor used in the
  "sensitive" rule is therefore zero, so that rule reduces to p95 |Δp| > 0.05.
- **Provenance:** the predictions and summary hashes match the run contract, and
  its code hashes match the local `perturb.py` and `perturbation_probe.py`.
  The kernel assembled 34 of its 37 bundle files from the attached control output
  by SHA-256 and embedded the three new code files.
- **Summary recomputed locally** from the returned predictions: every flag and note
  is identical. Numeric fields agree to 2×10⁻¹⁷, which is floating-point summation
  order on a different machine, not bit-for-bit.

## Results (p95 |Δp| and Spearman against the unmodified pass)

| Variant | p95 \|Δp\| range over 12 findings | Spearman range | Flagged |
| --- | ---: | ---: | ---: |
| Intensity shift ±5 | 0.003–0.011 | ≥0.9996 | 0 |
| Intensity shift ±10 | 0.005–0.024 | ≥0.998 | 0 |
| Intensity shift ±20 | 0.012–0.052 | ≥0.995 | 1 (Medial Meniscus, −20) |
| Noise σ 2 | 0.019–0.047 | ≥0.995 | 0 |
| Noise σ 5 | 0.052–0.099 | 0.979–0.994 | 12 |
| Noise σ 10 | 0.081–0.166 | 0.944–0.986 | 12 |
| Slices: central 12 (drop 2 at each end) | 0.005–0.035 | ≥0.993 | 0 |
| Slices: 12 evenly spaced | 0.066–0.229 | 0.896–0.978 | 12 |
| Slices: every other (8) | 0.112–0.352 | 0.793–0.946 | 12 |
| Side covariate toggled | 0.019–0.048 | ≥0.982 | 0 |
| Left/right mirror | 0.125–0.772 | 0.285–0.944 | 12 |

Per-finding tables are in `full/summary.json`, including the signed mean Δlogit
with a 95% study-bootstrap interval.

**Brightness is not a lever.** Shifting every voxel by up to 20 of 255 levels
barely reorders studies.

**Noise moves scores systematically but mostly keeps the ranking.** At σ 5, every
finding clears 0.05 at p95, while Spearman stays at or above 0.979. The shifts are
directional. At σ 10: Fracture −0.31 logit [−0.34, −0.29], Lateral Meniscus
−0.27, Medial Meniscus −0.22; Synovitis +0.15 and ACL +0.13. Scanners and field
strengths differ in noise level, so this is compatible with the reset's
scanner-shortcut hypothesis, but it does not test it.

**Slice density matters; the outermost slices hardly do.** Dropping two edge
slices at each end changes almost nothing (median |Δp| ≤ 0.001). Keeping the same
extent at lower density does: 12 evenly spaced slices reach p95 0.23 (Medial
Meniscus), and 8 alternate slices reach 0.35 with ACL Spearman 0.79. The shifts
are not uniform: at 8 slices, PF OA rises +0.60 logit [+0.55, +0.65] while Medial
OA falls −0.41. This quantifies the standing trap that inference geometry must
match training geometry.

**The side covariate is nearly ignored.** Inverting `is_right` on every series
(all outer series have a known side) flags nothing: p95 ≤ 0.048 and Spearman
≥ 0.982.

## The mirror result, and what it says about the repo's flip convention

The declared flip variant used the repository's convention: mirror every series
along the plane-aware axis, then swap the Medial/Lateral Meniscus and OA outputs
(`augment.flip_laterality`, `lateral_swap_order`). Under that reading the paired
findings look badly broken: Lateral Meniscus Spearman 0.285, Medial Meniscus 0.484,
and shifts of −1.08 / +0.90 logit.

The swap assumes the model reads a compartment from its **image position**. A
prediction-only check separates that from reading **anatomy**. The two
unflipped meniscus scores are nearly unrelated (Spearman 0.158), so they make a
clean test:

| Flipped raw output | vs same-name unflipped | vs twin unflipped |
| --- | ---: | ---: |
| Medial Meniscus | **0.862** | 0.285 |
| Lateral Meniscus | **0.728** | 0.484 |
| Medial OA | 0.936 | 0.822 |
| Lateral OA | 0.906 | 0.840 |

The flipped image's "Medial Meniscus" output ranks studies like the original
medial meniscus, not the lateral one. The model largely finds the medial
compartment by anatomy even in a mirrored knee. The OA pair is too correlated to
discriminate (unflipped twin Spearman 0.838). This agrees with geometry: a mirrored
right knee is a plausible left knee whose medial meniscus is still medial. It also
agrees with the pipeline's own load-time step: `mirror_volume` reflects left knees
without swapping labels.

Without the swap, the mirror still moves every finding (p95 0.125–0.421,
Spearman 0.728–0.944). Training only ever showed canonical-orientation knees, so
a mirrored input is out of distribution either way.

**Consequences, none of them active:**
- `FLIP_TTA` (notebook 10) averages each paired finding with its twin's output on
  the mirrored image. For this model that mixes medial and lateral evidence. It
  has been off in every submission (`FLIP_TTA = False`; protocol, 16/24 and 384
  packaging assert it off) and should stay off.
- `AugmentConfig.laterality_flip` (default 0) would train with swapped targets on
  mirrored knees, which contradicts the anatomy the model reads. Do not enable it
  in its current form.
- The comment on `config.LATERAL_LABEL_PAIRS` ("a horizontal flip swaps medial
  and lateral, so any augmentation that mirrors the image must also swap these
  pairs") is not supported for whole-knee mirroring. This is recorded here; the
  code has not been changed.

These are measurements of one model's behaviour on predictions only. Whether
flip-with-swap or flip-without-swap would score better was not measured, and no
experiment is proposed automatically.
