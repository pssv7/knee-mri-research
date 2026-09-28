# Silence-masking screen — OA, Synovitis, Baker's, Contusion (2026-09-27)

## Question and design

The OOF loss analysis attributed most of the OA, Synovitis, Baker's and Contusion
loss to "silent" negatives: reports that never mention the finding, carried as
target ~0.02 at weight 0.25, and likely including unmentioned positives. The user
chose to test masking them (partly revisiting D7) rather than relabelling.

- Candidate: the frozen runner with `src/model/silence_mask.py` applied after
  preflight. In **training rows only**, silent cells (weight 0.25) of Medial OA,
  Lateral OA, PF OA, Synovitis, Baker's and Contusion are weighted 0. Cells masked:
  1,037 / 1,210 / 1,013 / 1,824 / 1,070 / 958 of 2,208 training studies. Targets
  are untouched; checkpoint selection reads targets, not weights.
- Control: the unmodified frozen runner, concurrently. Private Kaggle kernel
  `pssv27/knee-silence-screen-v1`, one T4 per arm, 3.05 h. The control again
  reproduced the historical control exactly (macro Δ 0.0).
- **Declared primary endpoint:** AUC on explicitly labelled cells (weight ≥ 0.7),
  averaged over the six masked findings. Pass: ≥ +0.005, and eligible for
  confirmation only with a positive paired lower bound. Guard: the six unmasked
  findings no worse than −0.020 on full labels. Standard macro reported, not
  gated. 2,000 component-bootstrap draws, all valid.

## Result: failed

| Endpoint | Δ candidate − control [95% CI] |
| --- | ---: |
| **Explicit-label AUC, mean of six masked findings (primary)** | **−0.0001 [−0.0160, +0.0166]** |
| Standard macro, all twelve (not gated) | −0.0068 [−0.0169, +0.0033] |

| Finding | Explicit AUC control → candidate (Δ [95% CI]) | Full-label Δ |
| --- | --- | ---: |
| Medial OA* | 0.867 → 0.852 (−0.015 [−0.038, +0.008]) | −0.027 |
| Lateral OA* | 0.827 → 0.834 (+0.006 [−0.026, +0.039]) | −0.013 |
| PF OA* | 0.742 → 0.779 (+0.036 [+0.000, +0.074]) | −0.008 |
| Synovitis* | 0.788 → 0.794 (+0.007 [−0.061, +0.072]) | −0.009 |
| Baker's* | 0.894 → 0.895 (+0.002 [−0.029, +0.033]) | −0.016 |
| Contusion* | 0.875 → 0.838 (**−0.037 [−0.074, −0.007]**) | −0.046 |
| ACL | 0.920 → 0.940 (+0.020 [−0.002, +0.043]) | +0.022 |
| MCL | 0.893 → 0.881 (−0.012 [−0.088, +0.064]) | −0.005 |
| Medial Meniscus | 0.879 → 0.893 (+0.014 [−0.004, +0.030]) | +0.009 |
| Lateral Meniscus | 0.839 → 0.857 (+0.018 [−0.031, +0.068]) | +0.021 |
| Effusion | 0.900 → 0.898 (−0.002 [−0.023, +0.017]) | −0.003 |
| Fracture | 0.951 → 0.941 (−0.010 [−0.038, +0.016]) | −0.007 |

\* masked finding. The guard held (worst unmasked full-label Δ: Fracture −0.007).

## Reading

- **Masking silence does not improve explicit-label accuracy overall.** Gains and
  losses cancel. The primary endpoint is flat, and Contusion clearly gets worse.
  Stop this configuration.
- **PF OA +0.036 sits at the edge of its interval** (lower bound 0.000). It is one
  finding among twelve, not the declared endpoint, so it is a lead at most. It is
  the finding with the audited ~22% missed-positive rate, which fits the
  mechanism. A PF OA-only mask would be a new, separately declared screen.
- **Unplanned side effect:** the unmasked ligament and meniscus findings rose on
  full labels (ACL +0.022, Lateral Meniscus +0.021, Medial Meniscus +0.009); all
  intervals span zero. With one study per step and loss normalised by the summed
  weights, masking shifts each study's gradient share toward the unmasked
  findings. That is a hypothesis for a declared test, not a result. Trajectory
  noise of unknown size (see the determinism note in `docs/mirror_screen.md`)
  applies to every per-finding difference here.

Artifacts: `outputs/silence_screen_v1_results/` (gitignored). Tests:
`tests/test_silence_screen.py`.
