# Series-selection check — inference only, Tiny control, outer0 (2026-09-25)

## Question

The OOF loss analysis found that studies whose protocol selection drops a
distinct (plane, fluid-sensitive, fat-suppressed) protocol score −0.035 macro
[−0.059, −0.009] outside the "Philips" vendor cluster. Does re-selecting series
at inference, with the same trained model, recover accuracy? No training.

## Declared arms (before the run)

`src/model/selection_policies.py`; the frozen `dataset.py` is untouched.

| Policy | Rule | Outer0 studies changed |
| --- | --- | ---: |
| `protocol5` | Frozen rule (reference); must reproduce saved predictions | 0 |
| `sagfluid5` | Frozen choice, unless it drops every sagittal fluid-sensitive series; then force one in | 26 |
| `quality5` | Frozen greedy, but repeats chosen by header quality (thinnest slices, then more slices, then finer pixels) instead of UID spread | 145 |
| `protocol7` | Frozen rule with a cap of 7 | 205 |
| `all` | Every series (up to 11) | 205 |

Acquisition order is not recorded in any available metadata, so a tie-break by
acquisition order was not possible; header quality was used instead. The header
fields come from DICOM tags available at test time.

## Run and verification

Private Kaggle T4 kernel `pssv27/knee-selection-probe-v1` (torch 2.10), reusing
the perturbation kernel body: hash-verified assembly from the attached control
output, a 5-study smoke, then all 691 studies in 4.2 minutes. `protocol5`
reproduced the saved control predictions with max |Δp| **0.0**. The contract's
code hashes match the local files. Artifacts:
`outputs/selection_probe_tiny_outer0_v1_results/` (gitignored).
Scoring: `tools/score_selection_probe.py`, frozen outer0 mined labels, the
project's paired component bootstrap (`paired_proxy_evaluation`, 2,000 draws).

## Results

| Policy | Macro | Δ vs `protocol5` [95% CI] | Largest finding changes |
| --- | ---: | ---: | --- |
| `sagfluid5` | 0.8792 | +0.0004 [−0.0008, +0.0017] | Effusion +0.005 |
| `quality5` | 0.8824 | +0.0035 [−0.0001, +0.0077] | MCL +0.020, Lateral OA +0.009, Synovitis +0.009, ACL −0.006 |
| `protocol7` | 0.8810 | +0.0021 [−0.0007, +0.0052] | Lateral OA +0.012, MCL +0.011, Lateral Meniscus −0.008 |
| `all` | 0.8821 | +0.0033 [+0.0002, +0.0066] | MCL +0.012, Lateral OA +0.011, ACL +0.006 |

## Reading

- **Series selection is a small lever.** The best arms are about +0.003 macro.
  None reaches the project's +0.005 screen gate. Only `all` has an interval that
  excludes zero, narrowly, and it is one of four arms tested on an outer0 set
  that has been inspected before. Treat it as a lead, not a result.
- **The OOF association was mostly confounding.** Studies that drop a protocol
  scored −0.035 on the OOF set, but re-selecting their series recovers about a
  tenth of that. Whatever makes those studies harder is mostly not the dropped
  series.
- **Forcing a sagittal fluid series is null** (+0.0004, 26 studies). The earlier
  meniscus and effusion deficit of such studies was carried by the Philips cluster.
- `protocol7` and `all` give the model more series than it trained with. A
  model trained at those counts could do differently, in either direction.
- `all` raises inference cost in proportion to the extra series.

## If pursued

The affordable confirmation is inference-only: the five protocol224 fold
checkpoints on their own OOF folds (4,349 studies, 1,432 with more than five
series), `quality5` and `all` against `protocol5`, declared before running.
Nothing here changes a submission or a training recipe.
