# Mirror-augmentation screen — labels kept, nested outer0 (2026-09-26)

## Question and design

The perturbation probe found that mirrored knees cost the model-driven findings
(ACL −0.087, Lateral Meniscus −0.080, MCL −0.058 against written labels), and that
mirrors must keep medial/lateral labels (+0.028 over the repository's swap). Does
training on mirrored knees, labels kept, improve accuracy? User-authorised
training, departing from the 2026-09-24 reset's "no training before new evidence
gates".

- Candidate: the frozen runner with its training loader wrapped by
  `src/model/mirror_augment.py`. Each training study is mirrored with probability
  0.5 along each series' plane axis; targets, weights and `is_right` are kept.
  Everything else is frozen (Tiny, 224, 16×5, protocol selection, legacy jitter,
  seed 0, 12 epochs).
- Control: the unmodified frozen runner, trained concurrently.
- Private Kaggle kernel `pssv27/knee-mirror-screen-v1`, one T4 per arm, torch
  2.10.0+cu128; 3.08 h. Frozen evaluator (`tools/evaluate_nested_proxy.py`),
  2,000-draw component bootstrap on outer0 mined labels.
- Declared gate (unchanged): macro ≥ +0.005 and no finding below −0.020; eligible
  for confirmation only with a positive paired lower bound.

## Result: failed

| | Macro |
| --- | ---: |
| Control | 0.878838 |
| Mirror candidate | 0.878001 |
| Δ [95% CI] | **−0.0008 [−0.0084, +0.0072]** |

Per finding (candidate − control): ACL +0.000, MCL **−0.022** (breaches the
floor), Medial Meniscus −0.010, Lateral Meniscus +0.006, Medial OA +0.002,
Lateral OA +0.008, PF OA +0.008, Effusion +0.006, Synovitis +0.015, Baker's
−0.007, Contusion −0.006, Fracture −0.011. Every per-finding interval spans zero.

The targeted findings did not improve. Mirror robustness at inference is a real
weakness, but mirrored inputs occur only through side-classifier errors (2–5% of
series). Teaching invariance to them buys nothing measurable and appears to cost
MCL. No confirmation is proposed.

## Determinism finding

The concurrent control reproduced the historical Kaggle control **bit for bit**:
the same selected checkpoint (SHA-256 `38520640…`) and identical outer predictions
(macro Δ 0.0). On this Kaggle T4 / torch 2.10 setup the frozen recipe is fully
deterministic. Consequences:
- A matched pair on this setup needs no fresh control; the historical control is
  the control, provided inputs and runtime match.
- Zero rerun variation is **not** a seed-variance estimate. Any change alters the
  training trajectory chaotically, so screen deltas still combine the change with
  trajectory noise of unknown size. A seed repeat is the only way to measure it.

## Bookkeeping error

`screen_receipt.json` in this run says `"training_launched": false`. It is wrong:
both arms trained. The field came from the probe kernel's receipt template. The
downloaded artifact is left as produced; the generator now sets
`training_launched=True` for pair kernels.

Artifacts: `outputs/mirror_screen_v1_results/` (gitignored). Tests:
`tests/test_mirror_screen.py` (both arms trained on CPU through the frozen
evaluator before launch).
