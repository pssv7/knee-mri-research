# Revised experiment plan: spatial, multi-plane MRI modelling

This replaces the incremental experiment queue in target_0954_plan.md.
Planning only: implementation and new GPU launches are not performed here.

## Evidence and corrections

- Reviewed labels fold 1: control val macro 0.8811523318 versus candidate
  0.8779833317, delta -0.003169. Both training arms completed; the outer
  comparison used incorrect artifact paths. Recover outputs without retraining.
- User-provided Colab fold 3 logs: selected control val 0.878 versus
  candidate 0.876 (rounded), both epoch 9. This also fails the +0.003 screen.
- Stop expanding these eleven corrections. This does not establish that
  corpus-wide supervision repair is ineffective.
- Earlier advice incorrectly applied the per-finding validation gate to gold
  scores. Gold is diagnostic only. Protocol selection is already represented
  in the 0.914 incumbent; it is not intrinsically an invalid ensemble member.
- Historical slice transformer and V2 experiments did not establish a
  standalone improvement. A higher-resolution pipeline exists; verify its
  actual results before spending more compute on a duplicate run.

## Main hypothesis

Preserve spatial relationships across neighbouring MRI slices and allow each
acquisition plane to learn a distinct representation. The previous slice
transformer works on pooled slice features. The proposed spatial volume
encoder operates before spatial pooling, where anatomy can still be located.
This is a testable hypothesis, not a guaranteed route from 0.914 to 0.954.

## Architecture and input design

1. Start with the existing 24-slice 224px shards. Maintain slice order within
   each series; do not stack different planes as if they formed one volume.
2. Retain pretrained 2D ConvNeXt feature maps, then add small residual 3D
   convolution blocks over slice depth and spatial axes before pooling.
   Compare with a compute-matched 2D control using the same 24 slices.
3. Add separate sagittal, coronal and axial adapters after the shared encoder.
   Pool within each plane, then fuse three plane vectors with explicit masks
   for missing planes. Route unknown-plane series through a separate fallback.
   Keep twelve independent sigmoid outputs and the original training labels.
4. Initially retain the same five selected series to isolate architecture.
   Only after a positive result test additional series and 384px inputs.
5. Use A100 for development. Measure a complete optimizer update, validation,
   checkpoint export/reload and inference before committing to full training.
   Spatial 3D processing of sampled slices must not assume uniform physical
   spacing; document that limitation and audit available geometry metadata.

Multi-plane specialist modelling has precedent in MRNet, but that study uses
different tasks/data and does not predict this competition's score:
https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002699
The specific spatial adapter above is our proposed experiment.

## Execution order and decisions

1. Repair output discovery and execute the complete comparison path on saved
   artifacts. Recover both reviewed-label arms and verify provenance. No
   retraining is needed to resolve a comparison-path failure.
2. Build the spatial model and one matched 24-slice control. Run local tests
   for gradients, missing planes, valid masks, and checkpoint reload; then a
   short GPU smoke through both arms and the final comparison, including OOF.
3. Screen spatial model on fold 1 with fixed split, seed and training schedule.
   If promising, run fold 3, then plane adapters as a separate ablation. Avoid
   changing labels, resolution and architecture simultaneously.
4. Expansion threshold: candidate val macro at least +0.005 over its matched
   control on each screening fold, with no finding below -0.020. Predeclare
   these allocation thresholds; do not lower them after seeing results.
5. Evaluate a fixed 50/50 candidate/incumbent blend on the same held-out IDs,
   using only incumbent members that excluded each study from training.
   Require pooled delta >= +0.003 and a positive paired bootstrap lower bound
   before five-fold expansion. Report seed uncertainty separately; bootstrap
   alone does not measure retraining variance. Repeat a seed before expansion.
6. Verify existing 384 results. If untested, test true 384 cache as a separate
   resolution ablation; never upscale 224 and call it new image information.
7. Benchmark raw-DICOM inference on the deployment GPU, verify offline inputs
   and export agreement, then prepare one fixed submission for approval.

## Supervision and evaluation track

Audit a blinded random sample plus targeted failures for anatomical attribution,
negation and uncertainty. Model disagreement is a review queue, not a label.
Keep existing gold out of tuning. A larger image-labelled holdout would require
qualified image review and explicit resources; report labels alone cannot
resolve report omissions. Do not invent negatives from missing report text.
If architecture and resolution fail the screens, stop expanding models and
reassess supervision and validation before another full training campaign.

## Resource envelope

First commitment is a smoke and one paired fold, followed by a go/no-go.
The observed A100 baseline takes about 58 minutes per 12-epoch arm; the new
model's time and peak memory are unknown until measured. Estimate the remaining
campaign from that measurement. Public score improvement remains unproven.
