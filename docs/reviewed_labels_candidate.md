# Reviewed-label candidate v1

Built with `tools/prepare_reviewed_labels.py`. Artifacts:
`outputs/reviewed_labels_v1/` contains the separate label table, exact correction
ledger, receipt, and 224-pixel protocol `control_fold1.py` / `candidate_fold1.py`.

Eleven reviewed development study–finding targets are changed to zero under the
repository's extraction definitions: explicit intact/no-tear statements,
degeneration without a stated meniscal tear, soft-tissue rather than bone
contusion, and explicitly normal anatomy incorrectly assigned a neighboring
structure's disease. Every change carries its old target, exact source quote,
report hash, and review rationale. Original sample weights are retained.

Six other flagged positive targets are not changed: evidence omission, an
unspecified compartment, or an unidentified cyst is insufficient to establish
a negative. These corrections reflect report interpretation, not new image
adjudication. They are a small reviewed candidate, not general language rules.

## Training staging

Both scripts expect the ORIGINAL `train_labels.parquet` and the existing
224-pixel cache/metadata/side predictions in Kaggle inputs. Do not replace that
input with the candidate label file: doing so could change stratification.
The candidate script embeds the correction ledger and applies it to `folded`
after original split construction, immediately before StudyIndex construction.
The control executes the same code with correction application disabled.
Each writes `label_review_contract.json` with its arm, input hash, correction
hash/count and unchanged-validation/split declarations. Standard run contracts
record initialization and training settings. The candidate artifacts should be
kept with BOTH contracts; ordinary model sidecars do not embed this extra ledger.

Only folds 0/2/4 may be corrected. Fold 1 and fold 3 labels remain untouched.
Both use protocol ConvNeXt V1, 224 pixels, 16 slices, five series, 12 epochs,
effective batch eight. Sources are generated from the established paired
training generator and are staged, not executed or uploaded. They require
the usual Kaggle training environment; they are not standalone Colab launchers.

## Readiness

Local checks cover validation isolation, immutable inputs, report hash/quote
binding, stale targets, duplicate corrections, exact eleven-cell change count,
and generated source syntax. Live GPU execution is untested.

Only eleven targets are changed. There is no evidence that this small patch will
produce a meaningful macro-AUC increase. The remaining audit and separate
report-review controls should be completed before committing to a costly run
or expanding this into a general automatic relabeler. No GPU job was launched.

## Fold-1 result, recovered 2026-09-15

Both arms completed on Kaggle; the comparison that followed them did not, and
failed on arm output discovery rather than on anything the training did. The
artifacts were downloaded intact, so the comparison was rerun locally by
`tools/recover_reviewed_comparison.py` with no retraining. Provenance checked
first: identical run contracts and shared initial weights, identical 3,478
training and 871 validation studies, identical folds, both checkpoints matching
their exported sha256, the same corrections hash, and both contracts declaring
validation labels and folds unchanged.

Scored against the original labels for both arms, on the 871 fold-1 studies:

| macro AUC | control | candidate | fixed 50/50 |
| --- | --- | --- | --- |
| fold 1 | 0.881118 | 0.878032 | 0.888511 |

Candidate minus control **-0.003086**, against the pre-declared +0.003 screen.
The paired 1,000-draw bootstrap interval is **[-0.010543, +0.004496]**: the
screen fails, and the interval contains zero, so this does not establish that
the corrections hurt either. Four findings fall, two past the -0.020 floor
(Contusion -0.0328, MCL -0.0283, PF OA -0.0227, Fracture -0.0162); Baker's
+0.0177 and Medial Meniscus +0.0130 rise. User-reported Colab fold 3 agrees in
direction: control 0.878 against candidate 0.876 at epoch 9.

The 50/50 blend gains +0.007393 over its control with an interval of
[+0.004034, +0.011040]. Two near-identical runs averaging is what that
measures; it says nothing about label quality, and the control it beats is not
the incumbent.

Eleven corrected cells out of 4,349 x 12 could not have settled whether
corpus-wide supervision repair works. What is settled is that this patch does
not clear its screen, and expanding it is not the next thing to spend on.
Output: `outputs/reviewed_labels_fold1_comparison/`.
