# Fixed blend submission

**Final result, verified 2026-09-10:** submission **56135074** is COMPLETE
with public score **0.911**. Confirmed by the user and authenticated Kaggle
submission listing. This is the new incumbent, +0.005 over 0.906. Hidden-test
runtime was not returned by that listing. Historical Pending status below is
superseded; the user has since authorized further training experiments.

User authorized exactly one competition entry after reviewing the private GPU
benchmark. Submitted on **2026-09-10 at 02:06:11 UTC**.

- Competition: RSNA Knee Abnormality Detection.
- Submission ID: **56135074**.
- [Notebook](https://www.kaggle.com/code/pssv27/knee-16-24-fixed-blend): **version 1**, kernel ID 133776762.
- Status checked at 02:07 UTC: **Pending**; no public score returned.
- Notebook and model dataset remain private; internet disabled; T4 GPU.
- Nine-hour runtime limit; benchmark projects approximately 1.70 hours for
  1,300 studies, excluding queue delays and allowing for test-composition differences.
- Exactly one entry made. Kaggle reported four submissions remaining for the day.

**Handoff:** the user will report the result. Pending above is the last checked
status, not a live claim. No automatic polling, extra GPU work or resubmission
is requested. On return, follow the outcome branches in `target_0954_plan.md`.

## Fixed configuration

Ten selected ConvNeXt-Tiny checkpoints: five folds at 16 slices and five at
24 slices, five series per model, uniform selection, 0.1 probability weight per
checkpoint. No weight tuning, retraining, rank averaging or flip TTA. Full
DICOM decoding and the benchmarked laterality handling are retained.

All ten checkpoint hashes and the manifest are pinned in the notebook. Its code
was pulled back from Kaggle and verified identical to the staged version before
submission. Only ensemble weights, labels, the side-classifier output and the
competition are attached; training-cache shards are excluded.

## Checks before submission

- Kaggle visible test: three rows, exact columns and study IDs, finite [0,1]
  probabilities, zero fallbacks. Runtime 321.61 seconds including setup.
- CSV SHA-256: `a0499ce6c4b544ac92c8b49410734d17eb10c97eeef616b04049f95397083b9e`.
- All ten checkpoint hashes and equal weights matched the benchmark package.
- Full local suite: **655 passed**; `git diff --check` passed.
- Visible-test validation is an execution check, not an AUC estimate. Kaggle
  reruns notebook version 1 on hidden test data for competition scoring.

Local receipts and outputs:
`outputs/kaggle_submission_receipt/`, `outputs/kaggle_submission_results/`,
and `outputs/kaggle_submission/competition_receipt.json`.
Preparation is reproducible with `tools/prepare_kaggle_submission.py`; that tool
does not launch or submit. Do not repeat the submission command when checking
status. No further entry is authorized by this action.
