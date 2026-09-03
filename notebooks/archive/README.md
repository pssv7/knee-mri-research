# Archived reference notebooks

`08_finetune_reference.ipynb` and its Colab twin are notebook 08 at the
**reference configuration** — the one every measured result on this project
rests on, and the one to fall back to when an experiment goes badly:

    SLICES_PER_SERIES 16    MAX_SERIES 5    EPOCHS 12
    BACKBONE convnext_tiny.fb_in22k_ft_in1k
    LAMBDA_REPORT 0.0       LAMBDA_RANK 0.0        SILENCE_REWEIGHT False
    LAMBDA_STRATIFIED_RANK 0.0   RESIDUAL_BETA None   LAMBDA_COUPLING 0.0

Set `FOLD` per run; it is the only constant meant to change here. This is the
configuration behind folds 0-4 (val 0.8814 / 0.8726 / 0.8675 / 0.8686 / 0.8697)
and the LB 0.889 fold-1 result.

**These are copies, not the source.** `tools/make_notebook.py` generates every
notebook, and regenerating overwrites `notebooks/08_finetune.ipynb` with
whatever the generator currently says — which is how the live copy came to
carry the Stage 1 experiment. Upload the file from this directory to reproduce
a reference run without touching the generator, or set
`LAMBDA_STRATIFIED_RANK = 0.0` and re-run `python tools/make_notebook.py` to
put the generator itself back.

The synovitis code the generator gained on 2026-08-31 is inert at these
settings and does not change training: `KneeNet.couple` returns its input
unchanged with no residual pairs configured, and the ranking queue is only
pushed when one of the lambdas is above zero. The one addition that runs
regardless is `gold_Synovitis_conditional` / `_pooled` logged during
validation, which writes two extra columns to `metrics.csv` and touches
nothing else.

`tests/test_notebook.py::TestTheArchivedReferenceStaysAtReference` fails if any
lever in these files is ever switched on.


---

## `10_submit_ensemble_0906.ipynb` — the submission that scored 0.906

Notebook 10 exactly as it stood at commit `bbbddc4` (2026-08-26), which is the
version in the repo at the moment submission 55908661 scored **0.906** on
2026-08-31 06:05. Archived 2026-09-02 as a fallback, because the live notebook
has changed since and the reproduction has not yet been confirmed.

    BLEND 1.0    TUNED_SLICES 16    TUNED_MAX_SERIES 5
    no FAST_DECODE, no AB_CHECK, no GEOMETRY_CROSSCHECK

Those three are what the live copy has gained since:

- **`FAST_DECODE`** (2026-09-01) stops decoding series and slices the
  submission discards. It reads the same raw files but normalises over a
  shallower stack, so predictions move slightly. **Never verified on real
  data** — the A/B harness that measures it landed in the same change.
- **`AB_CHECK`** scores every study both ways and reports the difference. It
  roughly doubles the loop and does not affect what is submitted.
- **`GEOMETRY_CROSSCHECK`** (2026-09-02) compares the side classifier against
  the DICOM patient frame. Ships as a no-op: `DISAGREE_POLICY = "cnn"` leaves
  every prediction unchanged and only reports the disagreement rate.

**Use this file if the live notebook fails to reproduce 0.906.** Falling back
to it isolates the cause to those three changes, and `FAST_DECODE` is the only
one that can move a prediction.

Geometry constants were verified against the ensemble members before
archiving: the five checkpoints in `models_train/fold{0..4}` are 16 x 5, and
this file scores at 16 x 5. (A note in the project memory claiming notebook 10
shipped `TUNED_SLICES = 24` / `TUNED_MAX_SERIES = 6` from 2026-08-28 does not
match the repo — every version from 2026-08-26 onward reads 16 and 5.)
