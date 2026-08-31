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
