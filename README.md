# Knee MRI research toolkit

This repository preserves the reusable engineering work from the 2026 RSNA
Knee Abnormality Detection competition. It contains the data-indexing,
weak-label, model, validation, and local inference code. Competition datasets,
model weights, cached images, and generated outputs are intentionally excluded.

The project predicts twelve findings from a knee MRI study. Its central research
problem is weak supervision: most training studies contain a radiology report
instead of twelve expert image labels. The code therefore treats target
construction, image preparation, model training, and validation as separate,
auditable stages.

## Repository layout

- `src/data/`: DICOM indexing, geometric ordering, caching, and sampling.
- `src/labels/`: report extraction, review, repair, fusion, and audit logic.
- `src/model/`: classifiers, specialist models, checkpoint metadata, and splits.
- `src/application/`: reproducible batch inference and atomic result storage.
- `tools/`: command-line research, audit, training, and application utilities.
- `notebooks/`: the retained local workflow, from indexing through validation.
- `tests/`: unit and integration coverage for the reusable implementation.
- `configs/`: frozen experiment and label-review settings.
- `docs/`: architecture, evidence, rules notes, and completed research reports.

See [PROJECT.md](PROJECT.md) for the retained project state and
[docs/architecture.md](docs/architecture.md) for the original system design.

## Setup

Python 3.12 was used for the retained environment.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
```

Large inputs belong under `data/`, generated artifacts under `outputs/`, and
model weights under `weights/` or `checkpoints/`. These locations are ignored
by Git. Do not commit competition data, patient data, credentials, caches, or
trained checkpoints.

## Local inference

The application runner accepts an explicit ensemble manifest, cache metadata,
and study list. It verifies model metadata and input hashes, writes one atomic
result per study, and produces a prediction CSV only when the complete batch
succeeds.

```bash
.venv/bin/python -m tools.run_knee_application \
  --config /path/to/application.json \
  --studies /path/to/studies.json \
  --output outputs/application_run_001 \
  --device cpu
```

The expected configuration and output contract are documented in
[docs/grooming_application_plan.md](docs/grooming_application_plan.md).

## Scope

This is a research reference, not a clinical device. The repository does not
contain the RSNA or OAI datasets, and recreating an experiment requires obtaining
those assets under their respective terms. Historical metrics describe specific
competition experiments and are not claims of clinical performance.
