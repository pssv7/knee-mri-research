# Knee application implementation

Updated 2026-10-04. Scope: staged execution, configuration, model versioning,
validation, and local result tracking. Run directly against existing MRI caches.

## Implemented design

1. Parse and validate a study request, model version and preprocessing identity.
2. Read acquired cache volumes and aligned series/laterality metadata.
3. Reuse `src.model.artifacts` for checkpoint loading, series selection, mirroring,
   depth sampling, model execution and per-member probabilities.
4. Average probabilities using the manifest's explicit positive model weights.
5. Validate the twelve probabilities and persist the result atomically.
6. Record each study response and a final run receipt. Write a predictions CSV
   only when every requested study succeeds.

This structure gives our project one repeatable application entry point around
its existing models. Changes to weights, recipes or relevant source code produce
new content-based model versions. Traceable inputs and outputs help reproduce
runs and investigate errors. These engineering changes do not imply better AUC.

## Configuration

Use a JSON configuration with exactly these fields. Paths are relative to the
configuration file. `shards` keys must match `cache_path` in the cache manifest.

```json
{
  "schema_version": 1,
  "ensemble": "ensemble.json",
  "manifest": "manifest.parquet",
  "series_metadata": "series.csv",
  "sides": "sides.csv",
  "shards": {"cache_0.npy": "cache_0.npy"}
}
```

Set `sides` to `null` to use the cache manifest's `laterality_resolved` values.
Source caches must be acquired, unmirrored uint8 arrays with the geometry in
`src/config.py`. The application checks positions, dtype, geometry and nonempty
study inputs. Series metadata supplies `SeriesInstanceUID`, `Fluid_Sensitive`,
`Fat_Suppression` and `Anatomical_Plane`.

The ensemble file names members explicitly; no checkpoint directory glob is used:

```json
{"members": [{"checkpoint": "model_fold0.pt", "weight": 1.0}]}
```

Each checkpoint needs the existing `.pt.json` sidecar produced by
`src.model.artifacts.write_metadata`, including weight hash, exact label order,
cache geometry, experiment recipe and split provenance. Register only supported
KneeNet exports with known recipes. Missing metadata is an error; the application
never invents a historical model's configuration. It supports the recipes already
handled by `artifacts.load_ensemble` and `predict_members`; other formats need an
explicit adapter and parity check.

## Run

Create `studies.json` as a JSON array of study IDs, in the desired output order.
Then run from the project root in an environment with the project's modeling
requirements installed:

```sh
.venv/bin/python -m tools.run_knee_application \
  --config /path/to/application.json \
  --studies /path/to/studies.json \
  --output outputs/application_run_001 \
  --device cpu
```

Use `--device cuda` in the existing GPU environment. Output must be a new directory.
An existing directory is never overwritten. The runner executes inference only.

Outputs:

- `run.json`: resolved configuration, study order, device, library versions, timestamps, model
  registry, input SHA-256 hashes, output hashes and final status.
- `responses.jsonl`: one success or failure per study, flushed as work completes.
- `results/*.json`: atomic per-study probability files, keyed by request identity.
- `predictions.csv`: exact project label columns, emitted only after full success.

Input hashes include full cache files, which adds a sequential read before
inference. Keep inputs immutable while a run is executing. A failed study does
not stop later studies; the command returns a nonzero exit status. Initialization
failures after creation of the run directory are recorded in `run.json`. A hard
process termination can leave status `started`; per-study files and the response
journal retain completed work. Automatic resume is not implemented.

## Validation and limits

Tests exercise the complete local batch path using a small saved/reloaded KneeNet,
real cache arrays, metadata and sidecars. Predictions are compared directly with
our existing inference function, including laterality and depth selection. They
also cover changed checkpoint hashes, recipe/version changes, missing studies,
invalid cache positions, conflicting result IDs and persistence failures.
The targeted application and existing inference regression suite passed 49 tests.

```sh
.venv/bin/python -m pytest tests/test_knee_application.py \
  tests/test_knee_application_runtime.py tests/test_target_pipeline.py -q
```

The separate `tools.smoke_knee_application` command uses synthetic constant outputs
for basic plumbing checks only. The trained competition ensemble has not been
promoted to this entry point: its exact exports and sidecars must first be bound
and compared on saved reference studies. Existing training, label preparation
and competition submission entry points continue to work independently.
