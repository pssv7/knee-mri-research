# Validation fallback implementation — 2026-09-24

The user confirmed that no expert readers are available. The implementable part
of the [reset](validation_reset_20260924.md) is now prepared and checked locally.
Independent image annotation and trustworthy small-gain image-model ranking
remain unavailable. No new model has been trained or promoted, no production
labels have changed, and no submission has been made. The user-reported 0.917
20-model result remains an ensemble reference.

## Frozen artifacts and checks

Use `outputs/validation_fallback_v2/`; `v1` is preserved as preliminary history.
The later bundle includes runtime source hashes and a general grade-boundary
abstention fix discovered by a synthetic multilingual rule test. Neither bundle
is an independent image test set. Do not compare their rule results as a
confirmed improvement or tune further on these reviewed cases.

The final bundle contains 3,452 eligible studies in 3,394 conservative linkage
components. It excludes 955 studies, including historical gold, known previous
review exposure, rule-design cases, the protected text reserve, the future image
pilot, and their transitive known report links. Text equality is not proof of
patient identity; patient/study dates and site identifiers are still unavailable.
Historical model development also used much of this corpus. New outer splits
remove direct checkpoint-selection reuse for future fits, not that history.

| Design | Train | Inner checkpoint selection | Outer evaluation |
|---|---:|---:|---:|
| outer0, outer1 (each) | 2,208 | 553 | 691 |
| outer2, outer3, outer4 (each) | 2,209 | 553 | 690 |
| scanner_stress | 2,444 | 611 | 397 |

All twelve findings have both proxy classes in every role. Outer folds cover
each eligible study exactly once. Components cannot cross roles or protected
exclusions. Scanner stress holds out the largest eligible equipment proxy,
`Philips Medical Systems|Ingenia|3.0`, and its linked components; it does not
establish new-site generalization. It is a separate diagnostic, not additional
independent confirmation of the other folds.

Each design contains `roles.csv`, `proxy_labels.parquet`, `class_counts.csv`, and
`spec.json`. Label sharpening now fits its empirical distribution exclusively
on the training role. The frozen labels therefore differ from the historical
whole-corpus transformation; new AUCs will not be directly comparable with old
selected-fold AUCs. Frozen specifications bind labels, splits, protected
exclusions and runtime code. `status.json` records input provenance;
`manifest.json` hashes the generated files. These are reproducibility controls,
not an access-controlled lock or evidence of independent labels.

Verification: 87 targeted tests passed across `test_validation_fallback.py`,
`test_splits.py`, `test_audit_labels.py`, and `test_repair.py`. Separate checks on
the real bundle verified source/artifact hashes, exclusions, component separation,
outer coverage, all-class support, and float32 target-threshold stability.
Receipt: `outputs/validation_fallback_v2_checks.json`. The runner's file checks
were exercised with synthetic on-disk caches. Full real caches subsequently
passed the CPU preflight on Kaggle, as recorded below. CUDA training was not
exercised; do not describe the GPU path as validated.

## Supervision experiment: stopped at the inexpensive gate

`src/labels/pf_mention_candidate.py` is a review-only proposal generator. It
abstains on grade language, mixed anatomy and recognized uncertainty instead of
rewriting targets. The final existing-review screen produces five proposals:
two match the explicit-miss criterion, two remain uncertain, and one involves
a severity-definition dispute. This 2/5 design agreement fails the exploratory
90% point-agreement gate. It is neither independent precision nor image accuracy.
Details: `pf_candidate_design_screen.json`.

The branch is closed before fresh reserve review, label mutation or model
training. The protected reserve's membership was used for exclusion; its report
content and outcomes were not opened for this screen. Additional disagreement
review remains error analysis, not representative performance estimation.
The image-supervision experiment remains conditional on qualified readers.

## Executable computational fallback

`tools/train_nested_proxy_control.py` implements one fresh protocol224
ConvNeXt-Tiny control: 16 slices, five protocol-selected series, 12 epochs,
effective batch eight, original pretrained initialization, no gold monitoring.
Only the inner-selection loader is passed during fitting. One best checkpoint
is exported; the outer loader is created afterward. There is no averaging or
automatic loop over folds. Keep the earliest exact-tie checkpoint under the
checkpoint callback's strict improvement rule.

On a machine holding the full original 24-slice, 224-pixel caches, provide an
input JSON with this structure. Paths are relative to that JSON or absolute;
replace each hash with the SHA-256 of the actual file. List every cache shard.

```json
{
  "series_meta": {"path": "series_metadata.csv", "sha256": "ACTUAL_SHA256"},
  "side_predictions": {"path": "side_predictions.parquet", "sha256": "ACTUAL_SHA256"},
  "shards": [{
    "array": {"path": "cache.npy", "sha256": "ACTUAL_SHA256"},
    "manifest": {"path": "cache_manifest.parquet", "sha256": "ACTUAL_SHA256"}
  }]
}
```

The preflight verifies exact study/series coverage, cache dimensions and dtype,
positions, laterality, metadata, split membership, targets and weights. Run:

```sh
.venv/bin/python tools/train_nested_proxy_control.py \
  --design outputs/validation_fallback_v2/outer0 --inputs /path/to/inputs.json
```

That command does not train. A subsequent deliberate single-control run adds
`--execute --out outputs/nested_proxy_outer0_control_seed0` in a CUDA environment.
There is currently no surviving candidate to justify a matched training pair.
Do not launch five folds merely because their specifications exist. Prior
two-T4 timing does not validate this runner's single-GPU duration; measure
throughput/memory before budgeting additional fits. A control alone establishes
a new proxy reference, not improvement or image accuracy.

`tools/evaluate_nested_proxy.py` compares frozen singleton outer predictions
after a candidate has independently earned a screen. It requires exact
membership and checkpoint/prediction hashes plus matched initialization,
seed, recipe, runtime and input configuration. An isolated future intervention
must be recorded explicitly; this control runner does not implement candidates.

```sh
.venv/bin/python tools/evaluate_nested_proxy.py \
  --design outputs/validation_fallback_v2/outer0 \
  --control /path/to/control --candidate /path/to/candidate \
  --out /path/to/new_proxy_readout.json
```

The primary endpoint is macro ROC AUC across all twelve findings; a missing
class makes the primary macro unestimable rather than silently averaging eleven.
Per-finding AUC/deltas and paired known-component bootstrap intervals are
reported. Intervals describe fixed-prediction sampling uncertainty, not seed
variation, selection bias, systematic label error, or unknown patient links.
The comparator always records that proxy labels cannot support image-model
promotion. Output files cannot be overwritten; operationally log every readout,
and treat an opened outer set as development data for subsequent changes.

The reset's standalone screen (macro +0.005, no finding worse than -0.02),
paired-seed checks, independent confirmation and stop rules still apply to
image-reference evidence. They must not be silently replaced with proxy-only
promotion. Lower agreement with old mined labels alone does not invalidate
a better supervision method.

## Next action

The full-cache input gate is now complete. The user authorized proceeding, and
private CPU kernel `pssv27/knee-nested-proxy-preflight-v1` (135729182, version 1)
passed on the existing cache host with GPU, TPU and internet disabled. It hashed
all three arrays (29,348,143,488 bytes) and their manifests, then validated the
outer0 design against actual metadata and laterality predictions: 3,452 studies,
18,955 series; 2,208 train / 553 selection / 691 outer. All six frozen designs
share the same eligible population; the remote preflight checked outer0 roles.

The executed source hash and all 26 returned package files match the local
package. Downloaded metadata confirms private CPU/no-internet execution.
Receipts: `outputs/kaggle_nested_proxy_preflight_v1_results/preflight_receipt.json`
and `outputs/kaggle_nested_proxy_preflight_v1_receipt/verification.json`.
Actual input hashes and Kaggle paths are recorded in
`outputs/kaggle_nested_proxy_preflight_v1_results/nested_proxy_preflight/inputs.json`.
The packager is `tools/prepare_nested_proxy_preflight.py`; the existing package
is preserved. No model was loaded, trained, scored or submitted. This verifies
input structure and consistency, not image preprocessing correctness, GPU
execution, or predictive performance.

The PF candidate remains stopped. A subsequent authorized bounded input audit
identified a concrete sampling mechanism; see
[the audit and isolated candidate](baseline_input_audit_20260924.md). Its next
step is an inexpensive sampler/parity implementation, not an automatic training
campaign. Repeating proxy fits or label-model consensus cannot establish
image-level ranking.

The independent-review handoff is already prepared in
`outputs/validation_reset_20260924/`: the 120-study development pilot roster,
complete-series manifest and two blank blinded twelve-finding forms. Preserve
it until qualified review becomes available; reports alone cannot fill these
as image truth. The reset document specifies the first 40-study timing pilot,
representative sampling, positive/negative count and precision planning,
annotation budget, adjudication, uncertainty, and versioning. A locked final
image set has deliberately not been claimed or manufactured.
