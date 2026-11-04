# Report-label repair implementation

**Budget update:** the user declined the proposed $75 extension. The paid
full-corpus continuation is not active. The existing candidate labels remain
usable without new labeling calls.

Status: the 40-report AI development adjudication and 200-report Sonnet
extraction are complete. The user requested **one labeling model, without
further model comparisons**. Continue with Sonnet 5.5; the proposed additional
Opus review is superseded. No image-model training has started from these
new labels. Kaggle execution remains user-managed.

The run accepted **188/200 reports** and excluded twelve: eleven had inconsistent
state/certainty outputs and one failed quote validation. No paid retries were
made. Fresh token cost was $5.711782; shared accounted cost is **$9.822240 / $10**.
The separate candidate file changes **82 finding targets across 55 training
studies**. All selection/outer labels and sample weights remain unchanged.
These changes are experimental AI labels, not confirmed corrections or a
measured accuracy improvement.

The earlier single-model continuation and Opus-plus-Sonnet spending proposals
are historical.

## Which work used Opus?

Opus 5.5 independently annotated the **original 40 reports**, with Sonnet's
answers and old labels withheld. All 40 responses passed validation. It agreed
with Sonnet on 444/468 comparable finding states. The remaining twelve slots
had no accepted Sonnet response. This agreement is not an accuracy estimate.

The completed **200 fresh reports** used the revised Sonnet 5.5 extractor. They
have not been reviewed by Opus. This is a test of extraction operation and
differences from the old extractor, without an independent reference standard.

## Pilot development review

`outputs/report_label_adjudication_v1/ledger.jsonl` records 480 finding slots
from inspection of all forty original reports. The review was AI-assisted and
unblinded; it is development work, not an independent evaluation. There are
39 explicit case decisions and thirteen state revisions relative to Opus.

Fourteen of the fifteen novel positive calls corroborated by both models were
retained. One medial-meniscus case illustrates why agreement is insufficient:
the findings assert a tear, but the equally specific impression says
“Suspected small horizontal tear”. The revised interpretation is uncertain.

The extraction clarifications address findings/impression conflicts, scoped
negation, uncertain thresholds, local edema context and erroneous inference
from neighboring structures. The original report, rubric and literal evidence
remain the source; the extractor must not silently correct a report's typo.
See [frozen clarifications](report_extraction_clarifications_v1.md).

## Fresh extraction and candidate labels

The extraction requests, source hashes, shared-cost ledger and all 200 reports
were frozen before API execution. Old labels and sampling categories are not
sent to the API. The old-label comparison was opened after all 200 finished.
Requests are journaled before sending; invalid outputs are charged and excluded,
with no automatic paid retry. Unresolved request accounting stops the run.

The 200 reports comprise 120 randomly selected component controls, sixty
challenge cases and twenty coverage cases. Report these cohorts separately.
Rejected reports must remain visible in the denominator. Exact quotation and
state/certainty validation test output consistency, not medical correctness.

The candidate builder is deliberately a label-only experiment:

- Only definite positive/negative assertions can propose changed categorical
  calls. Uncertain, unmentioned and unresolved-threshold outputs retain the
  original target.
- An already-positive old call retains its existing severity and target. A
  low fused target alone is not evidence that extraction missed a finding.
- Existing probabilities, lexicon scores, fusion and the control's training-only
  calibration are preserved. The calibration is not refitted on the candidate.
- Sample weights, every selection/outer target, and other fields remain fixed.
- Every numeric change has a ledger entry with old/new targets and report evidence.
- A separate candidate Parquet file is produced; production labels are untouched.

The builder reproduced all 3,452 control study rows exactly with no proposed
repairs. The matched split contains 2,208 training, 553 selection and 691 outer
studies. The preview only covers accepted members of the fresh sample; it is
not a completed relabel of all training reports.

## Matched training handoff

The candidate artifact is
`outputs/extraction_repair_experiment_v1/fresh_sample_preview/candidate_proxy_labels.parquet`.
Its exact changes are recorded alongside it in `changes.jsonl`: 49 new positive
and 33 new negative categorical calls, with 22 numeric targets crossing 0.5.
The readout is `outputs/fresh_report_extraction_v1/comparison/summary.json`.
Random controls accepted 115/120 reports; challenge groups accepted 56/60 and
17/20. These are operational counts, not extraction accuracy.

Sixteen targeted tests passed across extraction, target repair, paired
evaluation and summary accounting. The actual staged preview passed exact
change-ledger, held-out-label and fixed-weight checks. Frozen packet/source
hashes verified, and offline completed resume sent zero requests. GPU training
has not run. The tools below concern image-model evaluation, not an additional
LLM annotation comparison; no second report-labeling model is required.

`tools/label_repair_pair.py` stages control/candidate designs and evaluates the
two resulting runs. Both use the established ConvNeXt-Tiny protocol224 recipe:
16 slices, up to five series, twelve epochs, seed zero, effective batch eight,
original pretraining and the legacy sampler. Keeping this model fixed isolates
the effect of label repair; this does not settle the separate OrthoFoundation
model-choice experiment.

The existing training runner can consume the two designs unchanged. Its legacy
`arm` field identifies the sampler; the new `label_arm` field identifies the
label comparison. Each run is bound to its own training-label hash and the same
split. The evaluator verifies identical initialization, runtime, input caches,
recipe and source hashes, and evaluates against the unchanged control outer
labels. It checks prediction/checkpoint hashes and uses paired component
bootstrap intervals.

The frozen screening rule is macro AUC improvement at least 0.005, no finding's
point decrease worse than 0.02, and a positive lower bound for the paired macro
interval. A passing result still needs confirmation. The endpoint uses mined
report labels and cannot establish clinical image accuracy; bootstrap intervals
also do not measure variation across training seeds.

These build commands have completed; they refuse to overwrite existing outputs:

```sh
.venv/bin/python tools/summarize_fresh_extraction.py
.venv/bin/python tools/label_repair_experiment.py build-preview
.venv/bin/python tools/label_repair_pair.py stage-preview
```

The following are handoff commands, not jobs launched by this implementation.
Use one hashed `inputs.json` referencing the existing full 224 caches for both
arms in the same CUDA environment. Preflight first by omitting `--execute`.
The preview measures only the partial-relabel intervention.

```sh
python tools/train_nested_proxy_control.py \
  --design outputs/extraction_repair_pair_preview_v1/control \
  --inputs inputs.json --out outputs/label_control_run --execute
python tools/train_nested_proxy_control.py \
  --design outputs/extraction_repair_pair_preview_v1/candidate \
  --inputs inputs.json --out outputs/label_candidate_run --execute
python tools/label_repair_pair.py evaluate \
  --design outputs/extraction_repair_pair_preview_v1 \
  --control-dir outputs/label_control_run \
  --candidate-dir outputs/label_candidate_run \
  --out outputs/label_preview_comparison.json
```

## Remaining scope and spending

The frozen training inventory contains 2,178 distinct literal-report requests
for 2,208 training studies. The 200 fresh requests also cover one additional
training study with the same literal report; 1,978 further unique requests are
needed for full coverage. Exact matches can reuse the existing result, including
retaining the original labels for previously rejected outputs.

Full training-set extraction exceeds the shared $10 authorization and requires
a new cap before execution. A larger cap does not authorize a new Kaggle run.
No additional Opus review is planned. The single-model continuation uses the
same frozen Sonnet method and reuses the existing outputs. The 1,978 remaining
unique requests are estimated to cost **$56.49 additional**, for **$66.31 shared
total**, based on the measured fresh-request mean. A proposed **$75 total cap**
provides a margin; it is not yet authorized. The existing $10 cap remains active.
