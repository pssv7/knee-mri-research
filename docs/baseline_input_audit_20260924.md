# Baseline input audit — 2026-09-24

The bounded audit found a reproducible sampling defect worth isolating: the
raw→24→16 path can discard distinct images while repeating others. It did not
establish that repairing this improves clinical accuracy. No training, model
promotion, label change or submission occurred.

## Scope and provenance

Restricted to the 3,452 eligible studies in the frozen fallback, excluding all
955 protected studies and known linked components. Used acquisition metadata,
DICOM header index, historical protocol224 predictions, and existing mined
labels. No reports or protected-reserve outcomes were reviewed. Technical
montage inspection did not assign clinical labels.

Local reproducible audit: `tools/audit_baseline_inputs.py`;
`outputs/baseline_input_audit_v1/` contains per-series measurements, study
summaries, exploratory subgroup metrics, mixed-side review leads and source
hashes. Outputs are new; no historical data replaced.

Private CPU-only Kaggle job: `pssv27/knee-baseline-pixel-audit-v1`, version 1.
The first, larger upload was rejected before a job was created; the reduced
package in `outputs/kaggle_baseline_pixel_audit_v2/` successfully executed.
Results: `outputs/kaggle_baseline_pixel_audit_v1_results/`. Small manifests and
side predictions were rehashed against the previous full-cache preflight.
Arrays were size-checked against its recorded hashes, not rehashed in this run.
The sample contains 96 randomly selected retained series (seed 20260924) plus
24 separately marked technical-risk series. This is series sampling, not a
representative study-level accuracy sample; never pool enrichment for prevalence.

## Verified findings

| Check | Result | Interpretation |
|---|---|---|
| Series selection | 18,955 available; 16,693 retained; 2,262 omitted | The five-series budget discards inputs. |
| Acquisition-plane coverage | No eligible study loses an available plane | No support for a blanket plane-omission correction. |
| Distinct protocol coverage | 427 studies each lose one plane/fluid/fat-suppression combination | Potential information loss; usefulness is not established by coarse metadata. |
| Loader mirror axis vs geometry | Zero mismatches, axis changes or orientation variations >0.01 among retained series | No measured basis for changing the mirror axis globally. |
| Physical aspect ratio | 852 retained series in 599 studies differ from square by >10% multiplicatively | Square resizing deforms physical proportions. No cropping operation was found in the cache builder. |
| Repeated input slices | 2,600 retained series contain fewer than 16 distinct raw images | Includes acquisitions too short to supply 16. |
| Avoidable repeats | 2,479 retained series in 810 studies have ≥16 raw slices but only 12–15 distinct input slices | A correctable sampling loss, independent of labels. |
| Maximum between-input slice gaps | Median 9.90 mm; 95th percentile 13.60 mm; maximum 20.40 mm | Sparse depth coverage; not proof that any particular finding was missed. |
| Pixel sample | 120/120 unique-image counts match header predictions; no zero or flat volumes | Confirms sampling behavior in actual caches; does not validate the entire cache. |
| Laterality vs available tags | 0 disagreements among 9,558 tagged series | Tags contribute to the side pipeline, so agreement is not independent accuracy. |
| Within-study side mixture | 20 studies; 19 remain mixed after series selection | 19 have both L/R source tags; do not overwrite with majority-side guesses. |

A 20-slice source illustrates the defect: building the 24-slice cache repeats
some raw slices; uniformly taking 16 positions from that cache leaves only
**12 unique raw slices**. Raw lengths 17–23 are affected in this corpus.
The alternative of selecting distinct raw-position representatives from the
existing cache can supply 16 distinct images for every raw length ≥16.
Synthetic checks cover lengths 16–100, ordering, and endpoint preservation.
Four targeted tests passed in `tests/test_baseline_input_audit.py`.

The montage shows nonblank, recognizably structured images across planes. It
cannot establish anatomical completeness, clinical assessability, or whether
acquisition boundaries omit relevant anatomy. Full raw-versus-cache pixel
comparison was not performed. Physical aspect calculations use Rows/Columns
and PixelSpacing; they distinguish physical distortion from matrix shape alone.

## Error associations are hypotheses, not causal proof

Historical selected-fold predictions agree less well with mined labels in
some input-risk groups:

| Study group | N | Historical proxy macro AUC |
|---|---:|---:|
| No omitted protocol combination | 3,025 | 0.896109 |
| One omitted protocol combination | 427 | 0.862846 |
| No >10% physical aspect distortion | 2,853 | 0.896308 |
| At least one such distortion | 599 | 0.869637 |
| No repeated input slices | 2,582 | 0.893820 |
| At least one repeated input slice | 870 | 0.885487 |

These are exploratory comparisons over a changed subset, with differing case
mix, finding prevalence and label quality. Historical checkpoints were selected
using these same observations. Some subgroup findings have only 9–20 positives.
No significance or improvement claim follows, and these figures cannot be
compared directly with the old full-corpus macro.

The risks are highly associated with equipment and report-language proxies.
For example, physical aspect distortion occurs in 271/272 eligible MAGNETOM
Vida studies and all 218 Avanto fit studies, but none of 390 Ingenia studies.
All 100 heuristic-German and 74 heuristic-Dutch studies carry this flag. Missing
protocol combinations affect 110/111 Achieva dStream studies. These are not
verified site identities or language ground truth. This confounding prevents
attributing residual differences to preprocessing alone.

No new evidence here demonstrates optimization failure or a representation
ceiling. A larger backbone is not justified by these checks. Supervision bias
and the absence of independent clinical reference labels remain unresolved.

## One prioritized candidate, not a training campaign

**Candidate:** duplicate-free 16-slice sampling only for affected raw lengths
17–23. Keep the existing sampler unchanged for all other lengths, including
short acquisitions. Identify repeated raw positions from the actual cache
construction mapping and manifest count; reject inconsistent manifests rather
than guess. Choose 16 ordered distinct representatives with both endpoints.
Apply precisely the same mapping during train, validation and inference.
Do not change intensity normalization or rebuild caches in this intervention.

**Matched control:** original protocol224 ConvNeXt-Tiny, five protocol-selected
series, old 16-of-24 mapping; same frozen splits, labels, initial weights, seed,
optimizer, 12 epochs and hardware. One checkpoint per arm, selected only on the
inner set. No blending. No candidate has been implemented in the production
loader yet.

**Cost:** no new annotations needed to establish the mechanical correction;
one matched pair is two 12-epoch fits, with unchanged nominal image count and
memory. At the outer0 sizes, at most 4.24 million single-slice inputs across the
pair before accounting for studies with fewer than five series. Budget timing
from a throughput smoke on the actual single-GPU runner; historical two-T4
numbers do not establish its duration. Independent clinical confirmation still
requires qualified annotation and the separate reset protocol/budget.

**Technical gate before any fit:** preserve shape, order, endpoints and all
unaffected inputs; demonstrate 16 unique source positions on affected series;
verify cache/submission index equivalence and unchanged normalization. Record
the protocol as a new version, preserving frozen v2 specifications. Stop on
failed provenance or parity checks.

**If a bounded proxy screen is undertaken:** predeclare one outer0 matched pair;
report twelve-finding macro and per-finding deltas, paired component bootstrap,
and affected/unaffected strata without choosing the winner from those strata.
Use +0.005 macro and no per-finding point drop below -0.02 only as an exploratory
compute-continuation gate, not image-level success. Failure or inconclusive
results end this bounded branch rather than trigger blending or many retries.
The gate is a resource decision, not proof that the correction lacks clinical
value. A lower noisy-label score alone cannot settle that question.

**Confirmation:** freeze the intervention after screening. Any affordable
continuation must use a previously unopened outer fold and paired seeds;
that establishes proxy robustness only. The technical corpus audit has already
used all eligible metadata and historical residuals, so those folds are not
untouched by development history. Final image-performance claims require new
independent image labels; no expert access means that claim remains unsupported.

**Exact next action:** implement and test the isolated sampler behind a disabled
option, with cache/inference parity checks and a revised matched-run contract.
This is more informative than another backbone experiment. Do not start a
training pair until those inexpensive checks pass. Keep aspect-preserving
resizing and extra-series selection as separate, untested hypotheses.
