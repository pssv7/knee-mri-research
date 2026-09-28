# Optional duplicate-free sampling — implementation

Implemented `distinct_raw_17_23_v1`, disabled by default. No experiment training,
submission, production label change, or model promotion occurred.

`src/data/sampling.py` supplies the shared mapping. For raw counts 17–23 it
chooses 16 distinct raw-position representatives from the existing 24-slot
cache, retains both endpoints, and accounts for depth reversal in left sagittal
volumes. Other lengths keep the original mapping exactly. The helper checks
the repeated slots implied by the raw count and rejects inconsistent pixels.
It cannot reconstruct an unknown historical decode failure; source-index and
manifest provenance remain required.

`StudyVolumeDataset(..., duplicate_free=True)` enables the policy. It requires
16 slices, disallows random resampling, and checks raw-count/cache provenance.
The default remains false. Old notebooks and submission call sites are unchanged.

`cache.build_model_series_volume(headers, side, plane, duplicate_free=...)`
is the new inference parity path for both arms. It decodes and normalizes the
full 24-slot cache equivalent, mirrors, then applies the shared sampler. It
rejects incomplete decoding. It deliberately does not use the historical
16-slice shortcut, whose percentile normalization uses a different population.
Both arms must use the full-depth path for a fair sampler-only comparison.
This may cost more decoding than historical optimized submission inference;
no runtime or leaderboard improvement is claimed.

## Validation and frozen pair

450 distinct tests passed across the new sampler tests, cache, dataset,
baseline-input audit, Lightning and fallback validation suites. The tests cover
raw lengths 1–100, exact unchanged outputs outside 17–23, 16 distinct positions
inside that range, endpoints/order, both sides and all three planes. Pixel
parity tests feed synthetic raw pixels through actual resize/normalization,
the cache-backed training loader and the new inference path. DICOM decoding
itself is mocked; this is not a new raw-DICOM or full-real-cache validation.
Lightning tests include small synthetic CPU fits, not competition experiments.

Negative tests reject inconsistent raw counts, altered repeated slots, missing
provenance, incompatible resampling, and incomplete inference decoding.
Receipt: `outputs/distinct_sampling_pair_v1_checks.json`.

`outputs/distinct_sampling_pair_v1/` contains one outer0 control/candidate pair.
Split, label and class-count files are byte-identical to frozen fallback v2.
The two specifications differ only in design name and sampling policy. Both
bind the same current source hashes and raw counts independently reconstructed
from the header index. Earlier frozen specifications remain unchanged; their
runtime hashes will correctly reject the newly modified source tree. Use their
archived source or the explicit new version, never silently rewrite old hashes.

The singleton runner reads policy from the specification and passes it to all
three loaders. Its preflight checks actual manifest raw counts against the
source-index counts for both arms. The comparator enforces the legacy-versus-
distinct policy pairing and matching provenance, in addition to existing split,
initialization, seed, input and checkpoint checks. It cannot promote an
image-performance claim from proxy labels.

Prepared commands on the cache host:

```sh
python tools/train_nested_proxy_control.py \
  --design outputs/distinct_sampling_pair_v1/control --inputs /path/to/inputs.json
python tools/train_nested_proxy_control.py \
  --design outputs/distinct_sampling_pair_v1/candidate --inputs /path/to/inputs.json
```

These do not train. The prior verified input configuration is saved under
`outputs/kaggle_nested_proxy_preflight_v1_results/nested_proxy_preflight/inputs.json`;
its absolute Kaggle paths must resolve on the host or be rebound with verified
file hashes. The full-cache preflight previously completed for legacy v2 does
not certify this new runner version.

## Full-cache gate completed

Private CPU-only Kaggle kernel `pssv27/knee-distinct-sampling-smoke-v1`, version
1 (135740434), passed both production preflights against all three cache shards.
The job rehashed the arrays, checked raw-count provenance against the source
index, and verified 3,452 eligible studies / 18,955 series. GPU, TPU and internet
were disabled. Executed source, returned package files and input configuration
hashes were verified locally; the frozen pair artifacts remain unchanged.

The bounded real-cache smoke covered 109 distinct retained series: 61 affected
and 48 unaffected. Every affected series changed and supplied 16 distinct
images; all unaffected series were byte-identical to the legacy output. The
sample covered all raw lengths 17–23 and 17 affected left-sagittal series,
including reversal parity and endpoint checks. Actual study assembly passed
for 16 studies / 78 series. Sampling was acquisition-enriched technical testing,
not representative accuracy estimation. No protected series were sampled.

Receipts:

- `outputs/kaggle_distinct_sampling_smoke_v1_results/smoke_receipt.json`
- `outputs/kaggle_distinct_sampling_smoke_v1_results/smoke_series.csv`
- `outputs/kaggle_distinct_sampling_smoke_v1_receipt/verification.json`

The staging tool is `tools/prepare_distinct_sampling_smoke.py`. It reuses the
previous verified preflight's frozen split/label outputs and embeds the new
versioned source and raw-count specifications. No model forward pass, raw-DICOM
decode replay, fitting, or scoring occurred. The earlier synthetic decode tests
and this real-cache smoke provide complementary, explicitly limited evidence.

The [short GPU execution/throughput check](distinct_gpu_probe_20260925.md) has
now passed for both arms, with matched original pretrained initialization,
two transient optimizer steps each and no retained weights or evaluation.
Next is at most the one matched 12-epoch pair specified in
[the input audit](baseline_input_audit_20260924.md) if execution is sound and
affordable. The frozen `protocol.json` records screening thresholds, stop rules
and confirmation limits. No independent clinical labels are available;
predictive benefit remains unknown. The user subsequently authorized the pair;
the [completed pair failed its frozen screen](distinct_training_pair_run.md).
Macro delta +0.000853, 95% interval [-0.007144, +0.008840]; fracture
delta -0.026898 breached the per-finding floor. The branch is stopped and
the sampler stays disabled by default. No confirmation or deployment follows.
