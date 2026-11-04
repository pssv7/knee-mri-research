# Prepared report adjudication and fresh evaluation

The user's chosen review mode is **AI-assisted only**. Both packets are prepared
and frozen. They support model-consistency checks, report-error investigation
and later candidate development; they do not supply independent human reference
labels or establish clinical/image accuracy.

## Open the right packet

- [Pilot adjudication viewer](../outputs/report_label_evaluation_v1/reviewer/pilot/review.html):
  40 original pilot reports, all 480 finding slots. This includes the 24 state
  disagreements, 15 corroborated missed-positive candidates, model agreements
  and the report rejected by Sonnet. These are exposed development cases.
- [Fresh evaluation viewer](../outputs/report_label_evaluation_v1/reviewer/fresh/review.html):
  200 additional reports, all 2,400 finding slots. Keep this set closed to
  prompt/rule development. Freeze the candidate extraction recipe before using
  its results.

Each viewer contains the original report, fixed rubric and blank annotations.
It supports browser-local draft saving, JSON export and importing a saved draft.
The corresponding folder also has `packet.json`, `reports.jsonl` and
`annotations.blank.json` for programmatic AI review. Record the actual AI model
and version in `reviewer`; never describe its annotations as human review.

Model answers, real study IDs, selection reasons and old extractor outputs are
in the separate `custodian/` directory. Those files are comparison aids to open
after annotations and the extraction recipe have been frozen. The viewer does
not load those files or use an external service. The rubric remains unchanged.

## Fresh sample design

| Cohort | Reports | Purpose |
|---|---:|---|
| Random controls | 120 | Estimate agreement and operational rates among eligible component representatives |
| Priority challenge | 60 | PF OA, Lateral Meniscus and Contusion: ten old-positive and ten old-nonpositive/silent cases per finding |
| Coverage challenge | 20 | Top up old-positive and language coverage, then draw a random supplement from remaining components |
| Total | 200 | All twelve findings reviewed for every report |

Controls were selected first, without model-disagreement filtering. Changing an
enriched quota cannot change the control draw. Challenge cases must be reported
separately; pooling them with controls would change the estimand. The random
sample represents the eligible component representatives, not all competition
studies. It is a bounded next-stage sample, not a power guarantee for a small
accuracy improvement.

Selection used 2,130 eligible training components. The exclusion inventory covers
901 studies / 895 components from recorded prior review, the completed pilot,
gold labels and the old protected review reserve. All selection/outer roles are
also excluded. Known exact/near-duplicate links propagate exclusions across
components. One report representative per component is selected deterministically
before sampling; all inputs, source code, seeds and output hashes are saved.

The fresh sample has no recorded report-review or Sonnet/Opus-pilot overlap.
Older ad hoc readings were not completely logged, and historical image-model
training is not excluded. This is an extraction-evaluation sample from the
training pool, **not a newly untouched image-model holdout**.

Language groups are heuristic, not verified language labels:

| Group | Reports |
|---|---:|
| English | 81 |
| Spanish | 27 |
| Greek | 20 |
| Croatian | 18 |
| Unknown | 15 |
| Turkish | 14 |
| Bulgarian | 13 |
| German | 4 |
| Dutch | 4 |
| French | 4 |

All planned quotas were met, including at least five old-positive calls per
finding and four cases per eligible language group. These are **screening
counts**, not confirmed findings. For example, there are ten old-positive
Fracture calls; their correctness has not been established.

## Workflow with AI-only review

1. Use the pilot packet to resolve or explicitly preserve the known
   disagreements under the fixed rubric. This is AI-assisted development
   adjudication. Review agreements as well; shared mistakes remain possible.
2. Freeze any revised extraction instructions and parser as a new version.
   Record model, effort and response schema before running the fresh packet.
3. Generate fresh annotations independently of the old answers, retain failures
   and abstentions, then compare after all intended responses are sealed. Report
   random-control and challenge results separately, including uncertainty,
   silence and per-finding/language coverage. Call the result model agreement or
   AI-assisted adjudication, not independently measured extraction accuracy.
4. Prepare only evidence-supported candidate repairs, keeping unresolved cases
   unchanged. Test extraction changes separately from severity/fusion changes in
   matched image-model runs. Keep the user-managed OrthoFoundation run separate.
   Existing outer0 is development evidence; confirmation requires an exposure
   audit and an appropriate reserved split. Any report-proxy gain remains a
   report-proxy gain without independent image reference labels.

No new adjudications or candidate labels were generated by preparation. No API
calls were made, and no Kaggle jobs were changed. The previous **$4.110458**
accounted total against the **$10** cap is unchanged. A larger paid extraction
needs a concrete request/cost plan; this packet does not silently expand spending.

## Reproduce and validate

```sh
# Refuses to overwrite the frozen bundle.
.venv/bin/python tools/prepare_report_label_evaluation.py

# Checks frozen inputs, source, outputs and blank-packet integrity.
.venv/bin/python tools/validate_report_evaluation.py verify

# Validate an exported AI annotation file; never modifies training labels.
.venv/bin/python tools/validate_report_evaluation.py annotations \
  --packet outputs/report_label_evaluation_v1/reviewer/pilot/packet.json \
  --annotations /path/to/pilot-annotations.json

.venv/bin/python -m pytest tests/test_report_evaluation.py -q
```

Eight targeted tests passed: reproducible sampling, control-draw independence,
duplicate-component exclusions, protected-role checks, rare-class shortfalls,
blind export, safe report rendering, and annotation provenance/completeness.
The generated JavaScript passed a syntax check with synthetic report text. No
browser was available for visual UI verification; the JSON packets and integrity
checks are the canonical artifacts.

Artifacts: [frozen manifest](../outputs/report_label_evaluation_v1/manifest.json),
[coverage counts](../outputs/report_label_evaluation_v1/coverage.json),
[protocol](../outputs/report_label_evaluation_v1/protocol.json).
