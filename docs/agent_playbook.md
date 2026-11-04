# Research playbook

This project is archived after the competition. These rules preserve the useful
reasoning without carrying forward obsolete queues or leaderboard instructions.

## Start with the information bottleneck

The task had 4,349 report-labelled studies and only 58 expert-labelled studies.
Model capacity was rarely the first constraint. Before changing an architecture,
check whether the target, selected series, anatomical region, spatial resolution,
or validation design hides the evidence the model needs.

Use an oracle test when possible. Perfect or manually supplied localization on a
small sample can test whether localization is limiting. A clean-label subset can
test whether supervision is limiting. If the oracle does not help, reject the
direction before paying for a full training run.

## Separate evidence from selection

- Group splits by study or patient identity.
- Keep a final reference set outside routine model and blend decisions.
- Report per-label results and sample counts with the macro score.
- Compare candidates on identical rows and preserve their predictions.
- Treat small validation sets as noisy measurements, not exact rankings.
- Record whether a result is independent, exploratory, or selection-overlapping.

The Gold58 set was inspected repeatedly during the competition. Historical
Gold58 results are useful diagnostics, but they are not untouched estimates.

## Run the cheapest decisive experiment

Order work by information gained per unit cost:

1. Static data and provenance audit.
2. Small deterministic diagnostic.
3. CPU or single-batch smoke test.
4. One controlled training comparison.
5. Cross-validation or full-data training.
6. Ensemble or deployment work.

Change one causal factor at a time. Define the success threshold and failure
interpretation before execution. Save enough metadata to reproduce the result.

## Verify the real artifact

Checks must inspect the object that will actually be used. Verify checkpoint
hashes, label order, preprocessing identity, cache geometry, study membership,
and submission columns. Tests derived from the same incorrect assumption as the
producer can pass while the artifact remains wrong.

Important imaging details retained by this project include geometric slice
ordering, plane-aware laterality normalization, physical field of view, series
selection, and consistent preprocessing between training and inference.

## Keep experiments reproducible

- Put reusable behavior in `src/`; keep command orchestration in `tools/`.
- Generate or validate notebooks from source where a generator exists.
- Store large data, caches, outputs, and weights only in ignored locations.
- Bind checkpoints to sidecar metadata and content hashes.
- Use explicit ensemble manifests instead of directory globs.
- Add focused tests for nontrivial reusable behavior.
- Write conclusions with numbers, scope, and limitations.

## Reframing checklist

When progress stalls, answer these before adding another model:

1. What evidence would a domain expert use?
2. Does the input preserve that evidence at sufficient resolution and context?
3. Does each target mean the same thing in reports, external datasets, and the
   competition reference standard?
4. Can the model localize the relevant anatomy?
5. Does validation measure image truth or agreement with the label generator?
6. Which single oracle experiment could disprove the current formulation?

Prefer a simpler classifier with better targets, localization, and validation
over a larger classifier trained on weak or misaligned information.
