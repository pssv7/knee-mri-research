# Competition constraints — verified

Pulled from the competition pages on 2026-08-23 via the Kaggle SDK
(`ApiListCompetitionPagesRequest`), not from secondary sources. Full text is in
`competition_pages.md`.

Regenerate with:

```python
from kagglesdk import KaggleClient, KaggleEnv
from kagglesdk.competitions.types.competition_api_service import ApiListCompetitionPagesRequest
with KaggleClient(env=KaggleEnv.PROD) as client:
    req = ApiListCompetitionPagesRequest()
    req.competition_name = "rsna-knee-abnormality-detection"
    pages = client.competitions.competition_api_client.list_competition_pages(req).pages
```

## Code requirements

| Constraint | Value |
|---|---|
| Submission | Notebooks only |
| Runtime | **≤ 9 hours**, CPU or GPU alike |
| Internet | **Disabled** |
| External data | Allowed if **freely and publicly available**, equally accessible to all participants |
| **Pretrained models** | **Allowed explicitly** |
| Output filename | `submission.csv` |

Pretrained weights being permitted is what unblocks the Phase 4 backbone
bake-off. DINOv2-S, ConvNeXt-T, Swin-T, and EfficientNetV2-S all qualify as
freely and publicly available. Weights must still be pre-uploaded as a dataset,
since the submission notebook has no internet.

## Metric

Macro-averaged ROC AUC over the twelve findings:

    Final Score = (1/12) * sum(AUC_i for i in 0..11)

Submission columns, exactly:

```
StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture
```

## Efficiency Prize — a second track

Not previously known, and it changes the calculus. A separate prize pool of
$7,000 / $6,000 / $5,000 ranks submissions by:

    Efficiency = AUC / (Benchmark - maxAUC) + RuntimeSeconds / 32400

Minimised. `Benchmark` is the `sample_submission.csv` score, `maxAUC` the best
private-leaderboard score. Runtime is divided by 32,400 — the 9-hour limit in
seconds — so a submission using the full budget carries a penalty of 1.0 while a
30-minute one carries 0.056.

Eligibility: must be among the team's selected submissions and must beat the
`sample_submission.csv` benchmark on the private leaderboard. A single
submission can win both tracks.

**Implication for design.** Runtime is no longer only a constraint to stay under
— it is scored. A frozen DINOv2-S trunk with a light aggregation head is
well-placed here: the backbone runs once per slice with no gradient, and the
head is small. Heavy test-time augmentation and large ensembles buy AUC at a
direct efficiency cost, so they should be measured against both metrics rather
than assumed good.

## Data security — the report-text question, settled

From the rules:

> You agree not to transmit, duplicate, publish, redistribute or otherwise
> provide or make available the Competition Data to any party not participating
> in the Competition.

Report text is Competition Data. Sending it to a hosted LLM API would provide it
to a non-participant, so the local open-weights approach taken in
`notebooks/03_extract_labels.ipynb` was the correct one — the model ran on the
Colab GPU and no report text left the session.

The competition data is additionally subject to <http://rsna.org/mira-license>.

## Sharing

- No private sharing of code outside teams.
- Public sharing is permitted, but only on Kaggle's forums or notebooks for this
  competition.

## Timeline

| Date | Event |
|---|---|
| 2026-07-30 | Start |
| **2026-10-15** | Entry deadline, team merger deadline |
| **2026-10-22** | Final submission |

## Data description — confirmations

- Per-study probability for each of twelve findings.
- "Only a small subset of training studies carry per-condition labels" — the 58.
- Reports are provided explicitly so participants "may wish to derive the labels
  for the remaining studies", which is the intended path and what the fusion
  pipeline does.
- `Fluid_Sensitive` and `Fat_Suppression` are "often correlated ... but not
  necessarily equivalent for every case", so both should be passed to the model
  rather than collapsed into one flag.
- Studies span "a diverse international mix of imaging sites", consistent with
  the missing-laterality pattern being institutional.
