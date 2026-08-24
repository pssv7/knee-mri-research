# RSNA Knee Abnormality Detection — Project State

Read this first when picking the work back up cold. It covers what exists, what
the numbers are, what has been tried and failed, and which traps have already
been paid for.

Last updated: 2026-08-24 (fold-0 LB 0.827 diagnosed as target-shape dilution; sharpened labels ready).

---

## The competition in one paragraph

Predict twelve binary findings per knee MRI study, scored by macro-averaged ROC
AUC. 4,407 training studies, but **only 58 carry radiologist labels** — the
other 4,349 have a free-text radiology report and nothing else. This is a
weak-supervision problem wearing an imaging problem's clothes, and the
leaderboard is won by whoever converts reports into usable labels. Reports exist
in train but **not** in test, so text is a training-time signal only.

**Deadlines:** entry and team merger 2026-10-15, final submission 2026-10-22.

---

## Where things stand

| | public LB |
|---|---|
| Frozen DINOv2-S + aggregation | 0.787 |
| Frozen + fine-tuned, 50/50 blend | 0.839 |
| **Fine-tuned ConvNeXt-T alone** | **0.845 ← best** |
| Leaderboard top | 0.952 |
| Leaderboard 20th | 0.941 |

**We are ~0.10 behind a very tight cluster.** Twenty teams sit within 0.011 of
each other at 0.94+, which means a known recipe exists and the gap is structural
rather than a matter of tuning.

### The leading hypothesis: labels, not models

Fused training labels score **0.854** against the gold 58. The model trained on
them scores **0.845**. The model is at its label ceiling, not its own. And 0.854
is low for what is essentially reading comprehension — the reports describe
these findings explicitly.

**The mechanism was diagnosed and is concrete.** The categorical extractor ties
**369 of 696 label slots** on a single value:

| finding | studies tied on one value (of 58) |
|---|---|
| Baker's | 41 |
| Fracture | 41 |
| MCL | 39 |
| Synovitis | 37 |

AUC is a ranking metric. It cannot order studies inside a tie block, so more
than half the training signal carries no ordering information at all.

---

## Verified constraints

From the competition's own pages. `docs/rules_notes.md` has the detail and a
snippet to regenerate it through the Kaggle SDK.

| Constraint | Value |
|---|---|
| Submission | Notebooks only |
| Runtime | ≤ 9 hours |
| Internet at submission | **Disabled** |
| Pretrained models | **Allowed** |
| External data | Allowed if freely and publicly available |
| Metric | Macro AUC over 12 findings |
| Output | `submission.csv` |

**A second prize track worth $18,000.** The Efficiency Prize scores
`AUC + RuntimeSeconds / 32400`, so inference cost is a *scored quantity*, not
just a ceiling. The full 9 hours carries a 1.0 penalty; 30 minutes carries
0.056.

**Report text may not be sent to a hosted API.** The rules forbid making
Competition Data available to non-participants, and reports are Competition
Data. The extractor runs locally on a Colab or Kaggle GPU.

---

## Data and assets

| asset | what | where |
|---|---|---|
| Image cache | 24,371 series, 24×224×224 uint8, ~29 GB, 3 shards | Kaggle notebook outputs |
| Test cache | 15 public test series | Kaggle |
| Frozen embeddings | DINOv2-S, 768 dims/slice, 0.9 GB | `knee_embeddings` |
| Grid-4 embeddings | 6,528 dims/slice, 7.6 GB — **worse, see below** | `knee_embeddings_v2` |
| Side predictions | L/R for all 24,371 series | `knee_side_classifier` |
| Fused labels | 4,349 studies × 12 soft targets + weights | `knee-labels` |
| Frozen fold weights | 5 folds | `knee-train` |
| ConvNeXt weights | 1 fold, 12 epochs, all shards | notebook 08 output |

Kaggle limits: **25 GPU-hours per week, one session at a time.**

---

## Pipeline

```
raw DICOM (570 GB, Kaggle only)
  └─ 01 scan        → index_train.parquet          (~820k headers)
      └─ 02 cache   → cache_*.npy + manifests      (~29 GB, 24x224x224 uint8)
          ├─ 04 side classifier → side_predictions.parquet   (98.97% study acc)
          ├─ 05 embeddings      → embeddings_*.npy (frozen path)
          │   └─ 06 train       → knee_fold*.pt
          │       └─ 07 submit  → submission.csv
          └─ 08 finetune        → convnext_fold0.pt (end-to-end path)
              └─ 10 submit_ensemble → submission.csv  (BLEND=1.0 for tuned only)

train.csv reports
  └─ 03 extract (Colab/Kaggle GPU) → llm_probs.parquet
      └─ fuse (local)              → train_labels.parquet

09 export_for_colab   thins a shard to 8 slices (3.2 GB) so it can transfer
```

Notebooks are **generated** by `tools/make_notebook.py`, which embeds the real
module source so Kaggle and Colab run the same code as the repo. **Edit `src/`
and regenerate; never edit a notebook directly.** A test fails if a notebook is
stale.

---

## Repo layout

```
src/
  config.py              labels, cache geometry, path detection
  data/
    dicom_io.py          header reads, geometric slice sort, laterality
    cache.py             series -> uint8 volume (shared with submission)
    index.py             threaded header scan -> parquet
  labels/
    lexicon.py           multilingual rule extractor
    llm_extract.py       prompts + parsing, categorical AND probability modes
    mode_fusion.py       combine the two LLM modes; four global strategies
    eval_fusion.py       score strategies on gold (LOO, jackknife) locally
    fuse.py              blend extractors into soft targets
    calibrate.py         score an extractor against the gold 58
  model/
    embed.py             frozen DINOv2-S encoder, POOL_GRID variants
    dataset.py           study assembly, mirroring, collation
    net.py               attention pool -> FiLM -> series transformer -> 12 logits
    splits.py            study-grouped folds, gold-leakage guard
    side_classifier.py   L/R prediction
tools/
  make_notebook.py           generates every notebook from src/
  check_notebook.py          executes the Colab notebook's cells locally
  check_kaggle_notebooks.py  runs 01/02 against a fake mini-dataset
  check_submission.py        runs 07 or 10 end to end on a fake test set
docs/
  plan.md                the forward plan, with the gate at every step
  agent_playbook.md      how decisions get made here -- read before acting
  worklog.md             dated session records, with numbers
  rules_notes.md         verified competition constraints
  dev_data.md            how to get a local dev series
tests/                   259 tests
```

---

## What has been tried

### Worked

**Fine-tuned ConvNeXt-T end to end** — gold 0.825, LB 0.845. All three shards,
12 epochs, 3 series x 8 slices per study. Best model so far. Only **one fold**
has been trained; five is the standard next gain.

**Probability-valued label extraction** — breaks the tie problem where it
applies. Nine of twelve findings improved sharply: Contusion 0.662→0.874,
Lateral OA 0.680→0.870, Baker's 0.818→0.957, Synovitis 0.619→0.777, ACL
0.855→0.983.

**Side classifier** — 98.97% study accuracy. The DICOM side tag is blank on 41%
of series and descriptions are anonymised, so this is how laterality is
recovered at all.

### Failed, with the reason

**Spatial patch embeddings (`POOL_GRID = 4`).** Hypothesis: the focal/regional
split (0.695 vs 0.865) is what a mean over 256 patch tokens predicts. Result:
gold 0.770 → **0.747**, eleven of twelve findings worse. 8.5x more input
dimensions against the same 4,349 studies made optimisation harder without
adding usable signal. *Confound recorded:* grid 1 uses an Identity projection
while grid 4 adds 3.3M new parameters, so features and projection changed
together. Control not run — both answers lead to the same next step.

**Frozen + tuned ensemble.** Per-finding gold comparisons suggested strong
complementarity. The leaderboard disagreed: frozen 0.787, blend 0.839, tuned
alone **0.845**. The blend lands between the two single models, which is what
averaging a weak ranker with a strong one does. A caution about how little 58
studies can settle.

**Sharpened meniscus/MCL definitions.** Reasoned from radiology — a tear
requires signal reaching an articular surface — rather than tested. Macro 0.830
→ 0.810, MCL 0.857 → 0.705 with 51 of 58 tied. Reverted.

**Per-finding fusion weights fitted on gold.** Scored 0.860 in-sample, 0.812
under leave-one-out, worse than the better single source. Replaced with a fixed
0.30/0.70 blend chosen from a plateau. **This lesson recurs: 58 studies cannot
support per-finding fitting.**

---

## The open lead: combine both extraction modes

**Status: measured and frozen (2026-08-23).** The 32B gold checkpoint ran and
`python -m src.labels.eval_fusion` settled it:

| scores on gold | macro |
|---|---|
| categorical LLM alone (14B) | 0.838 |
| probability LLM alone (**32B** — was 0.810 on 14B) | 0.860 |
| per-finding oracle of the two | 0.885 |
| **score_average of the two** | **0.889** |
| with 0.10 lexicon blend (the training targets) | **0.891** |

Averaging the two modes beats *choosing* between them per finding — the modes
err in different places, so the mean outranks the oracle. `score_average` and
`rank_average` tied (jackknife 0.54/0.47); the simpler, probability-scaled one
was frozen as `mode_fusion.DEFAULT_STRATEGY`. `LEXICON_WEIGHT` moved
0.30 → 0.10: with the stronger LLM the old weight cost ~0.010, and 0.00–0.15
is a plateau. MCL holds 0.955 and Medial Meniscus 0.951 (the two findings
probability mode used to collapse); nothing sits below Synovitis's 0.795.

Labels went 0.854 → 0.891 against gold. The next artefact is the ~1 h 32B
corpus run producing `llm_probs.parquet`; `fuse.build()` then emits the new
`train_labels.parquet` automatically. The table below records the 14B
per-finding numbers that motivated all this.

The categorical and probability extractors are complementary per finding:

| finding | categorical | probability |
|---|---|---|
| Medial Meniscus | **0.944** | 0.576 |
| MCL | **0.959** | 0.705 |
| PF OA | **0.826** | 0.777 |
| Fracture | 0.752 | **0.885** |
| Contusion | 0.760 | **0.807** |
| Lateral OA | 0.828 | **0.885** |
| Effusion | 0.757 | **0.809** |
| Synovitis | 0.706 | **0.759** |

Categorical wins where reports state a finding crisply. Probability wins where
reports are usually silent and the value is in ranking the studies nobody
mentioned it for. **Per-finding best-of-both scores ~0.869**, against 0.854 for
the labels currently in use.

Medial Meniscus and MCL stay collapsed under probability extraction — 12 and 8
distinct values, 45 and 51 tied — so those two should probably come from the
categorical extractor.

**The corpus run is done (2026-08-23, 32B).** `llm_probs.parquet` lives in
`data/meta/` and `python -m src.labels.fuse` rebuilt `train_labels.parquet`:
macro **0.891** on gold, weakest finding Synovitis 0.792, all twelve above the
0.70 kill threshold. Targets are continuous now — 300–975 distinct values per
finding against the tie-collapsed categorical labels — and target means track
gold prevalence where reports mention a finding.

**Immediate next steps (user, Kaggle):**
1. Upload the new `data/meta/train_labels.parquet` as a new version of the
   `knee-labels` dataset.
2. Update the Kaggle copies of notebooks 08 and 10 from the repo — both
   changed (validation-fold selection + `FOLD` constant; `BLEND = 1.0`).
3. Run notebook 08 with `FOLD = 0` first (~2 GPU-h). Gate: its monitor-only
   gold macro should land ≥ ~0.82 at the selected epoch (the old labels
   produced 0.825 *with* the selection leak; the new labels are 0.037 better,
   so materially less than that means something is wrong — stop and compare
   epoch curves before spending the other four sessions).
4. Folds 1–4 as four more sessions, editing `FOLD` each time. Save & Run All,
   never Quick Save. Upload the five `.pt` files as the `knee-convnext`
   dataset, then submit notebook 10.

**Beware the recurring trap:** per-finding selection on 58 studies is the same
overfit that sank the fusion weights. A robust blend capturing most of the
oracle beats twelve fitted choices.

---

## Decisions made deliberately

**Slice ordering by geometry, not `InstanceNumber`.** Projected
`ImagePositionPatient` onto the slice normal. On the first real series tested,
`InstanceNumber` ran 30→1 — exactly backwards. Sorting by it reverses the volume
with no error raised.

**Mirroring is plane-aware.** For coronal and axial the medial-lateral axis is
in-plane, so mirroring flips columns. For **sagittal it is the slice axis**, so
mirroring reverses slice order. A blanket column flip on a sagittal series does
not swap medial with lateral at all — it reflects anterior-posterior, putting
the patella behind the knee.

**Volumes are cached as acquired, not mirrored.** This is what saved the 29 GB
cache when the sagittal mirroring bug was found later.

**Laterality reaches the frozen model as a covariate** (`is_right`,
`side_known`), because precomputed embeddings cannot be flipped. The end-to-end
path mirrors geometrically instead. Which is better is untested.

**Silence is weighted down, not imputed.** `P(positive | unmentioned)` ranges
from 0.05 (MCL) to 0.58 (Effusion), so silence is informative — but using it as
a *target* was worse than zero on every variant tested.

**Splits are by study, never by series.** A study's ~5.5 series image the same
knee in one session.

**The gold 58 never enter training.** `splits.assert_no_gold_leakage` is called
by the entrypoint. The one known leak — notebook 08's best-epoch checkpointing
selected on gold macro, likely inflating that model's gold figure by 0.01–0.03
— **is fixed:** selection now happens on the validation fold's mined labels
binarised at 0.5, gold is printed as monitoring only, and `FOLD` is a constant
edited per session so five sessions produce `convnext_fold{0..4}.pt`. The
historical 0.825 gold figure for the existing fold-0 checkpoint still carries
that bias. Notebook 10 is now `BLEND = 1.0` (tuned-only, the leaderboard's
verdict) and averages across every fold checkpoint it finds.

---

## Traps already paid for

All guarded by tests now.

- **`.gitignore` `data/` also matched `src/data/`** — anchor with `/data/`.
- **nbformat `source` lines need trailing newlines**, or every cell collapses to
  one line. Validating by joining with `\n` hides it; join with `""`.
- **Writing `src/*.py` does not reload an already-imported module.** The
  bootstrap evicts `sys.modules` entries first. A cache build ran 15 minutes
  against stale code before this was found.
- **A docstring inside `nb.code("""...""")` terminates the string** and breaks
  the generator. Use comments in cell bodies. This happened twice.
- **Kaggle mounts competition data at `/kaggle/input/competitions/<slug>/`** and
  attached notebook output at `/kaggle/input/notebooks/<user>/<notebook>/`.
  Search by content, recursively, skipping the competition tree (820k files).
- **Kaggle slugs replace non-alphanumerics with hyphens** — a notebook titled
  `knee-cache-224-01_shard-1` has slug `knee-cache-224-01-shard-1`.
- **`Quick Save` does not capture `/kaggle/working`.** Only `Save & Run All`
  produces a version with output files. A shard was lost to this.
- **`/kaggle/working` caps at 20 GB** and a shard is ~9.7 GB. One shard per
  notebook.
- **P100 (sm_60) is unsupported** by Kaggle's PyTorch build. Use T4.
- **A process-pool worker defined in a notebook cell cannot be pickled.** It
  lives in `src/data/cache.py` as `build_series_record`.
- **`Sag` in these series descriptions means sagittal**, not Turkish `sağ`
  ("right"). Adding that token to the side patterns would mislabel a large
  fraction of the dataset.
- **vLLM breaks when Colab's torch shifts** — first a torchaudio CUDA mismatch,
  then a missing `libcudart.so.13`. Notebook 03 falls back to `transformers`.
- **Downloading ~10 GB from Kaggle to Colab fails** at ~5.5 GB with
  `IncompleteRead`, consistently. Notebook 09 exports a 3.2 GB thinned shard
  instead.

---

## Method notes for whoever continues

**Verify against the real artefact, not a proxy.** Several bugs here survived a
check that examined the wrong thing: a notebook validated by joining source
lines the same wrong way the generator wrote them; an import "confirmed" by
finding the module's own filename in the text; a test that skipped any cell
starting with a shell magic when the magic sat on line 3. When a check passes,
ask what exactly it looked at.

**The gold 58 are the only honest signal, and they are small.** Gold predicted
the leaderboard within 0.017 for the frozen model, which is why it is trusted —
but it also endorsed an ensemble the leaderboard rejected. Treat per-finding
differences on 58 studies as weak evidence.

**Cheap loops before expensive ones.** The gold checkpoint runs 58 reports in
seconds; the corpus takes an hour; a training run longer still. Two prompt
changes reasoned from domain knowledge both measured worse than what they
replaced.

**Conventions:** all project text is written in a neutral voice with no tooling
attribution, including commit messages and any notebooks pushed publicly. Tests
accompany every non-trivial module.
