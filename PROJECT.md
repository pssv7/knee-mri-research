# RSNA Knee Abnormality Detection — Project State

Read this first when picking the work back up. It covers what exists, what the
numbers are, which decisions were made deliberately, and which traps have
already been paid for.

Last updated: 2026-08-23.

---

## The competition in one paragraph

Predict twelve binary findings per knee MRI study, scored by macro-averaged ROC
AUC. 4,407 training studies, but **only 58 carry radiologist labels** — the
other 4,349 have a free-text radiology report and nothing else. This is
therefore a weak-supervision problem wearing an imaging problem's clothes, and
the leaderboard is won by whoever converts reports into usable labels. Reports
exist in train but **not** in test, so text is a training-time signal only.

**Deadlines:** entry and team merger 2026-10-15, final submission 2026-10-22.

---

## Verified constraints

From the competition's own pages (see `docs/rules_notes.md`; regenerate with the
snippet there):

| Constraint | Value |
|---|---|
| Submission | Notebooks only |
| Runtime | ≤ 9 hours |
| Internet at submission | **Disabled** |
| Pretrained models | **Allowed** |
| External data | Allowed if freely and publicly available |
| Metric | Macro AUC over 12 findings |
| Output | `submission.csv` |

**There is a second prize track worth $18,000.** The Efficiency Prize scores
`AUC + RuntimeSeconds / 32400`, so inference cost is a *scored quantity*, not
just a ceiling. A submission using the full 9 hours carries a 1.0 penalty; a
30-minute one carries 0.056. This is why the design favours a frozen trunk and
why heavy TTA and large ensembles must justify their cost on both metrics.

**Report text may not be sent to a hosted API.** The rules forbid making
Competition Data available to non-participants, and reports are Competition
Data. The label extractor therefore runs locally on a Colab GPU.

---

## Where things stand

| Phase | Status |
|---|---|
| DICOM index scan | done — 24,371 series, 4,407 studies, 0% missing geometry |
| Image cache | done — 3 shards, ~29 GB, verified 24,371 series |
| Test cache | done — 15 series |
| Report label mining | done — fused macro AUC **0.854** against gold |
| Side classifier | done — **98.97%** study accuracy |
| Frozen DINOv2 embeddings | done — 0.9 GB, 768 dims/slice |
| Aggregation training | done — gold-58 macro **0.770** |
| Submission | done — **public LB 0.787** |

### The numbers that matter

**Label quality against the 58 gold studies:**

| extractor | macro AUC |
|---|---|
| Lexicon (rules) | 0.735 |
| LLM (Qwen2.5) | 0.838 |
| **Fused** | **0.854** |

All twelve findings clear the 0.70 usability threshold. Under the lexicon alone,
five would have been dropped from the loss entirely.

**Side classifier:** 97.28% per series, 98.97% per study after voting. Trained
on the 14,403 series whose DICOM tag gives a side, applied to the 9,968 that
lack one.

**First model, frozen DINOv2-S, five folds:**

| | macro AUC |
|---|---|
| Out-of-fold, against mined labels | 0.789 |
| **Gold 58** | **0.770** |
| **Public leaderboard** | **0.787** |

The gold estimate tracked the leaderboard within 0.017. That is the most useful
result so far: local measurement on the held-out 58 predicts the real score, so
development can proceed without spending submissions to learn anything.

**Where the model is strong and weak** — the split is clean and actionable:

| large / regional | gold | | small / focal | gold |
|---|---|---|---|---|
| Lateral OA | 0.899 | | ACL | 0.728 |
| Effusion | 0.897 | | Contusion | 0.711 |
| Medial OA | 0.895 | | Medial Meniscus | 0.698 |
| Baker's | 0.862 | | Fracture | 0.683 |
| PF OA | 0.772 | | Lateral Meniscus | 0.680 |
| | | | MCL | 0.671 |
| **mean 0.865** | | | **mean 0.695** | |

A 0.17 gap along exactly the axis the embedding choice predicted: CLS plus
mean-patch captures regional appearance and loses fine local detail. This is
the evidence that makes the spatial-patch variant worth its storage.

Two smaller observations. Synovitis scores 0.883 out-of-fold but 0.746 on gold
-- the largest divergence of any finding, and it also had the worst mined-label
quality, so the model learned extractor artefacts there. Conversely Medial OA,
Lateral OA, and Effusion all score *higher* on gold than out-of-fold, meaning
the model is denoising its own training labels. So 0.854 is not a ceiling.

---

## Pipeline

```
raw DICOM (570 GB, Kaggle only)
  └─ 01 scan        → index_train.parquet          (~820k headers)
      └─ 02 cache   → cache_*.npy + manifests      (~29 GB, 24×224×224 uint8)
          ├─ 04 side classifier → side_predictions.parquet
          └─ 05 embeddings      → embeddings_*.npy (~0.9 GB, 768/slice)
              └─ 06 train       → knee_fold*.pt

train.csv reports
  └─ 03 extract (Colab)  → llm_labels.parquet
      └─ fuse (local)    → train_labels.parquet    (soft targets + weights)
```

Notebooks are **generated** by `tools/make_notebook.py`, which embeds the real
module source so Kaggle and Colab run the same code as the repo. Edit `src/`
and regenerate; never edit a notebook directly.

---

## Repo layout

```
src/
  config.py              labels, cache geometry, path detection
  data/
    dicom_io.py          header reads, geometric slice sort, laterality
    cache.py             series → uint8 volume (shared with submission)
    index.py             threaded header scan → parquet
  labels/
    lexicon.py           multilingual rule extractor
    llm_extract.py       prompt + response parsing (no GPU deps)
    fuse.py              blend the two into soft targets
    calibrate.py         score an extractor against the gold 58
  model/
    embed.py             frozen DINOv2-S encoder
    dataset.py           study-level assembly, mirroring, collation
    net.py               attention pool → FiLM → series transformer → 12 logits
    splits.py            study-grouped folds, gold-leakage guard
    side_classifier.py   L/R prediction
tools/
  make_notebook.py       generates every notebook from src/
  check_notebook.py      executes the Colab notebook's cells locally
  check_kaggle_notebooks.py  runs notebooks 01/02 against a fake mini-dataset
docs/
  rules_notes.md         verified competition constraints
  dev_data.md            how to get a local dev series
tests/                   209 tests
```

---

## Decisions made deliberately

**Slice ordering by geometry, not `InstanceNumber`.** Projected
`ImagePositionPatient` onto the slice normal. On the very first real series
tested, `InstanceNumber` ran 30→1 — exactly backwards. Sorting by it reverses
the volume with no error.

**Mirroring is plane-aware.** For coronal and axial the medial-lateral axis is
in-plane, so mirroring flips columns. For **sagittal it is the slice axis**, so
mirroring reverses slice order. A blanket column flip on a sagittal series does
not swap medial with lateral at all — it reflects anterior-posterior, putting
the patella behind the knee.

**Volumes are cached as acquired, not mirrored.** Side detection was expected to
improve, and baking a mirror decision into 29 GB would have meant rebuilding it
each time. This decision is what saved the cache when the sagittal mirroring bug
was found later.

**Laterality reaches the frozen model as a covariate, not by mirroring.**
Embeddings are computed from the image and cannot be flipped afterwards, so
`is_right` and `side_known` go through FiLM instead. Two flags, not one signed
feature, because "left knee" and "side unknown" call for different behaviour.
Mirroring vs. covariate is an untested A/B for the end-to-end path.

**Fusion fits nothing on the gold 58.** An earlier version derived blend weights
and silence priors from them and scored 0.860 — but leave-one-out said 0.812,
*worse* than the LLM alone at 0.838. The apparent gain was the fit. The
replacement uses a fixed 0.30/0.70 lexicon/LLM blend chosen from a plateau
(0.20–0.35 all within 0.002), and silence scores zero with a low sample weight
rather than a fabricated prior.

**Silence is weighted down, not imputed.** `P(positive | unmentioned)` ranges
from 0.05 (MCL) to 0.58 (Effusion) — so silence is genuinely informative — but
using it as a *target* was worse than zero on every variant tested. The
uncertainty lives in the sample weight.

**Splits are by study, never by series.** A study's ~5.5 series image the same
knee in one session; a series split puts near-duplicates on both sides.

**The gold 58 never enter training.** `splits.assert_no_gold_leakage` is called
by the entrypoint rather than trusted to discipline.

**Frozen trunk first.** Not because DINOv2 is expected to win — prior RSNA
competitions were won by fine-tuned ImageNet CNNs — but because cached
embeddings make the aggregation stack iterable in minutes, and a frozen trunk is
competitive for the Efficiency Prize.

---

## Traps already paid for

Each of these cost real time. They are all guarded by tests now.

- **`.gitignore` `data/` also matched `src/data/`** — anchor with `/data/`.
- **nbformat `source` lines need trailing newlines**, or every cell collapses to
  one line. Validating by joining with `\n` hides it; join with `""`.
- **Writing `src/*.py` does not reload an already-imported module.** The
  bootstrap evicts `sys.modules` entries before importing. A cache build ran 15
  minutes against stale code before this was found.
- **Kaggle mounts competition data at `/kaggle/input/competitions/<slug>/`**,
  and attached notebook output at `/kaggle/input/notebooks/<user>/<notebook>/`.
  Search by content, recursively, skipping the competition tree (820k files).
- **`Quick Save` does not capture `/kaggle/working`.** Only `Save & Run All`
  produces a version with output files. A shard was lost to this.
- **`/kaggle/working` caps at 20 GB** and a shard is ~9.7 GB. Run each shard in
  its own notebook.
- **P100 (sm_60) is not supported** by Kaggle's PyTorch build. Use T4.
- **A process-pool worker defined in a notebook cell cannot be pickled.** It
  lives in `src/data/cache.py` as `build_series_record`.
- **`Sag` in these series descriptions means sagittal**, not Turkish `sağ`
  ("right"). Adding that token to the side patterns would mislabel a large
  fraction of the dataset.

---

## Kaggle working notes

- Each shard build: its own notebook, `SHARD` set, **Save & Run All**, then
  promote the output to a dataset. They are independent and can run in parallel.
- Notebooks 01 and 02 need the competition data; 04, 05, 06 do not — they work
  from the cache.
- Datasets in play: `knee-index`, `knee-cache-224-00/01/02`, `knee-embeddings`,
  `knee-labels`, and (soon) `knee-model`.

---

## What's next

1. **Run `06_train.ipynb`.** Needs `knee-embeddings`, the three cache datasets
   (for manifests), `knee-labels`, and the competition data (for `train.csv` and
   `train_series.csv`). Reports OOF AUC against mined labels and gold-58 AUC
   side by side — **trust the gold column**.
2. **Write the submission notebook.** Loads `dinov2_s_reg4.pt` and the fold
   weights, embeds test series offline, writes `submission.csv` with a
   try/except fallback to the training prior. Submit a trivial baseline first to
   prove the plumbing.
3. **Backbone bake-off** on identical folds: DINOv2-S (done), ConvNeXt-T,
   Swin-T, EfficientNetV2-S. Medical-domain weights (RadImageNet, BiomedCLIP)
   are a reasonable fifth arm.

**Reference point:** the fused labels score 0.854 against gold. A model well
below that is limited by itself, not by its labels, and the lever is the
backbone rather than more aggregation.

---

## Open questions

- Does mirroring beat passing side as a covariate? Untested.
- Do the small local findings (meniscus, fracture) need spatial patch features
  rather than CLS + mean-patch? Storing the full 16×16 grid costs ~115 GB
  against 0.9 GB, so it was deferred, not dismissed.
- The side classifier is validated on tag-bearing series and applied to series
  that lack tags. Those groups come from different institutions, so true
  accuracy on the untagged set is unmeasurable and may be below 98.97%.
- 10 studies (0.7%) have conflicting laterality tags across their series and are
  left unresolved.

---

## Conventions

- All project text is written in a neutral voice with no tooling attribution,
  including commit messages and any notebooks pushed publicly.
- Tests accompany every non-trivial module; the suite is the regression guard
  for the traps listed above.
- Verification runs against the real artefact, not a proxy. Several bugs here
  survived a check that examined the wrong thing.
