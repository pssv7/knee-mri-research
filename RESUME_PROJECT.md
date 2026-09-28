# Knee MRI Abnormality Detection: Resume Project Profile

A reference for tailoring resumes, cover letters and interview answers. Every
number here comes from the repo's own records (`PROJECT.md`,
`docs/architecture.md`, `docs/worklog.md`, `docs/protocol_incumbent.md`,
`docs/submission_protocol.md`). Check the figures marked *as of 2026-09-16*
again before the competition closes on 2026-10-22.

---

## 1. One-line summaries (pick by role)

- **ML / Deep Learning:** Built an end-to-end multi-label MRI classifier
  (ConvNeXt with 2.5D slices, attention pooling and a transformer over series)
  trained on labels pulled from free-text reports in ~11 languages by a local LLM.
  Reached **0.914 macro ROC AUC** on the public Kaggle leaderboard.
- **Medical imaging / Computer vision:** Processed **570 GB of raw DICOM
  (24,371 series, ~820k headers)** into a verified 29 GB volume cache. Built
  a laterality classifier with 98.97% study-level accuracy and plane-aware
  medial/lateral mirroring.
- **NLP / LLM:** Pulled weak-supervision labels from 4,349 multilingual
  radiology reports using locally hosted, quantised **Qwen2.5-Instruct
  (7B/14B/32B AWQ) on vLLM**. Combined a categorical pass and a probability
  pass with a rule-based lexicon, which raised label AUC against radiologist
  gold from **0.854 to 0.891**.
- **MLOps / ML engineering:** Generated Kaggle and Colab notebooks from a tested
  Python source tree so training and inference stay identical. Ran multi-GPU
  DDP training runs on a quota, and backed the work with **800+ tests** aimed
  at silent failures.
- **Applied statistics / Experimentation:** Ran pre-registered experiments
  with paired-bootstrap confidence intervals, fixed gates and matched controls.
  Showed that a 58-study gold set cannot tell apart differences under ~0.02 AUC,
  and rebuilt model selection around that limit.

---

## 2. Project overview

| | |
|---|---|
| **Competition** | RSNA Knee Abnormalities Detection (Kaggle, Radiological Society of North America), $77,000 prize pool: main leaderboard plus an Efficiency Prize track |
| **Task** | Predict 12 binary findings per knee MRI study: ACL tear, MCL tear, medial and lateral meniscus tear, medial, lateral and patellofemoral osteoarthritis, joint effusion, synovitis, Baker's cyst, bone contusion, fracture |
| **Metric** | Macro-averaged ROC AUC across the 12 findings |
| **Data** | 4,407 training studies, 24,371 MRI series, 570 GB raw DICOM, ~1,300 hidden test studies |
| **Core challenge** | Only **58 studies have radiologist labels**. The other 4,349 have only a free-text radiology report in ~11 languages. Reports exist for training data only, never for test |
| **Constraints** | Notebook-only submission, **9-hour runtime cap, no internet**, report text may not go to hosted APIs (so every LLM ran on my own GPUs) |
| **Role** | Solo: data engineering, label pipeline, modelling, training infrastructure, evaluation and submission |
| **Timeline** | August 2026 to the final submission deadline of 22 October 2026 |
| **Result (as of 2026-09-16)** | **0.914 public leaderboard macro AUC** (a 15-model ensemble). Progress: 0.787 → 0.845 → 0.869 → 0.889 → 0.891 → 0.906 → 0.911 → 0.914 |

**Framing for interviews:** this is a weak-supervision problem dressed up as an
imaging problem. Text can't be a model input because test studies have no
reports, so the reports could only be used to *create training labels*.

---

## 3. Technology stack

### Languages and core libraries
- **Python 3.12**
- **NumPy, pandas, PyArrow / Parquet**: header index, label matrices,
  out-of-fold (OOF) prediction tables
- **OpenCV** (`opencv-python-headless`): in-plane resizing
- **Matplotlib**: error analysis and diagnostics

### Medical imaging
- **pydicom 3**: header parsing, pixel decoding
- **pylibjpeg, pylibjpeg-libjpeg, pylibjpeg-openjpeg**: compressed DICOM
  transfer syntaxes (JPEG, JPEG 2000)
- DICOM geometry: `ImagePositionPatient`, `ImageOrientationPatient`, slice
  normal projection, anatomical plane detection, laterality tags

### Deep learning
- **PyTorch 2.4+**
- **PyTorch Lightning 2.x**: DDP (`ddp_notebook`), mixed precision, gradient
  accumulation, callbacks, checkpointing, CSV logging
- **timm 1.0**: pretrained backbones
- Mixed-precision training, OneCycle LR schedule, AdamW
- EMA, SWA and model soups (uniform and greedy, after Wortsman et al., 2022)

### LLMs and NLP
- **vLLM** for high-throughput local inference, with a fallback to
  **Hugging Face Transformers**
- **Qwen2.5-7B / 14B / 32B-Instruct, AWQ-quantised** for structured
  information extraction
- **intfloat/multilingual-e5-small** for sentence embeddings of reports
  (auxiliary-loss experiments)
- Prompt engineering: structured JSON output, categorical and
  probability-valued extraction, evidence spans, self-consistency sampling,
  prompt ensembling
- A multilingual rule-based lexicon extractor (negation, severity, certainty,
  anatomy stems)

### Compute and platforms
- **Kaggle Notebooks**: 2× T4 DDP training, offline GPU inference, Datasets
  for model and cache artifacts, Kaggle API/SDK for launching and auditing jobs
- **Google Colab Pro**: A100 40 GB and L4 24 GB for LLM extraction and
  single-GPU training, with Google Drive for artifact storage
- **KaggleHub**: downloads large artifacts with resume and checksum checks
- Git, pytest, a virtualenv workflow

### Testing and engineering practice
- **pytest**: 803 passing tests over 43 test modules (as of 2026-09-15)
- Notebook generation from source (`tools/make_notebook.py`), with a test
  that fails when a notebook drifts from the source
- Fake mini-datasets for running Kaggle notebooks and the submission end to
  end on a local machine
- SHA-256 receipts for checkpoints, source bundles and inputs on every launch

---

## 4. Models used and evaluated

| Model | How it was used | Outcome |
|---|---|---|
| **ConvNeXt-Tiny** (`convnext_tiny.fb_in22k_ft_in1k`, 27.8M params) | Main 2.5D trunk, fine-tuned end to end | **Shipped**: backbone of every leaderboard model from 0.845 to 0.914 |
| **DINOv2-Small** (ViT-S/14 with registers) | First baseline: frozen embeddings + aggregation head, then a fine-tuning probe | Frozen scored 0.787 LB. Fine-tuned scored 0.820 on gold vs ConvNeXt's 0.861, so this route was dropped |
| **ConvNeXt-V2-Tiny** (`convnextv2_tiny.fcmae_ft_in22k_in1k`) | Matched A/B test against ConvNeXt on the same fold | Tested as a controlled architecture swap |
| **Slice-sequence transformer** (2 layers, 256 wide) | Transformer over slice features before pooling | Weaker on its own (0.870 vs 0.878 on matched folds). Kept as ensemble diversity |
| **Spatial volume encoder** (residual 3D convs, zero-initialised) + **per-plane adapters** | Adds depth context before spatial pooling; separate pooling for sagittal, coronal and axial | Built and tested (+1.97M params, output identical to the control at init). Fold-1 screen staged |
| **Custom CNN side classifier** (trained from scratch) | Predicts left or right knee from the image; slices are input channels | **98.97% study-level accuracy** |
| **Qwen2.5-14B-Instruct-AWQ** | Categorical report extraction | Shipped categorical extractor |
| **Qwen2.5-32B-Instruct-AWQ** | Probability extraction; categorical re-extraction test | Shipped probability extractor. Did worse than 14B at categorical extraction (0.824 vs 0.838) |
| **Qwen2.5-7B-Instruct-AWQ** | Early and cheap extraction | Development baseline |
| **multilingual-e5-small** | Report embeddings as an auxiliary regression target | Improved validation but lost on LB (0.880 vs 0.889), so it was dropped |

---

## 5. System architecture

```
 IMAGE BRANCH                                LABEL BRANCH
 raw DICOM, 570 GB, 24,371 series             4,349 reports, ~11 languages
   │  threaded header scan (~820k headers)      │
   ▼                                            ▼
 geometric slice sort → 24 slices →          LLM categorical pass (presence,
 224×224 → 1–99th percentile uint8           severity, certainty, evidence span)
   │  29 GB cache, 3 hash-sharded shards     LLM probability pass (0–100)
   ▼                                          rule-based multilingual lexicon
 side classifier (L/R per series,               │
 confidence-weighted study vote)             mode fusion → 0.9 LLM / 0.1 lexicon
   │                                          → quantile sharpening
   │                                          → 4,349 × 12 soft targets + weights
   └──────────────────┬──────────────────────────┘
                      ▼
      5-fold study-grouped training (ConvNeXt, DDP)
                      ▼
      multi-family fold ensemble → offline inference (≤9 h) → submission.csv
```

### 5.1 Data engineering: DICOM to cache
- **Ordered slices by geometry**: projected `ImagePositionPatient` onto the
  slice normal. `InstanceNumber` ran backwards (30→1) on the first real series
  I checked, and sorting by it would have silently reversed volumes.
- **Two-step slice sampling** (N → 24 → depth), shared by training and
  inference so both decode exactly the same files.
- **Per-volume percentile intensity stretch**, because MRI has no standard
  intensity units across scanners and sites.
- **Sharded by `md5(StudyInstanceUID) % 3`**, so every series of a study lands
  in the same shard with a representative mix of sites.
- **Verified the whole artifact, not just the code**: rebuilt 500 of 500
  sampled series byte-for-byte from raw DICOM, found 0 blank volumes out of
  24,371, and got 0 build errors.
- Geometry audit: all 24,371 series have full geometry. Slice spacing is even
  within a series (CV 5.1e-4) but varies 1.74× between series, and I wrote
  that limit down for the 3D models.

### 5.2 Laterality recovery
- 40.9% of series (9,968) have **no left/right information in DICOM metadata**
  (tags empty, descriptions anonymised), and a wrong side swaps 4 of the 12
  labels.
- Trained a CNN from scratch on the 14,210 series with a known side, feeding
  slices in as channels so it can pick up the direction of travel on sagittal
  series. Combined per-series predictions with a confidence-weighted study
  vote: **98.97% study accuracy**.
- **Plane-aware mirroring**: coronal and axial series flip columns; sagittal
  series reverse slice order. A plain column flip on a sagittal series swaps
  front and back, not medial and lateral.
- Validated on scanner vendors with no tags (Toshiba/Canon) against a
  zero-parameter geometric rule (McNemar p = 1.0).

### 5.3 Weak supervision: labels from radiology reports
- Reports in ~11 languages (English 39%, plus Greek, Croatian, Serbian,
  Spanish, Turkish, German, Bulgarian, Russian, Dutch, French, Italian,
  Portuguese), read **in the original language, with no translation step**.
- **Dual-mode LLM extraction.** The categorical pass tied 369 of 696 gold
  label slots on a single value, and AUC can't rank inside a tie. The
  probability pass breaks those ties. Label AUC against gold: categorical
  0.838, probability 0.860, per-finding oracle 0.885, **plain average 0.889**,
  and 0.891 with the lexicon blended in.
- **Silence handled with confidence weights, not imputation**: four agreement
  tiers (1.00 / 0.70 / 0.40 / 0.25). Silent weights scale with how often each
  finding goes unmentioned across the corpus (for example, synovitis's silent
  weight falls to 0.044).
- **Nothing fitted on the 58 gold studies.** A 24-weight per-finding fusion
  scored 0.860 in-sample but 0.812 under leave-one-out, so I replaced it with
  global constants chosen from plateaus.
- Label error analysis against gold: labels miss 45.8% of true positives and
  false-alarm on 7.2% of negatives. Only 73% of gold positives are mentioned
  in the report at all, which puts a ceiling on any label taken from text.
- Found a **language bias in the lexicon**: it is 3.2× less informative in
  Bulgarian than in English at matched report length, because multi-word
  anatomy stems break on Slavic article suffixes.

### 5.4 Model architecture
- **2.5D input**: the previous, current and next slices become 3 channels, so
  ImageNet-pretrained weights apply unchanged while still seeing depth context.
- **Hierarchical multiple-instance learning** over a variable-size set of
  series:
  ConvNeXt-T (768-d per slice) → **attention pooling over slices** →
  **FiLM conditioning** on 7-d series metadata (fluid-sensitive,
  fat-suppressed, plane one-hot, predicted side and a side-known flag) →
  **2-layer, 8-head transformer across series** with padding masks →
  **attention pooling over series** → LayerNorm, dropout, linear → 12 sigmoid
  logits.
- Replaced mean pooling with attention pooling because focal findings (a
  meniscus tear on 2–3 slices) got averaged away: 0.695 for focal findings vs
  0.865 for regional ones in the first model.
- Padding masks enforced in three places (pooling softmax, transformer key
  padding, normalisation), with tests proving a padded series can't change a
  prediction.
- Variants: class-attention (per-finding) pooling, a slice-sequence
  transformer, a residual 3D spatial volume encoder, and per-plane adapter
  pooling with missing planes masked out.
- **Protocol-aware series selection**: when a study has more series than the
  cap, the selection keeps plane and protocol coverage. That cut the number of
  studies losing an available plane from 228 (5.17%) to 0.

### 5.5 Training
- 5-fold cross-validation, **split by study** (never by series) and stratified
  by positive-finding count, with an assertion that no gold study leaks into
  training.
- AdamW (1e-4), OneCycle to a 3e-4 peak, 12 epochs, mixed precision,
  effective batch of 8 studies (1 study × 2 GPUs × 4 accumulation steps),
  soft-target weighted BCE with optional `pos_weight`. About 1–2.5 GPU-hours
  per fold.
- Augmentation includes laterality-consistent flips: flipping the image also
  swaps the medial and lateral labels.
- EMA and SWA weights kept alongside the best checkpoint in every run.
- DDP details handled explicitly: fork-safe CUDA detection, rank-consistent
  metrics, reading checkpoints from disk after a forked `fit`.

### 5.6 Ensembling and inference
- **Leaderboard ensemble (0.914):** 15 models, equal weights. Three families
  (16-slice, 24-slice and protocol-selection ConvNeXt) × 5 folds, packaged
  with hash-bound manifests.
- Selected the ensemble on **paired within-fold bootstrap over 4,349 OOF
  studies**: macro 0.8767 → 0.8852, Δ +0.0086, 95% CI [+0.0068, +0.0101].
  All 5 folds and all 12 findings improved.
- **Offline GPU benchmark before submitting**: 100 raw-DICOM studies with 0
  errors, 5.28 s/study mean, 8.12 s p95, 3.2 GiB peak GPU memory, projected
  1.93 h for ~1,300 studies (9 h cap).
- Profiling showed **DICOM decoding, not the model, takes ~52% of inference
  time** (84% for a single model), which shaped the Efficiency Prize strategy.
- Per-study fallback to the training prior, plus checks after writing the
  submission (header, row count, unique IDs, NaN, value range).

---

## 6. Results timeline

| Stage | Public LB macro AUC |
|---|---|
| Frozen DINOv2-S embeddings + aggregation head | 0.787 |
| Fine-tuned ConvNeXt-T, 8 slices × 3 series | 0.845 |
| ConvNeXt-T, 16 slices × 3 series | 0.869 |
| ConvNeXt-T, 16 slices × 5 series (single fold) | 0.889 |
| ConvNeXt-T, 24 slices × 5 series (single fold) | 0.891 |
| 5-fold ConvNeXt-T ensemble | 0.906 |
| 10-model 16- and 24-slice ensemble | 0.911 |
| **15-model 16/24-slice + protocol-selection ensemble** | **0.914** |
| Leaderboard leaders (reference, 2026-09-09) | 0.954 |

Improvement from the first baseline: **+0.127 macro AUC**.

Other measured results:
- Label quality against radiologist gold: **0.854 → 0.891** macro AUC.
- Doubling slices from 8 to 16 gave the biggest single jump: gold AUC
  0.817 → 0.840.
- Side classifier: **98.97%** study accuracy.

---

## 7. Experiments that didn't work, and why (good interview material)

| Experiment | Result | Lesson |
|---|---|---|
| Report-embedding auxiliary loss | Best validation ever (0.881), but LB **0.880 vs 0.889** | Validation labels come from the same reports, so the gain was circular |
| Gradient-cosine-gated aux loss (Du et al., 2018) | No collapse, no gain (gold 0.856) | The method's guarantee held; it just didn't help here |
| Fine-tuned DINOv2-S | Gold 0.820 vs 0.861 | ConvNeXt fit this data better |
| Frozen + fine-tuned blend | 0.839 vs 0.845 fine-tuned alone | Blending a weak ranker with a strong one pulls the strong one down |
| 4×4 spatial patch embeddings | Gold 0.770 → 0.747 | 8.5× more dimensions with no extra labels |
| Per-finding fusion weights fitted on 58 gold studies | 0.860 in-sample, 0.812 leave-one-out | 58 studies can't support per-finding fitting |
| Prompt variants, 32B re-extraction, prompt ensembles | None beat the shipped labels by the +0.005 gate | Run-to-run noise (~0.004 sd) was as big as the whole spread |
| Class-wise attention pooling | Failed standalone and ensemble gates on 2 folds | Stopped instead of tuning weights to rescue it |
| Manually reviewed label corrections | Δ −0.003, CI [−0.011, +0.004] | Small, targeted label fixes don't move a 4,349-study corpus |
| Synovitis-specific work (ranking loss, residual head, frozen path) | Closed after 4 measurements | Judged on effusion-stratified conditional AUC; pooled AUC was a proxy for effusion |

---

## 8. Evaluation and scientific rigour

- **Worked out what the gold set can resolve**: using Hanley-McNeil, 58
  studies give a macro AUC standard error of 0.015–0.025, so gold differences
  under ~0.02 aren't treated as evidence.
- **Found a biased checkpoint selector**: validation labels share the
  training labels' ~18% disagreement with the truth. Tested whether selection
  cost anything (it cost 0.000 on ConvNeXt) before changing the pipeline.
- **Paired bootstrap confidence intervals** on OOF predictions for every
  ensemble or architecture decision, with **gates set before each run**
  (e.g. +0.003 macro, no finding below −0.020).
- **Matched A/B pairs**: same fold, seed, initialisation, data and recipe, and
  a checked initial-weight hash.
- **Effusion-stratified conditional AUC** separates synovitis signal from its
  correlation with effusion.
- **Measured cross-split leaderboard spread (~0.012)** from two identical
  configs on different folds, which is why I stopped comparing single folds.
- Kept a dated work log of every result, including failures and corrected
  claims.

---

## 9. Silent bugs found and fixed

- **Normalisation buffer didn't travel with checkpoints**
  (`persistent=False`): gold AUC was 0.864 during training and 0.823 after
  reloading. Fixed with persistent buffers plus a compatible loader.
- **Training and submission picked different series**: training used
  `np.linspace` sampling, inference used head truncation. That was wrong for
  32.5% of studies and raised no error.
- **Alphabetical checkpoint recovery**: `-v1` sorts before `.`, so a resumed
  session loaded the wrong weights.
- **Sagittal mirroring bug**: the column flip swapped front and back instead
  of medial and lateral. Fixed in code without rebuilding the 29 GB cache,
  because volumes were stored as acquired.
- **Ensemble double-counting**: a fold's `.pt` and `.ckpt` held identical
  weights, so both could be averaged. Added duplicate-fold and
  checkpoint-family checks.
- **Lightning DDP issues**: `torch.cuda.is_available()` breaking forked
  processes, callback hook order returning NaN metrics, and the parent process
  losing `best_model_score`.
- **Diagnostic timer causing SIGSEGV** in long DDP runs on Kaggle, found from
  stack-dump timing across failed runs.
- **Label target-shape dilution** (worth +0.016) and a silence down-weighting
  step that had been quietly switched off.

---

## 10. Scale and key numbers

| | |
|---|---|
| Raw data | 570 GB DICOM, ~820k file headers |
| MRI series | 24,371 (~5.5 per study) |
| Training studies | 4,407 (4,349 report-only, 58 radiologist-labelled) |
| Cache | ~29 GB, 24 × 224 × 224 uint8, 3 shards |
| Label matrix | 4,349 × 12 soft targets + weights (104,376 cells, rebuild verified) |
| Report languages | ~11 |
| Backbone | ConvNeXt-T, 27.8M params; full model 38.7M |
| Final ensemble | 15 models (3 families × 5 folds) |
| Inference | 5.3 s/study, 3.2 GiB GPU, ~1.9 h projected for the hidden test set |
| Codebase | 52 tools, 20 generated notebooks, 43 test modules, 803 tests |
| Git history | 160+ commits |

---

## 11. Ready-to-use resume bullets

### Short set (3 bullets)
- Built a multi-label knee MRI classifier (12 findings) for the RSNA Kaggle
  competition. Reached **0.914 macro ROC AUC** with a 15-model ConvNeXt
  ensemble, up from a 0.787 baseline.
- Produced training labels for 4,349 studies from multilingual radiology
  reports using locally hosted **Qwen2.5 LLMs (vLLM, AWQ)**. Averaging
  categorical and probability extraction modes raised label AUC against
  radiologist gold from **0.854 to 0.891**.
- Engineered a **570 GB DICOM → 29 GB** preprocessing pipeline with geometric
  slice ordering, a **98.97%-accurate laterality CNN** and plane-aware
  mirroring, verified byte-for-byte against raw data.

### Deep learning / computer vision
- Designed a hierarchical multiple-instance model: 2.5D ConvNeXt slice
  encoder, attention pooling, FiLM metadata conditioning and a masked
  transformer across MRI series, trained with PyTorch Lightning DDP on 2× T4.
- Tested ConvNeXt-V2, DINOv2, slice transformers and residual 3D spatial
  encoders with per-plane adapters, using matched A/B pairs with fixed
  initialisation and pre-set gates.
- Raised public leaderboard AUC from 0.845 to 0.914 by scaling slice and
  series coverage, protocol-aware series selection and a multi-family 5-fold
  ensemble.

### NLP / LLM
- Built a structured-extraction pipeline over reports in ~11 languages with
  Qwen2.5-14B and 32B-Instruct (AWQ) on vLLM. It returns presence, severity,
  certainty and evidence spans as JSON, with no translation step.
- Found that categorical LLM outputs tied 53% of label slots, which AUC can't
  rank. Fixed it by adding a probability-valued extraction pass and averaging
  both modes, which beat a per-finding oracle (0.889 vs 0.885).
- Measured the limits of report-derived labels: 27% of positives never
  appear in the text, and the rule-based lexicon is 3.2× less informative in
  Bulgarian than in English.

### ML engineering / MLOps
- Generated Kaggle and Colab notebooks from a single tested Python codebase
  so training and inference preprocessing match, with a CI test that fails
  when a notebook drifts from the source.
- Wrote 800+ pytest tests aimed at silent ML failures (padding invariance,
  mirroring correctness, gold-set leakage, checkpoint round-trips). Found and
  fixed bugs worth up to 0.04 AUC that raised no errors.
- Ran GPU jobs on Kaggle and Colab (T4, A100, L4) with SHA-256
  receipts, resumable KaggleHub downloads, offline inference benchmarks
  (5.3 s/study, 3.2 GiB) and a 9-hour runtime budget.

### Data science / statistics
- Showed that a 58-study gold set can't resolve AUC differences under ~0.02
  (Hanley-McNeil), and moved model and ensemble decisions to paired bootstrap
  CIs over 4,349 out-of-fold studies.
- Chose a 15-model ensemble on a paired within-fold bootstrap
  (Δ +0.0086 macro AUC, 95% CI [+0.0068, +0.0101]). It improved all 5 folds
  and all 12 findings, and confirmed on the leaderboard at 0.914.
- Stopped weak experiments against pre-set gates, including an auxiliary loss
  whose validation gain was circular (+ on validation, −0.009 on the
  leaderboard).

---

## 12. Skills keywords (for ATS matching)

**Machine learning:** deep learning, computer vision, medical image
analysis, multi-label classification, weak supervision, noisy labels,
multiple-instance learning, attention pooling, transformers, FiLM
conditioning, transfer learning, fine-tuning, self-supervised backbones
(DINOv2), ConvNeXt, ConvNeXt-V2, Vision Transformer, 2.5D / 3D CNNs, ensemble
learning, model soups, EMA, SWA, cross-validation, auxiliary losses,
multi-task learning, data augmentation, class imbalance

**NLP / LLMs:** large language models, information extraction, structured
output, prompt engineering, self-consistency, prompt ensembling,
multilingual NLP, quantisation (AWQ), vLLM, Hugging Face Transformers,
Qwen2.5, sentence embeddings (multilingual-E5), rule-based NLP, negation
detection

**Medical imaging:** DICOM, pydicom, MRI, radiology reports, image
geometry, anatomical planes, laterality, musculoskeletal imaging, RSNA

**Tools:** Python, PyTorch, PyTorch Lightning, timm, NumPy, pandas, PyArrow,
Parquet, OpenCV, Matplotlib, pytest, Git, Jupyter, Kaggle API, KaggleHub,
Google Colab, Google Drive

**Infrastructure:** distributed data parallel (DDP), multi-GPU training,
mixed precision, gradient accumulation, NVIDIA T4 / A100 / L4, GPU profiling,
offline inference, reproducibility, artifact versioning, checksums

**Statistics and evaluation:** ROC AUC, macro AUC, bootstrap confidence
intervals, paired tests, McNemar test, Hanley-McNeil standard error,
conditional / stratified AUC, leave-one-out, jackknife, out-of-fold
evaluation, experiment design, A/B testing, error analysis

---

## 13. Interview talking points (STAR-style)

1. **"Tell me about a hard data problem."** Only 58 of 4,407 studies had
   labels, and test data had no reports. *Action:* turned the reports into
   training labels with two LLM extraction modes plus a lexicon, weighted by
   confidence, with nothing fitted on the gold set. *Result:* label AUC
   0.854 → 0.891, and the imaging model improved alongside it.
2. **"A time validation misled you."** A report-embedding auxiliary loss gave
   my best validation and gold scores, but the leaderboard dropped 0.009.
   *Diagnosis:* validation labels came from the same reports the loss
   embedded. *Change:* treat validation gains from report-derived losses as
   circular and gate them on held-out evidence.
3. **"A subtle bug."** Gold AUC was 0.864 in training but 0.823 after
   reloading the same weights. A `persistent=False` normalisation buffer
   wasn't saved in the checkpoint, and a strict load didn't complain. Fixed
   it and added a regression test.
4. **"How do you decide an experiment worked?"** Pre-set gates, matched
   controls, paired bootstrap CIs on 4,349 OOF studies. I also showed the
   58-study gold set's standard error (0.015–0.025) was too wide to rank most
   changes.
5. **"Domain knowledge that mattered."** On sagittal MRI, left-right runs
   along the slice axis, so mirroring means reversing slice order. Flipping
   columns would put the patella behind the knee. The cache stored volumes as
   acquired, so the fix was a code change, not a 29 GB rebuild.
6. **"Performance and cost trade-offs."** Profiling showed half of inference
   time was DICOM decoding, not the model, so extra ensemble members cost only
   ~0.2 h each. That drove separate entries for the accuracy track and the
   Efficiency Prize track.

---

## 14. Honest-framing notes (read before using a number)

- **0.914 is a public leaderboard score** (submission 56196386, 2026-09-13).
  The private leaderboard and final rank aren't known until after
  2026-10-22. Don't claim a placement until then.
- The leaders sit around 0.954. Present this as a strong solo result with a
  clear methodology, not a top finish, unless the final standings say
  otherwise.
- A later 18-model variant with slice transformers also scored 0.914, but
  its submission ID wasn't independently verified. Use the 15-model ensemble
  as the headline.
- Numbers such as the +0.0086 ensemble gain are **out-of-fold proxy scores
  against mined labels**, not leaderboard deltas. Say so if asked.
- The spatial 3D encoder, per-plane adapters and 384-pixel resolution are
  **built and tested, not yet trained**. List them as in progress, not as
  results.
- The competition data license (RSNA MIRA) doesn't allow sharing the data,
  so share code and write-ups only, never images or reports.
