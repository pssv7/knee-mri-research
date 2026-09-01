# Plan — model selection and training constraints

State document: `PROJECT.md` (read it first; it holds what exists and what
failed). This file holds only what happens next and the gates between steps.
History and per-run numbers live in `docs/worklog.md`.

Adopted 2026-08-28, replacing the phase plan that ran to Phase E. That plan
closed its own levers: Phase A (label delivery) and B (capacity) are done,
Phase C (label quality) measured flat inside the gold set's resolution, and
Phase C2 (auxiliary losses) lost 0.009 on the leaderboard. What remains is
below, and it is a different shape: correctness fixes, then selection-free
training, then aggregation.

---

## How this plan is used

**Follow it as written.** Where a judgement call departs from the text — a
different geometry, a reordered item, a gate read differently — say so and get
it agreed before acting, rather than substituting reasoning for the plan. Two
departures were taken silently on 2026-08-28 (running P1 at 24 slices, and at
`MAX_SERIES` 6) and both had to be unwound.

Two items cannot be executed exactly as written; both are recorded below with
the reason, and neither has been quietly dropped:

- **MixUp** (§3.4, §3.5d, P4.13) — not implementable at `BATCH_STUDIES = 1`.
- **Greedy soup gated on gold** (P2 item 7) — conflicts with the project's
  standing rule that gold selects nothing.

## Status — what has already landed

Code, not results. Every gate below still needs the run that answers it.

| item | state |
|---|---|
| P0.1 normalisation travels in the checkpoint | **done** — `KneeNet` registers the buffers persistently and adopts a checkpoint's own statistics on load (`src/model/net.py`) |
| P0.2 submission picks series by `np.linspace` | **done** — `pick_series` in notebook 10 matches `StudyIndex.series_for` |
| P0.3 no alphabetical checkpoint recovery | **done** — `recover_checkpoint` prefers the exact name, then the newest file |
| P0.4 `MAX_SERIES` 5 → 6 | **DONE, run pending** — set to 6 in notebook 08 and `TUNED_MAX_SERIES` with it. This is the run to launch; the cap is the only change in it, compared against the P1 run's val 0.874 / gold 0.882. Was **NEXT CHANGE** — applied, then held at 5 for the P1 run so gold differs from the 0.861 baseline on run length alone. Agreed 2026-08-29: raise it to 6 as the single change in the run immediately after P1, where its gate (gold ↑ or flat) can actually be read. Notebook 10's `TUNED_MAX_SERIES` moves with it — a test ties them |
| P1 fixed-absolute warmup | **done** — `KneeLitModule(warmup_steps=...)`; `schedule="cosine"` also available. **The 20-epoch run has not been made.** Configured at 16 × 5, the reference geometry: at 24 slices ConvNeXt has already produced the confounded shape (val peaked ep7, gold fell 0.865 → 0.842 — both curves down is overfitting, not a biased selector), so P1 is unreadable there |
| P2 EMA + SWA | **done** — `knee_lightning.WeightAverage` writes `avg_fold{N}_ema.pt` and `_swa.pt` from inside `fit`; notebook 08 scores all three candidates against gold |
| P2 model soup | **partly** — `averaging.average_states` / `greedy_soup` exist and notebook 10 has a `SOUP` switch, defaulted off. Folds are now seeded identically so a soup is well-posed; older checkpoints are not |
| P3.8 attention MIL | **already present** — the aggregation has always used attention pooling over slices and series. `gated_pool=True` adds Ilse et al.'s gated variant (eq. 9) as the untried upgrade |
| P3.10 ConvNeXt-V2 | **ready** — `backbone_defaults` keeps the convnet path for V2 and for DINOv3-ConvNeXt, so the swap is one constant |
| §3.4 augmentation | **done, off** — `src/model/augment.py`: photometric (scale/shift/gamma), series-shared gaussian noise, cutout with a fixed box through the stack, slice reversal, and a study-level laterality flip that swaps `LATERAL_LABEL_PAIRS` in the targets. All series-consistent. `AugmentConfig()` reproduces the incumbent single intensity scale exactly, so the default changes nothing. **Rotation/affine excluded** (needs scipy/cv2 per-slice interpolation — a loader bottleneck, not a free regulariser); **arbitrary slice permutation excluded** (breaks `stack_2p5d`'s neighbour channels; `slice_reverse` is the order change that does not) |
| §3.8 flip TTA | **done, off** — `FLIP_TTA` in notebook 10. The plan calls hflip "safe post-laterality-normalisation"; it is not, alone — a mirrored knee transposes four findings. `augment.lateral_swap_order()` permutes them back, and `augment.flip_volume` takes its axis from the same `PLANE_MIRROR_AXIS` table `mirror_volume` canonicalises with. Off because it doubles inference. **The cap reasoning here was never measured and is wrong**: the split was measured 2026-09-01 at 2.74 s/study decode and 0.50 s/study/fold forward, so five folds is 2.18 h and even five folds at 24 slices is 2.70 h against a 9 h cap. Flip TTA stays off on the efficiency arithmetic -- doubling the forward costs 1.05 h = 0.052 AUC-equivalent -- not because it would not fit |
| §3.9 rank-averaging | **done, off** — `averaging.rank_average`, wired as `MEMBER_RANK_AVERAGE`. Note the gap this closed: `to_ranks` already rank-blended *frozen vs tuned*, but members were averaged as probabilities inside `tuned_predict`, so it was never applied **across folds** — the only ensemble that exists now that `BLEND = 1.0` retires the frozen path |
| P3.11 CNN + GRU/Transformer over slices | **not started** — deliberately. The plan gates P3 behind P2, and P2's own gate (EMA/SWA vs the selected checkpoint) has not been read yet |
| P4.13 noise-robust loss | **partly** — `label_smoothing`, `pos_weight` and `symmetric_beta` on `weighted_bce`, all inert by default. **MixUp is not implemented**, see below |
| P4.14 grouped folds | **done, defaulted off** — `splits.make_folds(groups=...)` with `splits.scanner_key`, and `GROUPED_FOLDS` in notebook 08. Off because turning it on re-cuts every fold and invalidates fold 1's baseline; adopt it when the ensemble is rebuilt |
| P3.9 higher resolution | **not started** — needs a cache rebuild at `CACHE_SIZE` 288 or 384; upsampling the 224 cache adds no information |
| P4.12 larger clean validation set | **not started** — labelling effort, no code to write yet |

**Why MixUp is absent.** The plan asks for it as a noise regulariser (§3.4,
§3.5d, P4.13). It does not fit this architecture as specified. Training runs at
`BATCH_STUDIES = 1` per rank, so a within-batch permutation is the identity and
input MixUp is a no-op; and studies carry per-series metadata that FiLM reads,
which cannot be meaningfully interpolated between two different studies. The
implementable variant is manifold MixUp at the pooled study bottleneck against
a cross-batch queue, which is neither what the cited work measured nor what the
plan asked for. Label smoothing and symmetric cross-entropy carry the same
noise-robustness intent and are implemented. Revisit if the batch size ever
rises.

**Why greedy soup does not use gold.** The plan gates the soup on "soup gold >
best single member". Selecting a soup on the gold 58 is selecting on gold, and
this project has already inflated one figure that way. `greedy_soup` therefore
takes a caller-supplied score function so the scoring set is visible at the
call site; score it on mined validation or an out-of-fold dump. The uniform
soup, which selects nothing, needs no scoring set at all.

---

## Executive Summary / Recommendation

**Keep ConvNeXt as the backbone. Do not switch to DINOv2. The single highest-value action is fixing the model-selection / normalisation / series-selection defects you have already measured — those are worth more than any backbone swap, and the evidence base for the swap is against you.**

Bottom line, in priority order:
1. **The DINOv2-S axis is closed on your own numbers (0.820 gold vs a 0.85 gate; converged 0.864 level with ConvNeXt).** This is consistent with the literature: DINOv2's self-supervised natural-image features transfer only "on par or slightly better" than supervised CNNs on medical classification, underperform on kNN because of domain shift, carry high-norm artifact tokens, and only clearly beat ImageNet at high resolution with full fine-tuning. Nothing in the literature predicts DINOv2-S beating a fine-tuned ConvNeXt on 224px greyscale knee MRI with thousands of labelled studies.
2. **The val/gold divergence is real and is the textbook "early-learning then memorisation" signature of training on noisy (mined) labels.** Your validation set uses the same noisy label distribution as training, so late in training it becomes a *biased* selector, not merely a noisy one. The defensible fix is weight averaging (EMA/SWA/Model Soups), which is selection-free and beats best-val-checkpoint under exactly these conditions.
3. **With only 58 gold studies you cannot resolve a 0.015 AUC difference — the Hanley–McNeil standard error at AUC≈0.86 is ~0.045–0.08 per finding (~0.015–0.025 on the macro average).** You are making selection decisions inside the noise band. Build a larger clean validation set and/or switch to selection-free weight averaging.
4. **Architecture headroom is in the aggregation and the pipeline, not the backbone.** Every RSNA Kaggle winner from 2022–2024 used a 2D/2.5D CNN (EfficientNet-V2, ConvNeXt, CoaT) + sequence model (LSTM/GRU/Transformer) or attention pooling. Replacing your mean-pool with attention-based MIL is the best-supported architectural upgrade.

---

## Competition Facts (established with citation)

**Task.** Twelve binary findings per knee-MRI study — a multi-label classification task (detection/classification; the host also frames localization/categorization). Findings cover the clinically important knee abnormalities: meniscus tears (medial/lateral), cruciate/collateral ligaments, cartilage/osteoarthritis (patellofemoral and compartmental OA), bone marrow lesions, effusion, synovitis, and Baker (popliteal) cysts. The user's own taxonomy references (Lateral OA, Lateral Meniscus, PF OA, etc.) are consistent with 12 findings spanning meniscus, OA compartments, and ligaments. (Source: RSNA press release and challenge page; Kaggle competition page.)

**Metric.** Macro (column-averaged) ROC-AUC across the 12 findings — i.e. unweighted mean of per-finding AUC. This is corroborated by two independent public repositories mirroring the competition ("macro ROC-AUC", "12 binary findings per knee MRI study"). This differs from log loss in a crucial way: **AUC is threshold-free and rank-based, so absolute calibration does not matter — only the ranking of studies within each finding.** It also means each finding is weighted equally regardless of prevalence, so a rare finding you handle badly costs exactly as much as a common one. (Note: confirm on the official Evaluation tab before final submission; the Kaggle page is JS-rendered and I could not fetch its body directly. The metric is reported as macro ROC-AUC by multiple public trackers, and the RSNA 2024 predecessor used mean weighted log loss — so verify this has not changed.)

**Data.** >5,000 knee MRI exams from 16–19 sites across five continents, with associated radiology reports in ~12 languages. Multi-series per study (the user's cache: 4,407 studies / 24,371 series; mean 5.53 series/study). Standard knee MRI protocols include sagittal PD/T2-FS, coronal T1/T2-FS, and axial PD-FS (per MRNet's protocol description, a good prior for what sequences appear). This is the first RSNA challenge to pair images with multilingual free-text reports.

**Labels / provenance — critical.** Training labels are **weakly supervised**: only 58 of 4,407 studies carry radiologist "gold" labels; the other 4,349 are labelled from the radiology report text. Ground truth (the hidden test) is **image-derived** — two MSK radiologists plus an adjudicator, using explicitly severity-thresholded criteria with "on the fence" graded negative. **Report-derived labels agree with image-derived ground truth only ~82%.** That 18% gap is systematic, not random — which is exactly why late training on mined labels drifts away from gold. (Source: public competition-intel repos.)

**Submission mechanics.** Code competition, hidden test set, **≤9-hour runtime, internet disabled**. This is the binding constraint: pretrained weights must be attached as Kaggle datasets (no download at inference). A "most efficient model" prize track exists alongside the main track (total prize pool $77,000). Entry/final deadlines in October 2026 (public intel repo cites final submission 2026-10-22); winners announced November 2026 and recognised at RSNA 2026 (Nov 29–Dec 3, Chicago). (Source: RSNA challenge page; public intel repo.)

**Distribution shift & shakeup risk.** (a) **Site leakage**: random K-fold inflates AUC by ~0.053 via scanner memorisation; one measured model showed a +0.136 grouped-vs-random gap. **Use grouped (by-site or by-scanner) folds.** (b) **Train/test label shift**: you train on report labels but are scored on image-derived labels — an ~18% systematic gap. (c) The 58-study gold set is too small for stable per-fold gold metrics. Expect meaningful public/private shakeup; teams over-fitting to noisy CV or a tiny gold set will move.

**Competitive score.** Public baselines are visible: a public DINOv2 baseline notebook scores **0.809**; one public repo reports a 5-fold ConvNeXt-style ensemble at **0.664** on the public LB (with a weaker pipeline) and a text-only report labeler at 0.791 on the 58 gold studies. The user's own pipeline sits around **gold ≈0.86**. Given macro-AUC on 12 findings, competitive is likely high-0.8s to low-0.9s; the exact bar will depend on the hardest findings (small structures like lateral meniscus).

---

## 1. Backbone / Architecture Selection

### 1.1 Incumbent: ConvNeXt / ConvNeXt-V2

**Verdict: keep it.** ConvNeXt is a proven Kaggle-medical backbone (co-first-place backbone in RSNA 2022 cervical spine, alongside EfficientNet-V2-S) and your fine-tuned ConvNeXt already converges at gold ≈0.861–0.865. ConvNeXt-V2 (Woo et al., CVPR 2023) adds FCMAE self-supervised pretraining + Global Response Normalization: **ConvNeXt V2-B (FCMAE) reaches 84.9% ImageNet-1K top-1, +1.1% over V1-B supervised, and V2-L reaches 85.8%, +1.5% over V1-L supervised** (the paper further notes V2 Base/Large "outperform the next-level model sizes of V1"). It is a drop-in `timm` backbone with the same interface — a cheap, low-risk upgrade to try. Evidence class: **proven (incumbent measured) + cited (ConvNeXt-V2 paper)**.

### 1.2 Why DINOv2 underperformed here (and will keep underperforming)

Your measured result — DINOv2-S fine-tuned gold 0.820, converged 0.864 level with ConvNeXt, three findings *worse* than frozen DINOv2 (Lateral OA 0.820 vs 0.899, Lateral Meniscus 0.621 vs 0.680, PF OA 0.736 vs 0.772) — is consistent with the published evidence:
- **Bassi/Baumgartner et al., "Evaluating General Purpose Vision Foundation Models... DINOv2 on Radiology Benchmarks" (arXiv 2312.02366, 2023):** across 200+ evaluations on X-ray/CT/MRI, DINOv2 "performs on par or slightly better compared to supervised methods" for classification and **underperforms on kNN** because of "the domain shift between natural images in the pre-training and medical images." It is strong on *segmentation*, not classification. So a large classification win over a fine-tuned CNN was never on the cards.
- **Register/artifact issue (Darcet et al., "Vision Transformers Need Registers," ICLR 2024, arXiv 2309.16588):** DINOv2 feature maps contain high-norm artifact tokens — "roughly 10× higher norm at the output and correspond to a small fraction of the total sequence (around 2%)," appearing in the middle layers only after sufficiently long training of a sufficiently big transformer — that degrade dense features and interpretability. Register-token variants exist in `timm`/HF and are a cheap thing to try *if* you insist on a ViT, but they fix dense-prediction artifacts more than they fix classification transfer.
- **Resolution dependence (Tayebi Arasteh et al., "Resolution scaling governs DINOv3 transfer performance in chest radiograph classification," arXiv 2510.07191, 2025):** even DINOv3 only becomes the strongest initialisation at **512×512** (AUROC gains of 0.8–1.5 percentage points over DINOv2/ImageNet in 5/6 adult datasets), and **ConvNeXt-B remained superior to ViT-B/16** under both full and parameter-efficient adaptation; frozen DINOv3-7B features *underperformed* fully-finetuned 86–89M backbones. At your 224px, ViT-family self-supervised encoders have no expected edge.

**Would DINOv3 change the picture?** Possibly, but conditionally. DINOv3 (Siméoni et al., Meta, arXiv 2508.10104, Aug 2025) ships a **DINOv3-pretrained ConvNeXt** variant, and the radiology paper above shows DINOv3-ConvNeXt-B is the strongest initialisation *at 512px with full fine-tuning*. This is the one ViT/SSL-adjacent branch worth a probe — but only as a ConvNeXt initialisation at high resolution, not as a frozen ViT-S. Falsifier: if DINOv3-ConvNeXt at 384–512px does not beat ImageNet-ConvNeXt by >1 SD (>0.004 gold) it is dead.

### 1.3 Medical-domain pretrained encoders

The honest finding: **no off-the-shelf medical encoder is clearly better than ImageNet ConvNeXt for this task, and none even ships as a ConvNeXt.**
- **RadImageNet (Mei et al., Radiology: AI 2022, ryai.210315):** reports **+4.8% AUC (ACL, P<.001) and +4.5% (meniscus)** over ImageNet — but on *tiny* datasets (ACL: ~1,000 images; meniscus: ~4,200), and in the same paper the gains **shrink to 0.9–1.9% on the larger datasets** (1.9% pneumonia, 1.7% SARS-CoV-2 CT, 0.9% intracranial hemorrhage; 6.1% COVID-19 the outlier). RadImageNet weights exist only for **ResNet50/DenseNet121/InceptionV3/InceptionResNetV2 — no ConvNeXt.** A published correction (Thomas, "Possible Flaw in Recent Publication," Radiol AI 2023, e230003) found a code bug that inflated the ACL result (corrected AUCs lower, direction preserved). And the independent replication **Juodelyte et al., "Revisiting Hidden Representations in Transfer Learning for Medical Imaging" (TMLR 2023, arXiv 2302.08272)** found "ImageNet fine-tuned after freezing the classification layer leads to the highest AUCs in **six out of the seven datasets. Only knee reaches the highest performance with pre-trained RadImageNet weights, though... both ImageNet and RadImageNet performances were comparable for this dataset.**" For a competition with thousands of studies, the medical-pretraining edge is expected to be near zero.
- **BiomedCLIP (Zhang et al., NEJM AI 2025, arXiv 2303.00915):** strong on chest X-ray/VQA (ViT-B/16 + PubMedBERT, pretrained on 15M figure-caption pairs); **no MRI head-to-head vs ImageNet found**. Weak/indirect evidence only.
- **Merlin / CT-CLIP / CT-FM / MedSAM / SAM-Med:** CT- or segmentation-oriented; no evidence of beating ImageNet ConvNeXt for knee/MSK MRI classification.

Evidence class for "medical encoders don't help here": **strongly suggested by evidence** (multiple converging papers, none knee-MRI-Kaggle-specific).

### 1.4 3D vs 2.5D vs CNN+sequence hybrids — what actually wins

The strongest, most directly transferable evidence is the RSNA Kaggle winner lineage and a 2025 Siemens benchmark:
- **RSNA 2022 (Cervical Spine) 1st place:** 2.5D CNN (EfficientNet-V2-S or ConvNeXt) per slice + **LSTM** over slices; patient-level prediction over all 7 vertebrae (7×15 slices) through the same 2.5D CNN+LSTM, ensembled across backbones/folds.
- **RSNA 2023 (Abdominal Trauma) 1st place (Nischaydnk):** 2.5D — CoaT-Lite Small/Medium and EfficientNet-V2-S per slice-triplet + **GRU** sequence head, auxiliary segmentation loss, max-pool over slice logits. (2nd/other teams used CNN+biGRU+attention with EfficientNet-V2-S/ResNest50d.)
- **RSNA 2024 (Lumbar Spine) top solutions:** localize-then-classify; 2D backbone + attention head for keypoints, then 2.5D crops classified (2nd place: Ian Pan et al.; 1st place "Avengers").
- **Liu et al. (Siemens), "Revisiting 2D Foundation Models for Scalable 3D Medical Image Classification" (arXiv 2512.12887, Dec 2025):** across 12 tasks, **"2D-based methods surpass 3D architectures for 3D classification"**, and **"substantial performance gains (∆AUC=0.11) can be achieved with the same DINOv3 backbone by simply replacing this strategy [average-pooling of slice embeddings] with our method [query-based attention pooling]."** This is the single most important architectural lever.

**Conclusion:** stay 2.5D (2D CNN per slice + sequence/attention aggregator). Full 3D CNNs (3D ResNet, X3D, Video Swin, VideoMAE) are *not* supported by the Kaggle evidence for this class of task and lack good pretrained weights; deprioritise.

### 1.5 MRNet / knee-MRI-specific literature

- **MRNet (Bien et al., PLoS Medicine 2018, 15(11):e1002699):** AlexNet per slice, max-pool over slices, logistic regression to fuse 3 series (sagittal T2, coronal T1, axial PD). Internal-validation AUCs: **abnormality 0.937 (95% CI 0.895–0.980), ACL 0.965 (0.938–0.993), meniscus only 0.847 (0.780–0.914)** — meniscus is the hard, small-structure finding, mirroring your lateral-meniscus weakness (0.621–0.680).
- **ELNet, MRPyrNet (Dunnhofer et al., Comput Med Imaging Graph 2022):** feature-pyramid / detail-pooling additions specifically to catch small localized lesions (meniscus/ACL); MRPyrNet improves both MRNet and ELNet. Implication: **small-structure findings benefit from higher resolution and multi-scale features** — relevant to lateral meniscus.
- **Multi-view fusion:** multiple papers confirm three-plane fusion > single-plane for ACL/meniscus, with attention across planes helping; reported ACL AUCs on MRNet reach 0.96–0.98 with good fusion.
- **OA/KL-grade & MOAKS (OAI/MOST):** 3D CNNs grade cartilage/BME/meniscus/ACL with AUC 0.83–0.93 (Radiology: AI 2021, ryai.2021200165); MR-Transformer (Radiology: AI 2024, ryai.240373) uses an ImageNet-pretrained ViT + 3D spatial correlations for TKR prediction. These confirm the finding-specific difficulty ordering (OA/effusion easier; meniscus/cartilage hard).

### 1.6 Ranked backbone shortlist (with expected gain and falsifier)

| Rank | Option | Expected gold gain vs current ConvNeXt | Evidence class | Falsifier |
|---|---|---|---|---|
| 1 | **Keep ConvNeXt, add attention-MIL aggregation** | +0.005 to +0.02 | Strongly suggested (Siemens ∆0.11 pooling; Ilse MIL) | No gain over mean-pool at matched slices/series |

**Correction to rank 1.** This pipeline has never mean-pooled. `KneeNet` has used masked attention pooling (Ilse et al. eq. 8) over both slices and series since the frozen path, with a transformer over the series axis between them. So the headline gain here is already banked, and what is actually untried is the *gated* attention of eq. 9 — `gated_pool=True`, a strictly larger family, worth perhaps a fraction of the quoted range rather than the whole of it. The Siemens ∆0.11 is against average pooling, which is not the baseline here.
| 2 | **ConvNeXt-V2-Base (FCMAE)** as drop-in | +0.003 to +0.01 | Cited (CVPR 2023, +1.1–1.5% ImageNet) | ≤1 SD (0.004) over ConvNeXt-V1 |
| 3 | **Higher input resolution (288–384px)** on ConvNeXt | +0.005 to +0.015, concentrated on meniscus | Cited (MRPyrNet; DINOv3 resolution paper) | Meniscus AUC flat at higher res |
| 4 | **DINOv3-ConvNeXt @384–512px, full fine-tune** | 0 to +0.01 | Cited but off-domain (chest X-ray) | <0.004 over ImageNet-ConvNeXt |
| 5 | 2D CNN + GRU/Transformer over slices (vs pure pooling) | +0.005 to +0.02 | Proven (RSNA winners) | No gain over attention-MIL |
| — | DINOv2-S (any config) | negative to 0 | Proven closed (your 0.820) + cited weakness | — (already falsified) |
| — | Full 3D / video nets | unknown, likely negative | Evidence against (Siemens 2D>3D) | — |
| — | RadImageNet / BiomedCLIP / MedSAM | ~0 | Evidence against / absent | — |

---

## 2. The Val/Gold Divergence and Checkpoint Selection

### 2.1 Diagnosis: early-learning then memorisation of noisy labels

Your epoch curve — val (mined) rising to 0.860 (ep9) then falling to 0.843 (ep20) while gold holds flat at ~0.864–0.869 over eleven straight epochs — is the canonical signature described by **Liu et al., "Early-Learning Regularization Prevents Memorization of Noisy Labels" (NeurIPS 2020, arXiv 2007.00151)** and **Arpit et al., "A Closer Look at Memorization in Deep Networks" (ICML 2017):** networks first fit the clean/consistent structure, then memorise label noise. Because your validation labels are drawn from the *same mined distribution* as training (with the same ~18% systematic disagreement from gold), late-training memorisation *raises* mined-val agreement even as true (gold) performance plateaus. **This makes mined-val a biased selector, not just a noisy one** — exactly your hypothesis, and it is well-grounded.

### 2.2 What the literature recommends

- **Robust losses / regularisers that delay memorisation:** ELR (Liu 2020); symmetric cross-entropy; Generalized Cross Entropy (Zhang & Sabato, NeurIPS 2018); bootstrapping; label smoothing; MixUp as a noise regulariser. These make the *whole run* more robust and reduce how much the selection choice matters.
- **Selection-free weight averaging** — the cleanest fix for your exact problem:
  - **SWA (Izmailov et al., UAI 2018, arXiv 1803.05407):** equal-weight average of weights along the SGD trajectory finds flatter, better-generalising optima — "SWA is extremely easy to implement, architecture-agnostic, and improves generalization performance at virtually no additional cost over conventional training" (e.g. +0.8% for ResNet-50/DenseNet-161 on ImageNet running SWA for just 10 epochs). Now native in PyTorch.
  - **EMA / Mean Teacher (Tarvainen & Valpola, NeurIPS 2017):** exponential moving average of weights; standard, robust, cheap.
  - **Model Soups (Wortsman et al., ICML 2022, arXiv 2203.05482):** averaging weights of multiple fine-tunes "often improves accuracy and robustness" over picking the single best-val model — and explicitly attacks the "pick the individual model which performs best on a held-out validation set, discarding the remainder" recipe that is failing you.
- **Clean-validation and agreement-based selection:** where a small clean set exists, use it (or an ensemble-agreement proxy) rather than the noisy val.

### 2.3 Small-validation-set statistics — the hard number

Using the **Hanley–McNeil (1982)** approximation for the standard error of a single AUC:

SE(AUC) = √[ (A(1−A) + (n_p−1)(Q1 − A²) + (n_n−1)(Q2 − A²)) / (n_p · n_n) ],
where A = AUC, Q1 = A/(2−A), Q2 = 2A²/(1+A), n_p/n_n = positives/negatives.

For **n=58 studies, AUC≈0.86**, positive rate ~30% (n_p≈17, n_n≈41): Q1 = 0.86/1.14 = 0.754, Q2 = 2·0.7396/1.86 = 0.795. Numerator ≈ 0.86·0.14 + 16·(0.754−0.7396) + 40·(0.795−0.7396) = 0.1204 + 0.230 + 2.216 = 2.567; denominator = 17·41 = 697; SE ≈ √0.00368 ≈ **0.061**. For a commoner finding at 50% prevalence (n_p=n_n=29) SE ≈ **0.045**; for a rare finding (n_p≈6) SE exceeds **0.08**.

**So the per-finding SE on 58 gold studies is ~0.045–0.08.** The macro average over 12 (partly correlated) findings reduces this by up to ~√12 ≈ 3.5× in the best case, giving a macro SE on the order of **0.015–0.025** — i.e. **your run-to-run sd of 0.004 is measuring optimiser noise, but your true gold estimate has an SE of ~0.02, and a 0.015 difference is inside one SE.** You cannot resolve it.

**How many gold studies to resolve Δ=0.015?** To get a per-finding 95% CI half-width of ~0.015 (SE≈0.0075) at AUC≈0.86, Hanley–McNeil / presize-style calculations require on the order of **several hundred to ~1,000 studies per finding** at moderate prevalence (the NCSS/Hanley worked examples show ~976 per group for a 0.05 CI width at AUC 0.6; AUC 0.86 needs somewhat fewer but still many hundreds). Even exploiting the macro average across 12 findings, you need **roughly 300–500 gold studies** for the macro metric to resolve 0.015 reliably. **Conclusion: stop trying to resolve 0.015 on 58 studies; either expand the clean set to a few hundred or adopt selection-free averaging.**

### 2.4 Recommended checkpoint-selection protocol

1. **Stop using best-mined-val checkpoint selection.** It cost you 0.015 (your worst-case), consistent with the biased-selector mechanism.
2. **Adopt EMA of weights throughout training** (decay ~0.999) and submit the EMA weights; additionally compute an **SWA average over the last ~25% of epochs**. Both are selection-free and directly address the memorisation drift.
3. **Greedy Model Soup** across folds/seeds/mild-hparam variants for the final submission (fits the inference budget — averaging is free at inference).
4. **Enlarge the clean validation set.** The 58 gold studies are the ceiling on decision quality. Options: (a) manually adjudicate a few hundred more studies if feasible; (b) build a higher-precision multilingual report labeler and treat its high-confidence subset as silver validation; (c) use ensemble-agreement / OOF-consistency as a selection proxy. Target ≥300 clean studies.
5. **If you must early-stop, stop on the plateau, not the mined-val peak** — pick the epoch where the gold-proxy first plateaus, and prefer later (averaged) weights over the mined-val argmax.
6. **Fix the OneCycleLR restretch first** (Section 3.6) so epoch indices are comparable across runs before drawing any selection conclusion.

---

## 3. Constraints / Hyperparameters

### 3.1 Slice count & sampling
- Slice-count scaling: your own data shows 8→16 slices was +100% pixels and materially helped; MRNet-family evidence says small structures (meniscus) reward more slices/resolution. Use **uniform `np.linspace` sampling in BOTH training and inference** — your current mismatch (train linspace vs submission head-truncation `volumes[:MAX_SERIES]`) affects 32.5% of studies and is a pure, free bug-fix. Centre-weighted sampling can help for centrally-located findings but risks missing peripheral meniscus horns; keep uniform unless a per-finding ablation shows otherwise.
- Diminishing returns: past ~16–24 slices, expect flattening; your 24-slice run converged level, so treat slice count as roughly saturated and spend compute elsewhere.

### 3.2 Series / plane fusion
- **Attention-based MIL** (Ilse et al., "Attention-based Deep Multiple Instance Learning," ICML 2018, arXiv 1802.04712; gated-attention variant; TransMIL, Shao et al., NeurIPS 2021 for a transformer pooling). Mean pooling under-weights small focal abnormalities that appear on few slices/series; attention pooling learns a trainable weighted average and up-weights them. **Already in place** — `AttentionPool` over slices, a transformer over series, then `AttentionPool` again, all masked. The remaining move on this axis is eq. 9's gated attention (`gated_pool=True`), which replaces the tanh scorer with `tanh(Vh) * sigmoid(Uh)`; expect a fraction of the quoted gain, not the whole of it.
- **MAX_SERIES 5→6:** your accounting shows total remaining headroom on the series axis is only +14%, and 5→6 buys about half of it (+6.7% mean series/study, truncation 32.5%→16.7%). It is cheap (load-time knob, all series cached) and low-risk. Do it, but expect a small gain (well under the +61% you got from 3→5). Going to cap=inf only adds another ~7% of series for 2.3%→0% truncation — marginal.

### 3.3 Input resolution
- Evidence (MRPyrNet; the DINOv3 resolution-scaling paper, which found gains "greatest for small focal and boundary-dependent abnormalities"; MeniMV) says **higher resolution disproportionately helps small structures (meniscus, cartilage)**. Test 288 and 384px. Because AUC is threshold-free, the gain shows up as better ranking on the hard findings. Falsifier: meniscus AUC flat from 224→384.

### 3.4 Augmentation
- RSNA winners used geometric (flip, rotate ±25°, perspective, affine), intensity/photometric jitter, noise/blur, region dropout, slice-order permutation, and **MixUp** (which doubles as a noise regulariser here — directly useful given noisy labels). Keep augs **3D/series-consistent** (same transform across slices of a series). Laterality normalisation (mirror right knees to a canonical side) is used by public baselines and is sensible.

### 3.5 Loss function — align to the metric
- The metric is **macro ROC-AUC (rank-based, threshold-free, equal per-finding weight)**. Therefore: (a) **calibration is irrelevant** — do not spend effort on temperature scaling / isotonic for the LB (it cannot change AUC); (b) use **per-finding BCE with class-balancing** (pos_weight or focal loss) so rare findings are learned, since each finding counts equally; (c) consider a **noise-robust loss** (label smoothing, symmetric CE, GCE, or ELR) given the mined labels; (d) MixUp. Do **not** switch to a log-loss objective expecting LB gains — but do verify the official metric is AUC and not the RSNA-2024-style weighted log loss before finalising; **if it turns out to be weighted log loss, then calibration and per-finding weighting become first-order.**

### 3.6 LR schedule
- Your OneCycleLR `pct_start=0.3` issue is real: at EPOCHS=20 the LR peaks near epoch 6 vs 3.6 at EPOCHS=12, so epoch indices are not comparable across runs and the "selected" epoch shifts. Two fixes: (a) set warmup to a fixed absolute step count so schedules are comparable across run lengths; (b) prefer **cosine decay** for fine-tuning, which is standard in the RSNA/abdominal winners and removes the restretch artefact. For ViTs use **layer-wise LR decay**; for ConvNeXt a mild layer-wise decay (~0.8–0.9) helps but is less critical than for ViTs. Combine with EMA/SWA so the exact schedule matters less.

### 3.7 Cross-validation design
- **Grouped folds by site/scanner** are mandatory (random K-fold inflates AUC ~0.053; measured +0.136 grouped-vs-random gap).
- **Make CV correlate with LB** by validating on a **clean (gold/silver) subset**, not on mined labels — the mined val is the biased selector. Track both mined-val (for training diagnostics) and gold/silver (for decisions).
- **Build a larger clean validation set** (Section 2.4) — this is the highest-leverage CV improvement.

### 3.8 Ensembling & TTA within the 9-hour budget
- Work backwards from ≤9h, internet off, single Kaggle GPU (public baselines used T4 and warn that P100 is broken with current Kaggle PyTorch; **confirm the assigned submission GPU**). A 2.5D ConvNeXt at 224–384px over ~5–6 series × 16–24 slices for a few thousand test studies is feasible for a **~3–5 model soup + light TTA (hflip only)**. Prefer **weight-averaged soups (free at inference)** over logit ensembles of many separate checkpoints, and reserve logit ensembling for 2–3 genuinely different backbones (e.g. ConvNeXt + EfficientNet-V2). The "most efficient model" prize track rewards a single lean model — keep one such config.
- **TTA:** horizontal flip is safe post-laterality-normalisation; avoid vertical flip (anatomically implausible) unless ablation supports it.

### 3.9 Post-processing / calibration
- Under macro-AUC, **per-finding monotonic post-processing cannot change the score** (AUC is invariant to monotone transforms). So skip calibration for the LB. The only useful "post-processing" is **rank-averaging across ensemble members** (more robust than mean-of-probabilities for AUC). If the metric is actually weighted log loss, reinstate calibration as first-order.

---

## 4. Prioritised, Sequenced Experiment Queue

Assume A100, unlimited training time; each item lists expected gold gain, A100 cost, decision gate, and kill criterion. Ordered by expected value / risk.

**P0 — Free correctness fixes (do first, all together):**
1. **Fix normalisation travel bug** — register `input_mean`/`input_std` with `persistent=True` (or reconstruct at scoring). This is the leading explanation for the 0.823 vs 0.864 gap (an ImageNet-normalised ViT scored on raw uint8). *Cost: ~0. Gate: re-score reproduces 0.864 at both sites. Kill: n/a — pure bug.*
2. **Match series selection: use `np.linspace` in `tuned_predict`**, not `volumes[:MAX_SERIES]`. Affects 32.5% of studies. *Cost: ~0. Gate: submission-time OOF matches training OOF. Kill: n/a.*
3. **Fix the alphabetical checkpoint glob** (`sorted(glob("lit_fold{N}*.ckpt"))[0]` picks `lit_fold1-v1.ckpt` over `lit_fold1.ckpt`). Pin explicit checkpoint paths. *Cost: ~0. Kill: n/a.*
4. **MAX_SERIES 5→6.** *Cost: ~+7% inference. Gate: gold ↑ or flat with no cost blowup. Kill: no gain and inference budget pressure.*

**P1 — The user's own top hypothesis:**
5. **Rerun reference ConvNeXt at EPOCHS=20 with fixed-absolute warmup, and test whether val/gold divergence reproduces.** If it does, the biased-selector mechanism is confirmed and every fold's selection is costing ~0.015. *Cost: 1 training run. Gate: divergence reproduces → adopt weight averaging fleet-wide. Kill: no divergence → selection rule is fine, revisit.*

**P2 — Selection-free training (highest expected gain given P1):**
6. **Add EMA (decay 0.999) + SWA over last 25% epochs; submit averaged weights.** *Cost: ~0 extra. Gate: EMA/SWA gold ≥ best-mined-val gold + reduces variance. Kill: averaging underperforms argmax on the clean set (unlikely per literature).*
7. **Greedy Model Soup across folds/seeds for final submission.** *Cost: extra fine-tunes (cheap on A100). Gate: soup gold > best single member. Kill: soup < best member on clean set.*

**P3 — Architecture upgrades (backbone stays ConvNeXt):**
8. **Gated attention pooling** (`gated_pool=True`) over slices×series — the pooling is already attention-based, so this is eq. 8 → eq. 9, not mean → attention. *Cost: minor. Gate: +>0.005 gold, especially on meniscus. Kill: no gain at matched inputs.*
9. **Higher resolution 288→384px.** *Cost: ~1.5–3× training/inference pixels (fine on A100; watch inference budget). Gate: meniscus/cartilage AUC ↑. Kill: flat.*
10. **ConvNeXt-V2-Base drop-in.** *Cost: 1 run. Gate: >1 SD (0.004) over V1. Kill: ≤0.004.*
11. **2D CNN + GRU/Transformer sequence head** (vs pure attention pooling). *Cost: moderate. Gate: > attention-MIL. Kill: no gain.*

**P4 — Data / label quality (raises the ceiling on everything above):**
12. **Build a larger clean validation set (≥300 studies)** via adjudication and/or a higher-precision multilingual report labeler. *Cost: labeling effort. Gate: macro SE on clean set < 0.01. Kill: n/a — strategic.*
13. **Noise-robust training: MixUp + label smoothing / ELR / symmetric CE.** *Cost: 1–2 runs. Gate: gold ↑ or divergence reduced. Kill: no gain.*
14. **Grouped-by-site folds (if not already).** *Cost: refold. Gate: CV/LB gap shrinks. Kill: n/a — mandatory.*

**P5 — Speculative / evidence-thin (only if P0–P4 exhausted):**
15. **DINOv3-ConvNeXt @384–512px, full fine-tune.** *Cost: attach weights as Kaggle dataset; 1 run. Gate: >0.004 over ImageNet-ConvNeXt at matched res. Kill: ≤0.004.*
16. **Second backbone for logit/rank ensemble (EfficientNet-V2-S).** *Cost: 1 model. Gate: ensemble > soup alone. Kill: no gain within budget.*

**Explicitly killed / deprioritised:** DINOv2-S (any config) — closed on your 0.820 and against the literature; full 3D/video nets — evidence says 2D>3D for this task; RadImageNet/BiomedCLIP/MedSAM — no ConvNeXt weights and no expected edge at your data scale; LB calibration/temperature scaling — cannot change macro-AUC.

---

## Master Claims Table (claim → evidence class → source)

| # | Claim | Evidence class | Source |
|---|---|---|---|
| 1 | Task = 12 binary findings, macro ROC-AUC | Proven/cited (verify on official tab) | Kaggle comp page; RSNA press release; public intel repos |
| 2 | Labels weakly-supervised; 58/4,407 gold; report vs image labels agree ~82% | Proven (intel repo) | homeshwarnelakurthi & JunhaoLiXD GitHub repos |
| 3 | Code comp, ≤9h, internet off | Proven | RSNA challenge page; intel repo |
| 4 | Random K-fold inflates AUC ~0.053; +0.136 grouped-vs-random gap | Measured (this project's cohort) | intel repo STRATEGY |
| 5 | DINOv2 only "on par or slightly better" than supervised on medical classification; weak kNN due to domain shift | Cited | Bassi/Baumgartner, arXiv 2312.02366 |
| 6 | DINOv2 has high-norm artifact tokens (~2%, ~10× norm); registers fix dense features | Cited | Darcet et al., ICLR 2024, arXiv 2309.16588 |
| 7 | DINOv3 beats ImageNet/DINOv2 only at 512px (0.8–1.5pp); ConvNeXt-B > ViT-B/16; frozen features underperform | Cited (chest X-ray, off-domain) | Tayebi Arasteh et al., arXiv 2510.07191 |
| 8 | ConvNeXt-V2 FCMAE: V2-B 84.9% (+1.1%), V2-L 85.8% (+1.5%) ImageNet fine-tune | Cited | Woo et al., CVPR 2023 |
| 9 | RadImageNet +4.8%/+4.5% knee (small data) but 0.9–1.9% on large data; no ConvNeXt weights; ImageNet wins 6/7 in replication | Cited | Mei et al. ryai.210315; Juodelyte TMLR 2023 (arXiv 2302.08272); Thomas correction e230003 |
| 10 | 2D > 3D for 3D medical classification; ∆AUC=0.11 from attention pooling vs avg pooling | Cited | Liu et al. (Siemens), arXiv 2512.12887 |
| 11 | RSNA 2022/2023/2024 winners = 2.5D CNN + LSTM/GRU/attention (EffNet-V2-S, ConvNeXt, CoaT-Lite) | Proven (writeups) | Nischaydnk RSNA2023 repo; darraghdog RSNA22; Kaggle writeups |
| 12 | MRNet: meniscus is the hard finding (0.847 vs abnormality 0.937 / ACL 0.965) | Cited | Bien et al., PLoS Med 2018, e1002699 |
| 13 | Small structures reward resolution/multi-scale (FPN) | Cited | MRPyrNet, Comput Med Imaging Graph 2022 |
| 14 | Attention-MIL (trainable weighted avg) > mean/max pooling | Cited | Ilse et al., ICML 2018 (arXiv 1802.04712); Shao TransMIL NeurIPS 2021 |
| 15 | Noisy labels: early-learning then memorisation | Cited | Liu ELR NeurIPS 2020 (arXiv 2007.00151); Arpit ICML 2017 |
| 16 | SWA → flatter optima, better generalization "at virtually no additional cost" | Cited | Izmailov et al., UAI 2018 (arXiv 1803.05407) |
| 17 | Model Soups beat best-val-checkpoint selection | Cited | Wortsman et al., ICML 2022 (arXiv 2203.05482) |
| 18 | Hanley–McNeil SE(AUC); n=58 gives per-finding SE ~0.045–0.08, macro ~0.015–0.025 | Cited + computed | Hanley & McNeil 1982; computation herein |
| 19 | Val/gold divergence over 11 epochs; selection cost 0.015 | Measured | user session notes |
| 20 | DINOv2-S fine-tune gold 0.820; converged 0.864 level w/ ConvNeXt | Measured | user session notes |

## Caveats & Uncertainties (stated plainly)
- **Metric not confirmed from the official Evaluation tab** (JS-rendered; could not fetch its body). Macro ROC-AUC is reported by multiple public trackers, but the immediate predecessor (RSNA 2024) used mean weighted log loss. **Verify before finalising loss/calibration choices** — if it is weighted log loss, Sections 3.5/3.9 flip and calibration becomes first-order.
- **Exact 12-finding taxonomy and any sample/finding weighting** should be read off the official data description; I inferred the finding families from RSNA's description and the user's own label names (Lateral OA, Lateral Meniscus, PF OA, etc.).
- **Submission GPU** (T4 vs L4/A10/P100) determines the exact ensemble budget; public baselines used T4 and warned P100 is broken. Confirm the assigned GPU.
- **DINOv3-ConvNeXt evidence is chest-X-ray, not knee MRI** — genuinely off-domain; treat P5 item 15 as a hypothesis, not a recommendation.
- **Hanley–McNeil SE assumes independence and a binormal-ish structure**; it can underestimate for correlated rating data, so the true SE on 58 studies may be even larger — strengthening, not weakening, the "58 is too few" conclusion.
- **The +0.11 pooling gain (Siemens paper) was on their 12-task benchmark, not this competition** — the direction is well-supported but the magnitude will not transfer exactly.
- **RadImageNet's knee advantage is confounded** by a code bug (Thomas correction) and a small-data regime; the independent TMLR replication found ImageNet ≥ RadImageNet, so treat any medical-encoder gain at this data scale as ~0.