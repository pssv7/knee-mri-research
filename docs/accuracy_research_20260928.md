# Knee MRI accuracy: literature, specifications, and an evidence-based plan

Research date: September 28, 2026. Primary papers, supplementary experiments,
official model repositories, and this project's saved predictions were examined.
The objective is higher twelve-finding macro ROC AUC from one image model.

**Subsequent experiment:** the implemented frozen OrthoFoundation-L screen
completed on Kaggle and failed: macro 0.795836 versus Tiny 0.878838,
delta −0.083002 [−0.101378, −0.063875]. Keep Tiny. This is evidence against
the tested frozen-feature recipe; full fine-tuning remains untested. The literature below is
background evidence, not a successful prediction of this configuration's gain.

**Recommendation: prioritize anatomical localization and knee-specific real-image
pretraining. Do not make synthetic pathology generation the next training campaign.**
Localization has a directly relevant controlled knee-MRI ablation; domain
pretraining has measured transfer results and at least one publicly listed
checkpoint. Neither has yet proved an improvement on this competition. A paper's
result is evidence for choosing a test, not a guarantee of our leaderboard gain.

**The proposed model change is ConvNeXt-Tiny to OrthoFoundation-L as the image
encoder.** Keep the twelve-output task and the slice/series aggregation design,
adding a feature adapter where needed. Retain ConvNeXt-Tiny as the unchanged
reference, and compare OrthoFoundation with ordinary DINOv3-L to distinguish
the benefit of knee pretraining from the change in architecture and capacity.
Checkpoint compatibility, preprocessing and memory feasibility must be verified
before training. Adoption depends on the held-out comparison and confirmation
below. The optional encoder and frozen-screen runner are now implemented and
CPU-verified. Its later GPU frozen-head fit completed and failed the accuracy
screen, as recorded above. Test anatomical
cropping separately so its effect is distinguishable from the encoder change.

There are three different standards here: a method worked in its authors'
experiment; it works on our report-derived evaluation; it improves independently
labelled image findings in our target population. Only the first is supplied by
the literature. The second can be measured with our current infrastructure. The
third remains constrained by the lack of new image readers/reference labels.

**Our current model is already a substantial baseline.** The exact control is
`convnext_tiny.fb_in22k_ft_in1k`, 16 slices per series, up to five protocol-selected
series, and 224-pixel images. Adjacent slices occupy the three input channels.
Slice attention, metadata FiLM, a two-layer/eight-head series transformer, study
attention, and twelve outputs follow the encoder. The project records about
38.7M parameters for the complete network. Its recipe uses twelve epochs,
effective batch eight, learning rate 1e-4, and OneCycle scheduling. These are local
facts from `src/model/net.py` and the nested control's run contract.

The official encoder card reports 28.6M parameters including its original
ImageNet classifier, 768-dimensional features, and ImageNet-22k pretraining
followed by ImageNet-1k fine-tuning. These parameter counts describe different
objects and should not be confused. [Official timm model card](https://huggingface.co/timm/convnext_tiny.fb_in22k_ft_in1k).

The latest saved public result is **0.917 for the twenty-model ensemble**, reported
by the user; it is not the control model's score and was not rechecked with Kaggle
in this research. See `outputs/kaggle_combined20_submission/score_update_20260924.json`.

**The low-scoring findings were recomputed, rather than selected from old notes.**
On the current 691-study outer0 report-label set:

| Finding | Control AUC | Positive / negative | AUC on weight ≥0.7 cells | Silent cells |
| --- | ---: | ---: | ---: | ---: |
| PF OA | 0.7894 | 135 / 556 | 0.7421 | 44.9% |
| Lateral meniscus | 0.8128 | 70 / 621 | 0.8387 | 14.2% |
| MCL | 0.8545 | 38 / 653 | 0.8930 | 11.6% |
| Lateral OA | 0.8630 | 66 / 625 | 0.8275 | 55.4% |
| Medial meniscus | 0.8747 | 231 / 460 | 0.8791 | 9.0% |
| Synovitis | 0.8906 | 41 / 650 | 0.7877 | 83.2% |

Macro AUC is **0.878837614**. Weight ≥0.7 identifies a supervision subset, not
independent image truth. Its case mix differs, so the difference between the two
AUC columns does not measure the causal effect of silence. The very small
Fracture positive count, sixteen, also makes per-finding comparisons unstable.

PF OA and lateral meniscus are reasonable prespecified secondary endpoints.
Improving each by 0.030 would contribute 0.005 macro if all other findings stayed
constant. That is arithmetic, not an expected gain. PF OA also has documented
report-extraction/definition errors; a better encoder cannot be assumed to fix
those. Synovitis needs effusion-stratified diagnostics, given this project's
previous conditional analyses.

I independently reproduced all 48 per-finding AUCs for four saved candidates
with scikit-learn, checked prediction hashes against contracts, checked their
evaluation membership and exclusion from training/checkpoint selection, and
confirmed three saved control predictions were identical. The intervals below
are historical readouts, not newly generated intervals:

| Already tested here | Macro change [historical 95% sampling interval] | Interpretation |
| --- | --- | --- |
| Duplicate-free slice sampling | +0.00085 [−0.00714, +0.00884] | Failed declared screen |
| ConvNeXt-Small/224 | −0.00182 [−0.01179, +0.00788] | Failed; hardware/runtime also differed |
| Mirror augmentation | −0.00084 [−0.00844, +0.00721] | Failed; MCL breached loss floor |
| Six-finding silence mask | −0.00684 [−0.01689, +0.00331] | Its actual primary endpoint was also flat: explicit six-finding Δ −0.00007 |

This closes those tested configurations, not every possible use of larger
models, augmentation, or uncertain-label learning. It does make repeating them
without a new information source a poor priority. The earlier 384 family,
spatial residual encoder, and classwise attention also lack a confirmed
standalone gain.

**These are the most relevant models and what their specifications imply.**

| Model | Verified design/input | Evidence and fit to this project |
| --- | --- | --- |
| **CoPAS (2024)** | 3D ResNet-18 encoders; three PD sequences plus coronal T1 and sagittal T2; cross-plane/cross-sequence attention; meniscus-centered crops; volumes initially 224×224×24 | Particularly relevant to multiple series and small structures. Its twelve targets differ from ours. [Paper](https://www.nature.com/articles/s41467-024-51888-4) |
| **KAMRNet (2025)** | Meniscus localization with 3D U-Net; three plane-specific ResNet-18 branches; global and attention-localized crop passes; nine independently trained abnormality classifiers | A concrete localization implementation. Reproducing the full nine-model system would depart from our one-model objective. [Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12508577/), [model source](https://github.com/zqiuak/KAMRNet/blob/main/model_forallplane_MMAL.py) |
| **OrthoFoundation-L (2026)** | DINOv3 ViT-L/16 according to current release; approximately 300M encoder parameters; knee MRI/X-ray continued self-supervision | Closest available domain-pretrained encoder to our slice-based pipeline. Requires adapting feature extraction and downstream aggregation. [Release](https://github.com/ytrsk/OrthoFoundation), [base-model specification](https://github.com/facebookresearch/dinov3) |
| **KneePreM (September 25, 2026 preprint)** | 3D U-Net masked autoencoder; 40×224×224 voxels; 64% masking in 4³ blocks; pretraining on 19,011 OAI series/4,791 participants | Strong controlled evidence for domain pretraining, but a larger input/pipeline change. [Paper](https://arxiv.org/pdf/2609.31461) |
| **MRNet** | 2D slice encoding, spatial pooling and slice max-pooling; combines sagittal, coronal and axial predictions | Useful reference and external data source, not an obvious architectural upgrade to our existing hierarchy. [Authors' architecture description](https://stanfordmlgroup.github.io/projects/mrnet/) |

CoPAS studied 1,748 subjects and reported internal mean AUC 0.812, falling to
0.721/0.726 on its external cohorts. Its surgical population and different label
definitions prevent numerical comparison with our 0.8788 or ensemble 0.917.
Its rotated/reformatted volumes are views of existing acquisitions; they are
not newly generated patients. [CoPAS paper](https://www.nature.com/articles/s41467-024-51888-4).

The most actionable evidence is **CoPAS supplementary Table 10**:

| Published ablation | Mean AUC | Difference |
| --- | ---: | ---: |
| Baseline without anatomical ROI | 0.725 | Reference |
| Baseline with anatomical ROI | 0.781 | **+0.056** |
| Cross-plane attention | 0.774 | Does not exceed the ROI baseline |
| Cross-plane + cross-sequence attention | 0.799 | +0.025 versus preceding row |
| Full CoPAS | 0.812 | +0.013 versus preceding row |

For cropping, meniscus changed 0.624→0.659 and cartilage 0.734→0.778. These are
within-study ablations, stronger support than comparing unrelated papers'
headline AUCs. The table does not provide a paired interval for the ROI contrast.
It does not isolate every effect of sequence choice from added attention.
[Supplement, Table 10](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41467-024-51888-4/MediaObjects/41467_2024_51888_MOESM1_ESM.pdf).

My inference: test localization before importing the whole CoPAS architecture.
Our raw-to-cache preprocessing currently resizes whole frames without anatomical
cropping. A crop taken before downsampling can allocate more pixels to cartilage
and meniscus. Upscaling a crop of an already reduced 224 image cannot recover
detail discarded earlier. This is distinct from the failed whole-frame 384
comparison; neither result settles the other.

KAMRNet supplies corroborating implementation evidence, with 13,419 patients
and external evaluation. Its primary-abnormality mean AUC was 0.898 internally
and 0.852/0.812 externally. These values also demonstrate transfer limitations.
[KAMRNet paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12508577/).
The official repository links trained weights; downloadability and byte contents
were not tested here. Its code uses a fixed 24-slice geometry, separate plane
encoders, and 256-pixel local crops. It depends on ROI preparation and is not a
drop-in replacement for our variable-series loader.
[Repository](https://github.com/zqiuak/KAMRNet), [implementation](https://github.com/zqiuak/KAMRNet/blob/main/model_forallplane_MMAL.py).

KneePreM's matched classification comparisons changed only initialization:
fastMRI+ macro AUC **0.714→0.777**, and an eight-target APM task **0.741→0.787**,
versus random initialization; both reported p<0.001. Generic abdominal-CT
pretraining was worse. Pretraining took four days on an A100 80GB; classification
fine-tuning used batch eight and Adam at 5e-5. These are not comparisons against
our ImageNet-pretrained ConvNeXt. Parameter count and deployable checkpoint
availability were not verified. [KneePreM paper](https://arxiv.org/pdf/2609.31461).
Its linked [FuseMedML example](https://github.com/BiomedSciAI/fuse-med-ml/tree/master/fuse_examples/imaging/oai_example)
contains training infrastructure, but I did not verify a published KneePreM
weight download. Pretraining it from scratch is a lower-priority expenditure.

OrthoFoundation's current release reports pretraining on **1,251,655 images**
(MRI slices and radiographs, not that many independent patients), and fine-tuned
MRI AUCs of **0.9849 ACL, 0.9334 MCL, and 0.9338 cartilage**. Its meniscus number
is accuracy, not AUC. These are authors' results on their cohorts, not a promised
competition score. [Official release](https://github.com/ytrsk/OrthoFoundation).

I verified its published Git LFS pointer: **1,213,056,638 bytes**, declared SHA-256
`385a775822107b68eaa486336feb982e1ce7bd6d4e8c03ceb482a0bf546f2ff9`.
Implementation follow-up: the actual checkpoint was downloaded and its bytes
matched that hash. All 368 state entries loaded strictly into the pinned official
DINOv3-L architecture, with 303,154,176 encoder parameters and 1,024-dimensional
features. A CPU probe on a synthetic 5-series × 16-slice study passed finite-feature,
head-gradient, cached/image parity and reload checks. This establishes technical
compatibility, not diagnostic accuracy. The ordinary DINOv3-L download returned
HTTP 403 and the user has no local copy; the matched pretraining control remains
pending official checkpoint access.
[Checkpoint pointer](https://raw.githubusercontent.com/ytrsk/OrthoFoundation/main/OrthoFoudation-L.pth).

There are material reproducibility questions. The preprint describes DINOv3-L
in several places but DINOv2 initialization in Methods; the current README says
DINOv3. The released pretraining YAML has batch 24 and PolynomialLR, whereas the
README describes batch 32 and cosine scheduling. Resolve the actual checkpoint
configuration before assigning it the paper's results.
[Preprint](https://arxiv.org/pdf/2601.18250), [configuration](https://github.com/ytrsk/OrthoFoundation/blob/main/exp/pretrain/dinov3/full/config.yaml).

The released image loader converts individual images to RGB; its augmentation
uses 224-pixel global crops and ImageNet normalization. Our adjacent-slice RGB
triplets are a different input distribution. A fair initial transfer test should
repeat each grayscale slice across channels for both generic DINOv3-L and
OrthoFoundation, with identical normalization and aggregation. Use the official
DINOv3 feature API and verify compatible keys, tokens and shapes; the release's
custom forward path needs compatibility checks against the pinned DINOv3 version.
[Image loader](https://github.com/ytrsk/OrthoFoundation/blob/main/datasets/knee_multidata.py),
[transforms](https://github.com/ytrsk/OrthoFoundation/blob/main/datasets/augmentation.py),
[release wrapper](https://github.com/ytrsk/OrthoFoundation/blob/main/nets/dinov3_vit.py),
[official backbone](https://github.com/facebookresearch/dinov3/blob/main/dinov3/models/vision_transformer.py).

**Real external datasets can help, but their labels do not directly replace ours.**

| Resource | Useful content | Correct use and limitation |
| --- | --- | --- |
| [MRNet](https://stanfordmlgroup.github.io/competitions/mrnet/) | 1,370 exams; abnormality, ACL tear, any-meniscal-tear labels; patient-separated splits | Public train/tuning labels are report-derived; adjudicated test labels are hidden and the challenge is closed. No medial/lateral tear distinction or three OA compartments. Useful pretraining, not a fresh twelve-label image reference. |
| [fastMRI+](https://github.com/microsoft/fastmri-plus) | Expert pathology boxes linked to real knee MRI | Authors explicitly warn labels are not exhaustive or clinical ground truth. Unannotated findings cannot automatically become negatives. |
| [fastMRI+ label definitions](https://pmc.ncbi.nlm.nih.gov/articles/PMC8983757/) | 1,172 annotated knee volumes; cartilage defects, meniscus tears, ligament findings | Cartilage defect is not automatically compartment OA; a combined fracture/contusion/dislocation category cannot supervise those as separate diagnoses. Prefer localization/representation tasks with explicit mappings. |
| [SKM-TEA](https://github.com/StanfordMIMI/skm-tea/blob/main/DATASET.md) | qDESS images with tissue segmentations and pathology boxes | Useful for a cartilage/meniscus localizer. qDESS differs from our routine acquisitions; published full download is large, so verify a usable image-only subset first. |
| [OAI](https://nda.nih.gov/general-query.html?q=query%3Dfeatured-datasets%3AOsteoarthritis+Initiative+%28OAI%29) | MRI plus longitudinal clinical and MRI measurements | Match subject, knee, visit, sequence, and MRI annotation availability. X-ray KL grade must not be copied into PF/medial/lateral MRI OA labels. Access and packaging require preparation. |

The most defensible external-data use is transferring image representations or
localization, with unknown diagnostic targets masked. Any supervised mapping
must be documented, and negative labels must have an actual basis. Keep all
visits and both knees from a known patient in one partition. Account for images
seen in foundation-model pretraining: fastMRI cannot simply be declared an
unexposed external test for a backbone pretrained on fastMRI.

The saved competition rules permit qualifying public external data/pretrained
models. The current Kaggle rules page did not expose readable rule text through
the browser tool, so this review does not claim renewed eligibility verification
of individual assets. Confirm the actual downloaded asset's terms and applicable
host conditions during preparation. See [saved rules](rules_notes.md) and
[official competition rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules).

**Synthetic images have a narrower evidence base for our objective.** POWDR is
especially relevant: a conditioned 3D wavelet diffusion model with a ResUNet
denoiser preserves a real lesion region and generates surrounding anatomy. It
includes knee MRI examples, but the knee result is qualitative. Its measured
downstream improvement was brain-tumor segmentation Dice **0.6992→0.7137** with
fifty synthetic cases, not knee classification AUC. It also reports contrast
differences in generated knees. [POWDR](https://arxiv.org/html/2601.09044v1).

There is positive evidence elsewhere: a multiple-sclerosis MRI study reported
external AUC **0.836→0.933** with synthetic augmentation. This supports the
possibility, not transfer to knee MRI. [Original study](https://doi.org/10.1148/ryai.230514).
Recent generative knee-OA augmentation results concern **radiographs and KL
classification**, a different input and target from compartment-level MRI OA.
[MICCAI 2026 workshop paper](https://papers.miccai.org/miccai-2026-sat/RHEUM-AI_002.html).

My inference: generating more examples conditioned on our existing uncertain
labels does not supply independent evidence that the pathology is present.
Preserving one real meniscal lesion also does not justify inheriting all twelve
labels after generating the rest of the knee. A synthetic-data branch would need
verified lesion regions, patient-disjoint generation/training/evaluation, and an
equal-compute real-data/ordinary-augmentation control. Its endpoint must be
performance on real held-out images, not visual realism, FID, or agreement with
the teacher that produced the labels. Those prerequisites are currently less
ready than the two recommended directions.

**The concrete sequence is deliberately small and falsifiable.** The following
are proposed experiments, not launched jobs or guaranteed improvements.

1. **Freeze the reference and protect confirmation data.** Keep the existing
   labels, splits, control recipe and 0.917 ensemble artifacts. Outer0 is now
   development data because multiple candidates have been evaluated on it.
   Audit exposure before reserving another existing outer fold for confirmation;
   use a fresh nested selection split for each fit. Continue excluding the gold
   58 and protected review reserve from training. Do not tune on the public score.

2. **Prepare the localization test first.** Verify an external knee localizer on
   training-only images, including failure cases and different scanners. This is
   anatomical/pipeline QA, not a claim of diagnostic adjudication. If a usable
   localizer cannot be validated with available references, pause this branch
   before training a classifier; use the pretrained-encoder branch below.
   Generate crops from native pixels or a verified higher-resolution cache,
   preserving physical proportions and recording coordinates. Keep the original
   full field as well: a tight meniscus crop could remove patellofemoral disease,
   posterior cysts or peripheral injuries. Do not invent a universal crop size
   without inspecting training geometry.

3. **Run one isolated localization comparison when prepared.** Use the current
   ConvNeXt encoder, labels, sixteen slices and five series. A shared encoder
   sees two views per series. Control: two copies of the full-field view.
   Candidate: full field plus the anatomy crop. Both use the same two-view
   aggregation, optimizer-step count, initialization, seed and training budget.
   This isolates crop information from extra branches/compute. Compare both
   against the original one-view Tiny control as well; beating a weakened
   two-view control alone is insufficient. This is a new adaptation inspired by
   the crop evidence, not a claimed reproduction of CoPAS.

4. **Prepare OrthoFoundation as the second candidate, or the fallback if
   localization is not ready.** Retrieve and verify its checkpoint and compatible
   base model, pin implementation and preprocessing, and run a bounded
   feature-extraction/memory/reload check. Start with frozen embeddings from
   OrthoFoundation and ordinary DINOv3-L on exactly the same slices and use the
   same newly initialized study head. Their difference tests knee pretraining
   while holding capacity fixed. The existing Tiny remains the practical
   benchmark. A frozen probe is a cheap triage signal; failure is not proof that
   fine-tuning could never work. Only a convincing signal warrants a bounded
   low-learning-rate fine-tuning comparison. Choose its recipe on inner
   selection data and evaluate the frozen choice once on the outer split.

5. **Use a declared decision rule and confirmation.** For either main candidate,
   primary endpoint is full twelve-label macro AUC against the unchanged proxy;
   require at least +0.005 and a positive lower bound from a paired component
   bootstrap to proceed. Report every finding and the PF OA/lateral-meniscus
   average as prespecified secondary results, plus explicit-label and scanner
   diagnostics. Apply the existing −0.020 per-finding point-loss guard, while
   recognizing it is not simultaneous statistical noninferiority. A pass earns
   confirmation on the reserved split with two matched seeds, not deployment.
   Require improvement in both seed comparisons and on the confirmation split;
   report seed variability separately from patient/component resampling.
   Repeated outer0 screening does not become a final significance test.

6. **Make the strength of the conclusion match its reference.** Consistent
   improvement on the proxy establishes better report-label ranking. To claim
   better image-finding accuracy, obtain a separately adjudicated target-domain
   evaluation with reconciled definitions. Published external annotations can
   supply bounded tests of mapped targets, but cannot certify all twelve RSNA
   findings. Since no expert readers are currently available, the practical
   result may remain a well-controlled proxy gain. Do not substitute synthetic
   labels for that missing validation.

For budgeting, the saved Tiny control took about 2.72 hours on one T4. Two-view
encoding increases encoder work; that is not a measured runtime estimate for
the proposed model. OrthoFoundation is roughly 300M encoder parameters versus
our 38.7M complete model, but parameter ratios do not determine wall time or
VRAM. Measure both on a small representative batch before committing runs.
There is no justification yet for five-fold expansion, full-scale foundation
pretraining, or training a diffusion generator.

Success would be a frozen model with a reproducible gain, retained on a new
comparison and with its label-reference limits stated. A null result closes the
specific configuration without a rescue blend or a new post-hoc endpoint.

**Research deliverables and verification.**

- [Reproduction script](../tools/accuracy_research_snapshot.py): run
  `.venv/bin/python tools/accuracy_research_snapshot.py` from the project root
  to create a new snapshot when the output directory does not already exist.
  It intentionally refuses to overwrite the saved readout.
- [Evidence JSON](../outputs/accuracy_research_20260928/evidence.json): source
  hashes, numerical comparisons, historical interval provenance and limits.
- [All twelve findings](../outputs/accuracy_research_20260928/finding_diagnostics.csv).

This work added research and local numerical verification. It did not train a
model, change targets, create synthetic patient images, download large external
datasets, or submit to the competition. Papers and released code were inspected;
their reported accuracy was not independently reproduced here.
