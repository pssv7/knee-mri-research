# Worklog

Chronological session record. `PROJECT.md` holds current state; this holds
what happened, with numbers. Newest entry first.

---

## 2026-09-25 — Perturbation probe of the Tiny control

- User asked for controlled input variants to find what the model is sensitive
  to, without treating variants as truth. Agreed a declared grid before any run:
  identity, shift ±5/10/20, noise σ 2/5/10, three slice removals, left/right
  mirror with medial/lateral output swap, and `is_right` toggle. Model: the
  historical Tiny control on the 691 outer0 studies.
- Colab notebook 23 prepared; the user then chose a Kaggle run launched from here
  (a recorded departure from the Colab plan). The first push was rejected:
  Kaggle caps kernel source at **1 MB**, and the bundled kernel was 2.6 MB. The
  kernel now embeds only the three new files (52 KB) and takes the other 34
  frozen files from the attached control output by SHA-256. A test assembles
  the bundle from the real downloaded output byte-for-byte.
- Ran on T4, torch 2.10: 24.0 minutes for 10,365 passes. The unmodified pass
  reproduced the saved predictions with max |Δp| **0.0**. The locally
  recomputed summary matches in every flag, and numerically to 2×10⁻¹⁷.
- Results: brightness and the side covariate are inert; noise shifts scores
  directionally while keeping the ranking; slice density matters and edge
  slices do not. Under the flip-with-swap convention, Lateral Meniscus Spearman
  is 0.285. A prediction-only check shows why: the flipped raw Medial Meniscus
  output tracks the original medial meniscus (0.862), not the lateral one
  (0.285). The model reads anatomy, so swapping labels on a mirror is wrong for
  it. `FLIP_TTA` and `laterality_flip` are off everywhere; no code changed.
- Also found this session: Colab now runs Python 3.13. Lightning was missing on
  the Small run's VM, and `subprocess.run` output does not show in Colab cells.
  Notebook 23's setup now checks imports in a child process and captures output.
- Full suite: 1,341 passed; 3 pre-existing failures come from the uncommitted
  `require_complete_decode` edit to `src/data/cache.py`, which no longer matches
  the source embedded in notebooks 17/18.
- Details: `docs/perturbation_probe.md`; artifacts:
  `outputs/perturbation_probe_tiny_outer0_v1_results/`.

## 2026-09-03 — the labels are precise and incomplete; notebook 14 and its fixtures

**Outcome: measured against the radiologist image labels, the label matrix
misses 45.8% of true findings and false-alarms on 7.2% of true negatives. The
error is almost entirely misses, it is concentrated in the findings reports do
not narrate, and the pipeline's own sample weights do not predict it.**

Step 3 verified that the matrix reproduces. That is a statement about
consistency, not correctness -- a pipeline wrong in the same way every time
reproduces perfectly. This is the correctness half, and it is the last thing
the gold 58 can be asked.

### Where the labels lose

    weight  meaning                     slots  truly +   missed  false +
      1.00  both signals agree            279    52.3%    28.8%    16.5%
      0.70  only one signal spoke         176    33.5%    62.7%     7.7%
      0.40  the two signals disagree       31    32.3%    60.0%     9.5%
      0.25  neither signal spoke          210    11.9%   100.0%     0.0%
            ALL                           696    34.5%    45.8%     7.2%

**The weights do not track error.** They are assigned before anything sees
truth, so they either predict it or they do not: the full-weight bucket still
misses 28.8%. The 0.25 bucket's 100% is arithmetic rather than a defect --
silence writes a zero, so every true positive under it is missed by
construction. **25 of those 210 slots are truly positive.** That is the price
of "silence means absent, weighted low", now measured instead of assumed.

Per finding, worst first: Synovitis 74.1% missed, Fracture 66.7%, Effusion
60.0%, PF OA 52.4%, Baker's 50.0%, Lateral Meniscus 43.5% ... ACL 20.8%.
Synovitis being the worst is an independent corroboration of its descoping --
its labels are mostly zeros where the image says otherwise. ACL at the other
end is the expected shape: reports reliably mention a torn ACL.

**This is the ceiling of report-derived labels, not an extraction bug.** A
finding the radiologist saw and did not write down is absent from the text and
no better extractor recovers it. It also explains how macro AUC stays at 0.891
while nearly half the positives are missed: AUC is a ranking statistic and the
soft targets still order studies well.

Slot-level agreement lands at 81.5% (English) and 78.3% (non-English), which
sits on the ~82% report-vs-image agreement the public intel repos report for
this dataset -- weak independent corroboration that the pipeline is where the
literature expects it to be.

### Notebook 14, and what its fixtures caught

`notebooks/14_verify_labels.ipynb` runs the whole audit on Kaggle: provenance
(md5 against the repo copy, plus a check that no gold study is in the training
matrix), reproduction, properties, the bootstrap interval, language reach, and
the table above. CPU only, no GPU hours, internet off. **It has not been run
yet**, and Parts B/D/E/F need the four extractor dumps added to `knee-labels`
(1.5 MB total); without them A and C still run, which is enough to catch a
stale upload.

`tests/test_audit_labels.py` is why the audit is worth trusting: on the real
corpus every check passes, which proves nothing, so each check is exercised
twice -- once on data that should pass, once on data broken in the way that
check exists to catch. Fixtures are a six-language synthetic corpus with
generated extractor dumps. Both historical regressions are replayed as
fixtures, and the Bulgarian/Croatian stem misses are pinned so a repair fails
the test rather than silently invalidating the worklog.

Writing them found two defects in the audit itself, both invisible on real
data: `check_against_gold` raised `KeyError` when no finding had positives,
and `language_reach` took mined ids and ignored them, reloading the corpus
from `config.TRAIN_CSV` -- right answer on the real data by coincidence, wrong
population on anything else. 24 tests, suite at 621.

### Next

**Step 4: the five checkpoints.** Steps 1-3 verified the inputs; step 4 is the
first step where the artifact is the model. It cannot be answered the way the
others were -- training is not bit-reproducible and costs five GPU-hours -- so
the answerable questions are: whether the five `.pt` files carry the labels and
sides just verified or older copies; whether any fold leaked a study across its
own train/val boundary; which of the three saved weight sets per fold actually
ships; and the 995 unreferenced files in `checkpoints/`.

---

## 2026-09-02 (c) — step 3: the label matrix reproduces; the lexicon does not reach its own corpus

**Outcome: `train_labels.parquet` rebuilds from source with all 104,376 cells
identical, and every documented property holds. The exposure found is not a
divergence but a reach problem -- the lexicon, which sets the sample weight of
every slot, is 3.2x less informative in Bulgarian than in English at matched
report length, and the cause is nameable: 15 of the 16 Medial Meniscus anatomy
stems are multi-word and require adjacency.**

Step 1 verified the image tributary against its artifact. This is the same
check for the other input to all five folds, and it is cheaper: nothing here
decodes a pixel, so the reproduction covers 100% of the corpus in ~10 seconds
rather than the 2% sample the cache check could afford.
`tools/audit_labels.py` is the script; it exits non-zero on a failure.

### A. The artifact reproduces, unlike notebook 04's

    [PASS] 104,376 cells identical across 4,349 studies and 24 columns
    [PASS] extreme targets  0.7475   documented 0.75
    [PASS] mean weight      0.6252   documented 0.625
    [PASS] at WEIGHT_SILENT 0.3452   documented 0.345

Step 2 found the side classifier's documented 98.97% belonged to a run that no
longer existed. The label pipeline is the opposite case: every number the docs
quote is the artifact's own, and the code that produced it still produces it.
This is now the step-1 guarantee for the label side.

### B. The lexicon's reach is a function of language, and the weights read it

Both signals' informativeness -- did the lexicon say anything, did the LLM say
anything -- is what sets each slot's weight. The LLM's reach is roughly flat;
the lexicon's is not:

    lang     n     lexicon %   llm %   median chars
    en    1786          52.0    69.5           1253
    es     521          25.6    49.4            308
    tr     426          27.6    57.9            656
    hr     357          22.4    63.7           1164
    el     318          18.9    53.4            853
    bg     217          16.3    59.7           1001

English/Bulgarian is 3.2x on the lexicon and 1.16x on the LLM, so this is the
lexicon, not the reports and not the extractor.

**Length is a real confound and does not carry the result.** Spanish reports
are 308 characters against English's 1253, and a shorter report genuinely
mentions fewer findings. Inside corpus-wide length deciles the gap survives
intact -- at decile 5, English 43.3% against Bulgarian 15.4%, Greek 20.4%,
Croatian 23.1%.

### C. The cause, not just the gap

    bg: 99.1% of reports contain the bare stem "мениск"
        0.0% match the Medial Meniscus stem list
    hr: 97.5% contain "menisk"    0.8% match

The list holds `медиальн мениск` -- a Russian adjectival form -- and
`вътрешния менискус`. Bulgarian writes `медиалния менискус`, with the definite
article as a suffix. Croatian coordinates: `medijalni i lateralni menisk`,
which puts three words between the two the stem needs adjacent. 15 of the 16
Medial Meniscus stems are multi-word, and English is the language where an
adjective sits uninflected immediately before its noun.

Every finding does carry Greek and Cyrillic stems, so this is not an omission;
it is a matcher that assumes English word formation. `_norm` and
`_stem_pattern` are both correct -- the pattern anchors only at the start,
which is right for inflected languages -- and when the anatomy stem does match,
the rule calls the finding **100.0%** of the time. The loss is entirely
upstream of the rule logic.

### D. The two halves of the lexicon's influence, and only one has a knob

    LEXICON_WEIGHT   gold macro   mean weight
              0.00       0.8891        0.6252
              0.10       0.8911        0.6252     <- shipped
              0.30       0.8810        0.6252

The **target** half is worth 0.002 against a macro SE of 0.015-0.025 on the
gold 58: nothing. It is also where the bias lands -- on slots the LLM calls
positive at above 0.7, the lexicon backs 85.9% of English and 53.5% of
non-English, and the shipped target is 0.904 against 0.855 for equally
confident calls. Setting `LEXICON_WEIGHT = 0` removes that shrinkage for a
measured cost of nothing.

The **weight** half has no knob at all. Sample weights are byte-identical
across the entire sweep, because informativeness gates them independently of
this constant. English is 40.9% of studies and carries 46.0% of total training
mass (ratio 1.124); Spanish 0.845, Dutch 0.860, Greek 0.872.

### What was not done, and why

Nothing was changed. Both halves are unmeasured hypotheses about label
quality: the gold 58 cannot test them, because per-language cells are 1 to 8
studies -- and the gold's own language mix does track the corpus (English
48.3% against 41.0%), so step 2's population defect does **not** repeat here.
Rebuilding the label matrix costs five fold retrains, exactly like the side
parquet, and the same rule applies: measure first.

The cheap half is worth pre-registering. `LEXICON_WEIGHT = 0` is a one-line
change on a measured plateau that removes a language-correlated target bias,
and its cost is one rebuild of a parquet plus five retrains to actually use --
so it should be bundled with any future retrain rather than run for its own
sake. The stem fix is the larger question and is not costed here.

---

## 2026-09-02 (b) — step 2 opened: two defects in the side classifier

**Outcome: notebook 04's classifier is validated on a population it is barely
applied to, and its augmentation mislabels 59.5% of the corpus. Neither is a
demonstrated error in the shipped predictions; both are unmeasured.
`notebooks/13_side_validation.ipynb` is the measurement and has not been run.**

The classifier decides the side for 9,968 series (40.9%), and a wrong side
transposes Medial/Lateral Meniscus and Medial/Lateral OA rather than adding
noise.

### The documented accuracy is not the artifact's

Docs say 98.97% study accuracy. The log of the run that produced the shipped
`side_predictions.parquet` says **series 0.9733, study 0.9922** — the vote
lifts it 2.7 points. 98.97% belongs to some earlier run. Left uncorrected in
the docs until notebook 13 says what the honest number is.

### Defect 1 — the augmentation is plane-blind

`side_classifier.flip_input` exists to mirror along each series' own
medial-lateral axis. **Notebook 04 never calls it**, never joins plane
metadata at all, and hardcodes `x = x[::-1]` with `y = 1 - y`. For coronal and
axial series — 8,609 + 5,898 = **59.5% of the corpus** — the slice axis is not
medial-lateral, so the transform leaves the side unchanged while the label
flips. Confirmed directly: `flip_input` at axis 2 does not equal `x[::-1]`.

Measured effect, agreement with the study vote on model-predicted series:
sagittal 97.9%, axial 95.9%, **coronal 95.2%**. Real but modest — a reversed
stack is a distinguishable input the model can fit as a separate mode, and one
that never occurs at inference.

**Notebook 04's own markdown promised the diagnostic that would have caught
this** — "if errors concentrate in one acquisition plane" — and the cell under
it bins by `n_slices_raw` instead.

### Defect 2 — validated on one population, applied to another

Tag presence tracks the scanner almost perfectly:

    manufacturer               tagged   predicted   mean conf   agree
    Siemens Healthineers        5,304           5       0.843   100.0%
    SIEMENS                     3,168       1,292       0.873    96.6%
    Philips Medical Systems     1,715       1,899       0.885    96.9%
    GE MEDICAL SYSTEMS          1,041       3,873       0.897    96.8%
    Philips Healthcare              8         487       0.875    94.9%
    TOSHIBA                         0       1,015       0.886    94.8%
    CANON_MEC                       0         328       0.967    99.7%

**1,343 series across 226 studies come from manufacturers with zero labelled
training examples.** The 0.9922 is measured overwhelmingly on Siemens and
Philips; the work is done on GE, Toshiba and Canon. The model is *confident*
on the unseen vendors, which is what confidently wrong looks like if it is
wrong, and is no evidence either way. This is an unvalidated extrapolation,
not a demonstrated error.

Toshiba and Canon carry no labels and can never be scored directly. Notebook
13 holds out GE and Philips Medical Systems as proxies — large in the applied
population, small in the training set, so the holdout does not confound
"unseen scanner" with "less data".

### What is correct and was checked

The classifier does **not** override known tags: `source="tag"` rows keep
`laterality_resolved` and only the 9,968 unknowns get model predictions.

### Notebook 04 has diverged from the artifact it produced

Its split shuffled the `ArrowStringArray` that `.unique()` returns, which numpy
warns "may contain duplicates" — on that line a duplicate leaks a study into
both train and val and inflates the accuracy silently. Zero duplicates over
five trials on pandas 3.0.5, so **0.9922 is not thought to be inflated**, but
the notebook now casts to a numpy object array first and therefore no longer
reproduces the shipped run byte for byte. Nothing else was changed: the
plane-blind flip stays until notebook 13 measures whether fixing it is worth
anything.

### Order of operations

Run **13 before anything else, and 04 possibly never.** 13 writes nothing;
re-running 04 would produce a new `side_predictions.parquet` on an unmeasured
configuration, and the 0.906 folds were trained on the incumbent. If a v2 is
warranted, notebook 13's `WRITE_PREDICTIONS` produces it — trained plane-aware
on all tagged data — and swapping it in costs five fold retrains.

---

## 2026-09-02 — step 1 verified end to end: the cache is what the code says

**Outcome: the 29 GB that trained the 0.906 ensemble reproduces byte for byte
from raw DICOM. Step 1's foundation was assumed for four months and is now
measured. The mirroring concern raised at the start of the session was
investigated and closed at 24,371/24,371.**

Step 1's entire evidence base was 49 synthetic unit tests and no real series --
`tests/test_cache.py` has no `TestRealSeries` at all -- while every number this
project has produced was measured on its output.

### The Kaggle run (notebook 12, CPU kernel, no GPU hours)

    [PASS] A1-A4  3 shards, 24,371 series / 4,407 studies, no duplicates
    [PASS] A5     0 build errors          <- the manifest column never read
    [PASS] B1     no study split across shards
    [PASS] B2     md5(study_uid) % 3 reproduces at 100.00%
    [PASS] C1/C2  0 blank, 0 degenerate volumes across ALL 24,371
    [PASS] D1     500/500 series byte-identical when rebuilt from raw DICOM

D1 covers 2% of the corpus, which is enough: the failure it guards against --
shards and `src/data/` drifting apart -- is systematic, not per-series. A stale
cache fails on essentially every series, not on a rare one.

### Four corpus facts nobody had written down

    build errors                    0 of 24,371
    series lacking geometry         0 of 24,371
    acquired depth                  median 30, range 11-320
    shorter than 24 slices          4,247 (17.4%), median 4 duplicated of 24,
                                    287 volumes over one-third duplicate
    intensity (uint8)               mean 74.6 +/- 22.4, per-slice std 66.3

The padding number is the one worth carrying: a series acquired at 11 slices is
stored as 24, so more than half its depth is copies. Documented behaviour
working as designed, but the magnitude was unknown.

### The mirroring question, raised and closed the same day

Two implementations decide which axis a left knee reflects through:
`cache.mirror_axis` from `ImageOrientationPatient`, and
`dataset.PLANE_MIRROR_AXIS` from the CSV's `Anatomical_Plane` string. **Only
the second runs** -- `MIRROR_IN_CACHE` is False, so `canonicalize_side` is
never reached and the geometry implementation is dead code. A wrong answer
transposes Medial/Lateral Meniscus and Medial/Lateral OA rather than adding
noise.

`tools/audit_orientation.py` over the 819,078-header index:

    24,371/24,371 agree; 0 disagree
    obliquity   sagittal 0.0-40.9deg, coronal 50.3-90.0, axial 53.6-90.0
                against the 45deg boundary where the answer changes
    100% carry ImageOrientationPatient; 100% have a recognised plane
    the mirror axis is constant down every series

They can only diverge past 45 degrees of obliquity, i.e. only if the CSV's
plane label is flatly wrong. It never is. **Collapsing the two implementations
is now a tidy-up with no accuracy at stake, not a fix.** Real sagittal series
are prescribed obliquely -- the dev series sits at 9.1 degrees, the corpus
reaches 40.9 -- so the margin is real but never crossed.

The Kaggle audit reproduced the local one exactly: identical decision columns,
obliquity agreeing to 1.4e-14.

### An independent corroboration worth more than either run

The August cache build's own logs record 57+88+48 = **193** series recovered by
study propagation and 10+11+5 = **26** conflicting studies. The audit derived
both numbers today from the header index by a completely different route. Two
computations three months apart agreeing exactly is much stronger evidence that
the header pipeline is deterministic than either measurement alone.

### Also fixed

`docs/architecture.md` said 41.7% / 10,161 of 24,371 series have no side
*after* propagation. That is the **before** figure; after propagation it is
9,968 (40.9%), and the 193 difference is exactly what propagation recovers.

### What this does NOT close

**Laterality.** 9,968 series (40.9%) have no side from metadata and rest
entirely on the classifier at 98.97% study accuracy -- roughly **45 of 4,407
studies mirrored backwards**, four findings transposed on each, baked into
every measurement including the 0.906. Plus the 26 conflicting studies. That is
a measured limitation rather than a defect, it belongs to notebook 04, and it
is the largest known exposure left in the image tributary.

Tooling that stays: `tools/verify_cache.py`, `tools/audit_orientation.py`,
`notebooks/12_verify_cache.ipynb`. Re-run 12 after any change under
`src/data/`.

---

## 2026-09-01 (c) — the inference budget, measured: decode is half of it

The dry run, five folds attached, `AB_CHECK` off:

    tuned folds : 5
      decode  2.74s/study   (52% of the loop)
      forward 2.52s/study   over 5 model(s), 0.50s each
    projected for 1,500 studies: 2.20 h (efficiency penalty 0.245)

**Half the inference budget is DICOM reading, not the model.** Per fold the
forward pass is 0.50 s/study -- 0.21 h over the test set -- against 1.14 h of
decode that five folds share.

### My earlier arithmetic was wrong by a factor of five, and it mattered

On 2026-08-31 I estimated per-fold forward at 1.65 h by splitting a *projected*
8.6 h five-fold total (itself a pre-run guess from an old worklog note) against
a *projected* 2.00 h single fold. From that I concluded five folds at 24 slices
would take 12.6 h and could not fit the cap. The real per-fold cost is 0.21 h,
so:

    5 folds, 24 slices:  2.74 + 5 x 0.50 x 1.5 = 6.5 s/study = **2.70 h**

**It fits, with six hours to spare.** The claim that 24-slice ensembling is
dead at the 9 h cap -- carried in `docs/plan.md` §3.8 and repeated by me -- is
false, and it was never measured. The user pushed back on the arithmetic and
was right to.

### The efficiency entry should be ONE fold, not five

At 1 hour == 0.050 AUC, and scoring `-AUC/(maxAUC-Benchmark) + seconds/32400`
minimised:

    configuration                              AUC   hours  penalty    score
    5 folds, 16 slices (the 0.906 ensemble)  0.906    2.18    0.243   -1.771
    5 folds, 24 slices                       0.906    2.70    0.300   -1.713
    1 fold,  16 slices                       0.889    1.35    0.150   -1.826
    1 fold,  + slice-depth decode            0.889    1.01    0.112   -1.864

Four extra folds cost 0.83 h = **0.042 AUC-equivalent** and buy +0.017. Net
**-0.025**: the ensemble is the right AUC entry and the wrong efficiency entry,
which is exactly the question the exchange rate was derived to answer.

### And the blocked optimisation is now worth unblocking

Decode is 52% of a five-fold run and **84% of a single-fold one**, so on the
efficiency entry the dominant cost is reading images the model never looks at.
Cutting the decode by the ~30% the slice-depth change is worth improves the
score by **0.038 -- more than twice what the whole five-fold ensemble buys.**

What stands in the way is unchanged: the side classifier takes the slice axis
as input channels with 24 of them, so it needs retraining at 16 (notebook 04,
minutes), and that is a laterality behaviour change, which silently transposes
four of the twelve findings when wrong. Now it has a number attached to it,
which it did not before.

---

## 2026-09-01 (b) — which five checkpoints are the 0.906 ensemble

Confirmed by the user: the members are the five in
`models_train/fold{0..4}/convnext_fold*.pt`. Read from each fold's own
metrics.csv at its selected epoch:

    fold  sel epoch   val_macro   gold_macro   steps
      0        6        0.8814      0.8683      5220
      1        6        0.8726      0.8747      5220
      2        8        0.8675      0.8659      5220
      3        8        0.8736      0.8915      5232
      4        6        0.8697      0.8753      5232

All aux-free with `gated_pool` False, verified from the stored
hyper_parameters rather than from filenames. The 12-step difference between
folds 0-2 and folds 3-4 is one epoch's rounding.

**The bookkeeping was wrong about fold 3.** `ENSEMBLE_DONE` recorded the
retired 2026-08-26 run at val 0.8686 (LB 0.877 solo); the member actually used
is the 2026-08-31 run at val 0.8736 / gold 0.8915. That dict is described in
its own comment as "the only record a cold session has", and it was
misdescribing a member of the project's best result. Now corrected, with the
gold column added -- the previous entries carried val only, which is what let
the error stand.

**A fold-3 retrain the same day came in below the member on disk** -- val
0.8710 / gold 0.8760 against 0.8736 / 0.8915. It was run on an older copy of
notebook 08 (its banner still listed folds 0, 2 and 4 as outstanding), though
at the reference config, so it is a valid member; it is simply worse. Not
swapped in, and the dict now says so.

Worth naming the near-miss: the retrain looked like an improvement because it
was being compared against the 0.8686 in the bookkeeping rather than against
the 0.8736 on disk. Swapping it in would have quietly degraded the ensemble
that holds the project's best score, and nothing would have raised.

---

## 2026-09-01 — the Efficiency Prize, and a submission-wide failure caught in time

The AUC track is close to finished -- 0.906 from the five-fold ensemble, nine
consecutive single-variable nulls after it, and a gold set whose per-finding
noise (+/-0.10 paired) is wider than anything left to chase. The Efficiency
Prize is a separate $18,000 pool that has never been worked on, and its levers
are mechanical rather than statistical.

### The exchange rate the whole track turns on

    Efficiency = AUC / (Benchmark - maxAUC) + RuntimeSeconds / 32400   (minimised)

At Benchmark 0.5 and maxAUC 0.95 the first term is -2.22 x AUC, so:

    **one hour of runtime == 0.050 AUC**

For scale: the entire spread between this project's best and worst
configurations is about 0.06 AUC, and the five-fold ensemble's advantage over a
single fold is +0.017 -- **worth twenty minutes**. If four extra folds cost
more than that, the ensemble is the wrong efficiency entry while remaining the
right AUC entry. Nothing on this project has ever measured which it is.

### What reading the submission loop found

`build_volumes` decoded **every** series in a study, and only afterwards did
`pick_series` keep five and `KEEP` keep 16 of 24 slices. On a seven-series
study that is 7 x 24 = 168 slices decoded where 5 x 16 = 80 are used. 32.5% of
studies carry more than five series.

**Shipped: series are chosen before anything is read.** `pick_series` needs
only the count, and both sides order by SeriesInstanceUID, so exactly the
series the reference path keeps are the ones decoded. The end-to-end harness
reports a **maximum absolute prediction difference of 0.0000** -- strictly less
work, same answer.

### The half that is blocked, and why the harness earned its keep

Decoding each series to `TUNED_SLICES` instead of `CACHE_SLICES` would drop a
third of the remaining pixel work. `cache.build_series_volume` now takes a
`slices` argument that does it correctly, and "correctly" is doing real work
there: `submission_indices` composes the coarse and fine selections, because
picking 16 evenly across the raw stack **in one step lands on different files
for 41 of the 44 series lengths between 16 and 60**. That is the
`volumes[:MAX_SERIES]` head-truncation bug in a new costume -- shapes agree,
nothing raises, and the model is scored on anatomy it was not trained to read.

It is still switched off, because the **side classifier reads the slice axis as
input channels and its first convolution has 24 of them**. A 16-deep volume
raises, every study falls back to the training prior, and the submission scores
**AUC 0.5**. `tools/check_submission.py` caught it on the first run against
notebook 10.

Two notes on that, both worth more than the optimisation:

1. **The harness defaults to notebook 07.** The first run exercised none of
   this and reported "ran clean". `CHECK_NOTEBOOK=10_submit_ensemble.ipynb` is
   the invocation that tests the ensemble path, and it is not the default.
2. The DICOM laterality tag would sidestep the side classifier, but only
   **58.5% of studies** carry it on any series; 1,828 need the model.
   Unblocking means retraining the side classifier at 16 slices -- minutes in
   notebook 04 -- but that changes laterality behaviour, which silently
   transposes four of the twelve findings when wrong. Its own experiment.

### Stage E0 is now one run instead of two

The loop times decode and forward apart and prints both per study. The plan had
E0 inferring that split from two dry runs differenced against each other, each
extrapolated from three studies on a cold GPU. One commit of notebook 10 now
answers both questions directly: what a sixth fold costs, and whether the
decode is worth attacking further.

Set `AB_CHECK = False` before reading a runtime figure -- it scores every study
both ways, which is untimed but inflates the loop total and the projection. The
decode and forward lines are unaffected either way.

**Not yet measured.** The local numbers (decode 5%, forward 95%) come from a
CPU run on fake two-series studies and mean nothing for Kaggle.

---

## 2026-08-31 (i) — POS_WEIGHT measures null, and the reference constant was stale

Fold 1, `POS_WEIGHT_CAP = 4.0`, everything else at reference. Paired on the
gold 58 against the fold-1 predictions:

    finding            prev    wt   baseline   pos_weight   delta      paired 95% CI
    Lateral Meniscus  10.9%  2.85     0.821       0.707    -0.114   [-0.242,+0.004]
    Contusion         12.3%  2.68     0.843       0.922    +0.078   [-0.025,+0.206]
    MCL                5.1%  4.00     0.921       0.975    +0.054   [-0.006,+0.140]
    Medial Meniscus   32.5%  1.44     0.888       0.934    +0.046   [-0.023,+0.122]
    Lateral OA        11.3%  2.81     0.899       0.853    -0.046   [-0.146,+0.024]
    Fracture           2.5%  4.00     0.849       0.889    +0.040   [-0.067,+0.164]
    ACL               11.8%  2.73     0.952       0.918    -0.034   [-0.100,+0.021]
    macro                             0.876       0.875    -0.001

**Not one interval excludes zero**, and the pre-registered diagnostic came out
backwards. The prediction was "rare findings rise, common ones hold":

    rare (<12% prevalence, 7 findings)   mean delta  -0.019
    common (>= 20%, 3 findings)          mean delta  +0.007

Lateral Meniscus -- the finding this was aimed at, weight 2.85 -- is the single
largest move and it is *down* 0.114. That is not a weak mechanism, it is 58
studies with a per-finding SE of 0.045-0.08. **Null.**

### The powered instrument confirms it, and identifies the gold swings as noise

The same two models on 871 out-of-fold studies, paired:

    finding            prev    wt   baseline    posw    delta      paired 95% CI
    Lateral OA        11.3%  2.81     0.821    0.841   +0.020   [-0.012,+0.053]
    Fracture           2.5%  4.00     0.932    0.946   +0.014   [-0.013,+0.049]
    Baker's           10.3%  2.95     0.920    0.933   +0.013   [-0.009,+0.036]
    Effusion          14.2%  2.46     0.893    0.882   -0.011   [-0.028,+0.006]
    MCL                5.1%  4.00     0.904    0.889   -0.015   [-0.058,+0.023]
    ACL               11.8%  2.73     0.919    0.898   -0.022   [-0.052,+0.008]
    macro                             0.873    0.872   -0.000

    rare (<12% prevalence)   mean delta  +0.000
    common (>= 20%)          mean delta  -0.001

Flat everywhere. Every interval straddles zero and the largest move is Lateral
OA at +0.020, P(not better) = 0.10.

**And it settles what the gold table was showing.** Lateral Meniscus -0.114 on
gold is -0.008 here; Contusion +0.078 is +0.009; MCL +0.054 is -0.015. Three
swings of 0.05 to 0.11 on 58 studies, all of them within 0.015 of zero on 871.
That is what the gold set's per-finding noise looks like when something with
power is put beside it, and it is worth keeping as the calibration.

### The stale constant this run exposed

The notebook compared against `REFERENCE_GOLD = 0.861` and by that number the
run read **+0.014**. That constant dates from the 2026-08-26 era. The fold-1
checkpoint every current comparison is actually made against -- the 2026-08-30
run in `models_train/fold1` -- logged **gold_macro 0.8747** at its selected
epoch, and its saved predictions score macro 0.876.

So the constant understated the reference by 0.014, which is almost exactly the
"gain" this run appeared to make. Every comparison drawn against it since
2026-08-30 was flattered by the same amount. It is now 0.875, and the run reads
what it actually is: **-0.001, paired**.

Two other stale strings fixed in the same cell: the run banner still said
"CURRENT RUN: fold 4, the last of the five" above `FOLD = 1`, and the ensemble
paragraph still described the five-fold sweep as outstanding when it is
complete and scored LB 0.906. A cell whose job is telling a cold session what
run it is has to be right about that.

### Where this leaves the ledger

**`POS_WEIGHT_CAP` is back at None** (2026-08-31, caught by the user before it
cost anything). Every other lever was returned to reference the moment its
experiment finished; this one was left on. Nothing had been retrained since, so
the only cost was the risk: a fold trained now would have carried a null term
into the ensemble, and an average mixing weighted with unweighted members is a
blend of two models that the averaging hides. The notebook's `_expected` check
would have printed OFF-REFERENCE, so it was not silent -- but the default was
wrong, and the default is what gets run.

Eight single-variable changes, eight nulls, against a gold set whose macro SE
is 0.015-0.025. The pattern is not about the ideas. `pos_weight` was the
best-motivated of them -- implemented, never used, aimed at a weakness measured
on 4,349 studies rather than guessed -- and it still could not be read on 58.

The out-of-fold set has now resolved three questions in ten minutes each --
both synovitis levers and this one -- while the gold 58 gave an ambiguous or
misleading answer to all three. Any further per-finding work should be judged
there first and on gold second, or not run.

**And the wider read, which is not about pos_weight.** The reference
configuration sits where single-variable changes do not move it. The one thing
that has worked since 2026-08-25 is the five-fold ensemble, at +0.017, and it
worked by reducing variance rather than by changing the model. The leaderboard
cluster is 0.941 at 20th against this project's 0.906; that gap is not a
lambda, a weight or a loss term away. Anything aimed at it has to be
structurally different -- more input (24 slices was worth +0.002 on one fold
and was WRONGLY believed blocked at five folds by the 9 h cap -- measured
2026-09-01 (c) at 2.70 h, it fits), or an input the model
does not currently get. That is a decision about what to spend seven weeks on,
not a next run.

---

## 2026-08-31 (h) — the ensemble scored 0.906; next run is POS_WEIGHT, pre-registered

**LB 0.906 is the five-fold ensemble** (confirmed by the user; submission
55908661, 2026-08-31 06:05). Against the 16x5 single fold's 0.889 that is
+0.017, above the +0.005-0.010 the ensemble was expected to be worth, and it is
the project's best result. `PROJECT.md` had it flagged as unverified; that is
now settled. **Notebook 10 is done -- do not re-run it.**

### Where the headroom is, measured rather than assumed

Per finding over the 4,349 out-of-fold studies, which carries 100x the
positives of the gold 58:

    finding             OOF     pos     gold 58
    PF OA              0.773    925      0.826
    Lateral Meniscus   0.825    476      0.821
    Lateral OA         0.832    490      0.899
    Contusion          0.847    533      0.843
    Medial Meniscus    0.869   1412      0.888
    Synovitis          0.899    251      0.701

Synovitis reads 0.899 against the extractor and 0.701 against radiologists --
its problem was never the images, which is the descope in one line. **PF OA is
now the weakest thing in the model**, on 925 positives, and Lateral Meniscus is
second with both instruments agreeing.

### Two hypotheses killed for free before spending a session

**Every medial/lateral pair has the rarer lateral side worse by the same
margin** -- Meniscus 0.869/0.825, OA 0.883/0.832. Two candidate mechanisms,
both testable from files already on disk:

*Laterality.* Left knees are mirrored to a canonical side using a predicted
side, and a wrong mirror swaps medial with lateral. **Refuted:** the gap is the
same among the 2,544 studies whose side came from the DICOM tag for every
series (-0.039, -0.043), where no prediction error is possible. Only 26 studies
have series that disagree internally.

*Label quality.* The reports are German, Spanish and Dutch, and an extractor
confusing Innenmeniskus with Aussenmeniskus would poison the lateral labels.
**Refuted, and the replacement is more interesting.** On gold the lateral
labels are the *better* ones:

    finding             labels vs gold   model vs gold   mined positives
    Medial Meniscus         0.765            0.888            1412
    Lateral Meniscus        0.816            0.821             476

On medial the model **beats its own labels by 0.123**; on lateral it lands on
them exactly. Enough positives to imitate the extractor, not enough to
generalise past it.

### The lever that follows, and has never been used

`KneeLitModule` has accepted `pos_weight` since it was written and **every
fold's hparams.yaml says None**. Its own docstring already made the argument:
the metric is macro AUC, so a finding present in 2.5% of studies counts as much
as one present in 32.5%, and without this the loss is dominated by whatever is
prevalent. It is now wired into notebook 08, computed from the labels rather
than hard-coded so it cannot drift:

    POS_WEIGHT = min(sqrt(neg / pos), POS_WEIGHT_CAP)   cap 4.0

    Fracture 4.00 (2.5%)   MCL 4.00 (5.1%)   Synovitis 4.00 (5.8%)
    Baker's 2.95   L Meniscus 2.85   L OA 2.81   ACL 2.73   Contusion 2.68
    Effusion 2.46   PF OA 1.92   M OA 1.84   M Meniscus 1.44 (32.5%)

**The cap is load-bearing.** Raw neg/pos is 38.9 for Fracture and 18.7 for MCL;
at those weights the rare findings swamp the loss the way the stratified
ranking term did earlier today. The square root is the standard damping and 4.0
binds only the three rarest findings.

### Pre-registered, before the run

- Fold 1, 16 x 5, 12 epochs, everything else at reference. One variable.
- **Judged on macro**, not on one finding, because it touches all twelve. Pass
  if fold-1 gold macro clears 0.861 by more than the 0.02 noise floor.
- **The per-finding table is the diagnostic**, whatever macro does: the rare
  findings should rise and the common ones should not collapse. If Fracture,
  MCL and Lateral Meniscus move up while Medial Meniscus and Medial OA hold,
  the mechanism worked even if macro reads null.
- **Honest prior: modest.** Seven single-variable changes have measured null on
  this gold set, and this one is worth perhaps +0.01 macro. It is on the list
  because it is implemented, untested, and aimed at a weakness that was
  measured rather than guessed.
- `pos_weight` lives on the Lightning wrapper, not on `KneeNet`, so it does not
  reach the exported checkpoint and notebook 10 is untouched. Held by
  `tests/test_notebook.py::TestPosWeightReachesTheLossAndNotTheCheckpoint`.

---

## 2026-08-31 (g) — Stage 4: synovitis is descoped, on the literature rather than on exhaustion

The plan's §10 sent a Stage 1 conditional under 0.70 to Stage 3 -- crops on
Hoffa's fat pad and the suprapatellar pouch. A literature search took that
option off the table before it cost a session.

### The result that motivated Stage 3 is a different task

`plan.md` §7 cites "the best published result on this task (AUC 0.76, roughly
matching senior radiologists at 0.77)" from "explicit segmentation of the
synovial capsule and infrapatellar fat pad". That is Wang et al., *Academic
Radiology* 31(4):1518-1527 (2024): **376 patients who all have synovitis,
classified into three synovitis types.** Subtype classification, not detection.
The number does not transfer to "does this knee have synovitis", and it was the
only evidence for Stage 3 -- the plan itself files the stage as
"literature-motivated only" and "not measured on this data".

### Three more findings, all pointing the same way

**The reference standard is the weakest item in MOAKS.** Hoffa-synovitis
carries intra-rater kappa 0.42 and 55% percent agreement -- the only MOAKS
feature under the 75% threshold. A label a reader cannot reproduce against
themselves caps what any model can score against it. This project's own
figure sits beside it: 37% report coverage for this finding.

**The confusion is definitional.** The AJR review of synovial inflammation
imaging calls Hoffa-synovitis and effusion-synovitis "sensitive but
non-specific" surrogates with "limited ability to characterize the degree of
synovitis", and states that "absence of effusion does not rule out synovitis"
while "effusion may occur without synovitis". That is the coupling this project
spent three experiments on, written down before any model was trained.

**There is nothing public to build a crop on.** SKM-TEA segments bone,
cartilage and meniscus; OAI-ZIB bone and cartilage. Every infrapatellar fat pad
segmentation paper used private annotations. Stage 3 by segmentation is an
annotation project, not a crop.

### The crop was costed anyway, and declined rather than deemed impossible

Worth recording so nobody re-derives it. It needs no segmentation model at all:
`ImageOrientationPatient` gives the in-plane axes in patient coordinates and
`PixelSpacing` the scale, so a box can be cut in millimetres -- the same header
`dicom_io.mirror_axis` already reads for laterality. Restricted to the sequence
the finding is defined on, it is **4,667 sagittal fluid-sensitive
fat-suppressed series over 4,150 of 4,407 studies, 5.6 GB** at the existing
geometry, against the 49 GB full rebuild this project rejected in August. 56 of
the 58 gold studies are covered. Cropping the *existing* cache would add
nothing -- `cache.resize_volume` already squashes each slice to 224x224, so the
detail is spent; the crop has to happen before the resize, which is why it
needs a build at all.

It is declined because the evidence for it did not survive being read.

### And the arithmetic that was true the whole time

`plan.md` §3: a perfect synovitis fix moves macro by 0.022, inside one standard
error. Meanwhile the leaderboard has moved without it -- the submissions list
shows **0.906** on 2026-08-31 06:05 and 0.895 twice, against the 0.891 recorded
as "best" in `PROJECT.md`, the worklog and the project memory. Which run
produced 0.906 is **not verified**: `pssv27/submission-2`'s retrievable log is
a later single-fold run (`convnext_fold3.pt`), and the rule here is to read it
from the log rather than assume. Flagged in `PROJECT.md` rather than guessed.

### What was written down

`docs/synovitis.html` gains §08 -- the verdict, the four Stage 0 answers, the
three null interventions with their paired intervals, the ceiling arguments
with citations, and what would reopen it. `plan.md` is marked RESOLVED with a
stage-by-stage outcome table so it stops reading as live work. Notebook 08 is
back at the reference configuration (`RESIDUAL_BETA = None`,
`LAMBDA_STRATIFIED_RANK = 0.0`) so the next ensemble member trains
on-reference, and `ENSEMBLE_DONE` finally records fold 4.

**No code was removed.** The Stage 1 levers are tested, default to off and cost
nothing at those defaults; they are the record of what was tried. Two pieces
stay in active use regardless of the descope: `src/model/conditional.py` with
`tools/synovitis_gate.py`, whose paired-bootstrap machinery is not
synovitis-specific and is the right instrument for any per-finding claim on 58
studies; and the conditional AUC logged each epoch, which named the run-1
divergence three epochs in where macro read as an ordinary bad run.

**This is a negative result with a mechanism, not an abandonment.** What is
open is §6 -- the gold set -- and it is the binding constraint on every
synovitis experiment already run, not just the next one.

---

## 2026-08-31 (f) — Stage 1 run 2: the residual head is a null, and the plan's own rule fires

A clean run this time -- BCE falls monotonically 0.488 -> 0.234, nothing
diverges. `hparams.yaml` confirms the configuration: `residual_beta 0.64`,
`lambda_stratified_rank 0.0`, everything else at the reference.

At the selected epoch (7, best mined val macro 0.8734):

                          this run   baseline    delta
    gold macro              0.865      0.861     +0.004   guard passes
    Synovitis conditional   0.617      0.630     -0.013   target was 0.75
    Synovitis pooled        0.707      0.701     +0.006
    val conditional (mined) 0.803      0.801     +0.002

**Null on every instrument**, and null in the exact shape the pre-registration
named: pooled drifts up while conditional drifts down, which is a residual head
hard-coding the coupling it was given and `g` learning nothing beyond it.

Paired on the same 58 studies, same resamples, which is the test the standing
rules require:

    conditional  -0.024   95% CI [-0.130, +0.079]   P(not better) = 0.667
    pooled       +0.006   95% CI [-0.072, +0.086]   P(not better) = 0.458

The interval is centred just below zero and covers it comfortably. Nothing
happened, and the paired test is narrow enough to say so: +/-0.10 against the
+/-0.17 an unpaired reading carries here.

(The 0.617 in `metrics.csv` and the 0.606 above are the same model. The
training loop logs conditional AUC stratified on *predicted* effusion, since no
gold effusion label exists in the mined validation fold; the tool defaults to
the gold label. The baseline shows the same offset -- 0.630 gold, 0.617
predicted -- so the comparison is like for like either way.)

The error moved the wrong way inside the strata, too. The within-stratum share
of total error rose from 55.6% to 60.4%, and the Eff=1 vs Eff=1 cell -- the one
carrying most of the metric -- went from 38.8% misranked to 41.6%. Same
direction the frozen path took in Stage 0a: slightly more gated, not less.

### The thing this run actually did: Effusion 0.888 -> 0.986

The notebook printed "the Effusion head as a Synovitis classifier: 0.763"
against the baseline's 0.722, which reads as a head that has become far more
gated. It is not, and finding out why exposed a bug in `synovitis_gate.py`:
with `--baseline` given it computed that line from the *baseline's* effusion
column for every model. One model's gap read off another model's parent is
meaningless. It now prints per model with the parent's own AUC beside it, and
`TestTheSanityGapIsPerModel` holds it there.

With the right columns, per finding, paired on the same 58 studies:

    finding          baseline  residual   delta      paired 95% CI   P(not better)
    Effusion            0.888     0.986  +0.098   [+0.026, +0.186]      0.002
    Medial OA           0.955     0.978  +0.023   [+0.000, +0.062]      0.037
    Baker's             0.964     0.986  +0.022   [-0.002, +0.059]      0.056
    Synovitis           0.701     0.707  +0.006   [-0.072, +0.086]      0.458
    PF OA               0.826     0.773  -0.053   [-0.126, +0.016]      0.926
    Fracture            0.849     0.787  -0.061   [-0.142, +0.012]      0.949
    Lateral OA          0.899     0.820  -0.079   [-0.238, +0.056]      0.877
    macro               0.876     0.865  -0.010

So the gap widened because the **parent improved**, not because the child got
more gated -- a distinction the tool now prints rather than leaving to be
worked out. And the residual parameterisation, aimed squarely at synovitis,
did nothing for synovitis and produced the largest paired per-finding gain this
project has measured on **effusion**: +0.098, P = 0.002, which survives a
Bonferroni correction across all twelve findings (0.05/12 = 0.004).

A mechanism that fits: the parent is detached, so effusion's own training
signal is untouched, but the synovitis head no longer needs the trunk to encode
effusion on its behalf. Freed of representing it twice, effusion sharpens.

**It did not replicate, and the check took ten minutes.** The same two models
on this fold's 871 out-of-fold studies, paired, against mined labels:

    finding          baseline  residual   delta      paired 95% CI   P(not better)  pos
    Effusion            0.893     0.885  -0.008   [-0.027, +0.012]      0.788       124
    MCL                 0.904     0.872  -0.032   [-0.060, -0.007]      0.994        45
    Synovitis           0.899     0.903  +0.004   [-0.018, +0.028]      0.354        46
    macro               0.873     0.873  +0.001

**Effusion moves -0.008 here against +0.098 on gold.** With 124 positives
instead of ~30 and an interval a third the width, this is the better
instrument for that finding -- effusion is one of the reliably mined labels
(12.0% report silence), so agreement with the extractor is close to agreement
with truth. The gold gain was 58-study noise. It was flagged as needing
replication before the check was run, and it failed it.

MCL at -0.032 is the only interval clear of zero, and with twelve findings
tested one crossing is what chance produces. Read it as nothing until it
replicates too.

Synovitis on the same set: pooled +0.004, conditional 0.803 against 0.809, a
paired +0.011 with 95% CI [-0.034, +0.059]. Flat, on the instrument with 871
studies rather than 58.

So the residual parameterisation changed **nothing, anywhere**. Not synovitis,
which it was aimed at; not effusion, which briefly looked like a windfall; not
macro. That is a cleaner result than the one it appeared to be: one lever, one
run, and no surviving effect to chase.

The mined validation fold is the reading that settles it. `val_Synovitis_
conditional` sits between 0.77 and 0.81 for **all twelve epochs** and lands at
0.803 against the out-of-fold baseline's 0.801. That instrument carries ~870
studies rather than 58 and can resolve about 0.02-0.03; it saw nothing at all.

### One pattern examined and dropped

Gold conditional averages 0.684 over epochs 0-6 and 0.618 over 7-11 -- a
-0.066 decline as the model converges, which would have been a real finding:
checkpoint selection on val macro actively selecting against synovitis, and
the best gold conditional of the run (0.767, epoch 3) sitting where nothing
can select it.

**It does not hold.** The same quantity on the mined validation fold moves
+0.015 over the same epochs. The higher-power instrument shows no decline, so
the gold trend is 58-study noise, and reading epoch 3's 0.767 as anything would
be the gold-selection leak this project banned. Recorded because the hypothesis
is attractive enough that someone will have it again.

### Where the plan sends us

§10's decision table, pre-registered and now firing:

    Stage 1 conditional AUC < 0.70  ->  Stage 3 before further objective work

0.617. Both Stage 1 levers are spent -- stratified ranking diverged on a
property of the loss, the residual head measured null -- and neither failure
was about tuning. Read together with Stage 0a, which showed the coupling
survives removing the capacity that was supposed to cause it, the picture is
consistent: **the coupling is not something the objective can be talked out of.**

That is what Stage 3 is for. Crops centred on Hoffa's fat pad and the
suprapatellar pouch change what the model can see rather than what it is asked
to rank, and the literature result the plan cites (AUC 0.76, roughly senior-
radiologist level) came from exactly that -- explicit segmentation of the
synovial capsule and infrapatellar fat pad. The plan files it as a bet, and it
still is. But it is now the only bet left that has not been measured, and the
two that have were measured for reasons that will not change with a different
lambda or a different beta.

Stage 2 remains the binding constraint on reading any of it.

---

## 2026-08-31 (e) — Stage 1 run 2 configured: the residual head, alone

    FOLD 1   RESIDUAL_BETA 0.64   everything else at the reference config
    LAMBDA_STRATIFIED_RANK 0.0   LAMBDA_RANK 0.0   LAMBDA_COUPLING 0.0
    AUX_MODE "fixed"   16 x 5   12 epochs   ConvNeXt-T

Reordered from the plan's own menu, with agreement: §5 lists stratified
ranking as the primary lever and the residual parameterisation as the
"alternative or complement", and the two swap places for two reasons. Stage 0a
showed the coupling is **structural** -- it survives removing the capacity that
was supposed to cause it -- which is exactly what a reparameterisation
addresses. And the ranking term has now demonstrated the instability §2.4
predicted, so even repaired it is the riskier of the two. The residual head
adds no loss term: beta is fixed, the parent is detached, nothing can run away.

### Why beta = 0.64, and how well it is known

    gold labels vs predicted effusion logit    0.636 +/- 0.245   [0.15, 1.12]
    mined labels, n=4349                       1.141 +/- 0.064
    the model's own self-coupling (OLS)        0.931

The gold fit is wide and it is still the right starting point. The mined
estimate is tight and wrong for this purpose -- the silence that makes mined
synovitis track effusion mentions *is* the coupling being escaped, so fitting
on it hard-codes the bias. And 0.93 is what the model already does, which the
residual head exists to replace rather than reproduce. 0.64 sits below both on
purpose: it leaves room for `g` to add signal instead of re-deriving effusion.

If this run is null, beta is the obvious knob -- but each setting costs 1.9 h,
so it is a hypothesis to argue for, not a sweep to launch.

### Checked before the run, not after

The residual coupling travels as two buffers, and a run writes three
checkpoints. `residual_pairs` is a long tensor, so `WeightAverage` routes it
through `_fixed` rather than averaging it, and `residual_beta` is float and
averages to itself because it never changes. Both come back through
`ema_state()` / `swa_state()` and reload into a bare `KneeNet` still coupled --
verified, and now held by
`tests/test_averaging.py::TestTheResidualCouplingSurvivesAveraging`. Had either
been dropped, the EMA and SWA files would have scored as uncoupled models with
nothing raising, which is the shape of failure the normalisation buffers
already cost this project once.

### Pre-registered

- **Primary:** conditional AUC on the gold 58, paired against the aux-free
  fold-1 baseline (0.630). Pass at **>= 0.75**.
- **The reading that matters more than the number.** This run hard-codes part
  of the coupling, so **pooled synovitis AUC may rise while conditional does
  not move at all** -- that is the residual head doing nothing except what
  beta was told to do, and it is a null, not a win. Judge on conditional, and
  watch whether the effusion-head-as-synovitis gap (currently +0.021 pooled)
  widens: if it does, `g` has learned nothing and the head is now more gated
  than before, by construction.
- **Guard:** macro more than 0.02 below the fold-1 reference kills it. After
  run 1 took macro to 0.706, check this line first.
- **Secondary, higher power:** out-of-fold conditional on mined labels against
  0.801, paired.

---

## 2026-08-31 (d) — Stage 1 run 1 diverged; the loss is fixed, not just the weight

    epoch  bce_loss  strat_rank  val_macro  gold_macro  gold_syn_conditional
      0      0.598      0.538      0.763      0.706         0.560
      1      0.688      0.559      0.756      0.724         0.612
      2      0.945      0.731      0.653      0.652         0.559
      3      1.376      0.983      0.659      0.646         0.5000
      6      3.441      2.590      0.594      0.604         0.5000
     11      5.987      6.166      0.538      0.550         0.5000

Killed by the pre-registered macro guard on the first line that mattered:
gold macro 0.706 against the fold-1 reference's 0.861, eight times the 0.02
threshold. But the guard undersells it -- **BCE rose by a factor of ten**.
This was not a model trading eleven findings for one, it was a training run
that diverged, and the conditional 0.590 it printed measures nothing.

The exact 0.5000 from epoch 3 onward is the tell: the synovitis head saturated,
every study scored identically, and the tie-averaged AUC has nowhere to land
but chance. The gold table agrees -- Effusion 0.934 and Synovitis 0.709 held
while ACL fell 0.947 -> 0.625, MCL 0.900 -> 0.565, Baker's 0.964 -> 0.723. The
two heads the objective touches survived; the trunk that served the other ten
did not.

### The mechanism, which is a property of the loss and not of the weight

A **satisfied** pair's RankNet loss decays to zero as its margin grows. A
**violated** pair's grows linearly, forever. So when the pair constraints
contradict each other, inflating the logit scale is a bargain: the majority's
loss falls faster than the minority's rises, and nothing stops it.

Within-stratum synovitis pairs contradict each other by construction. The
mined targets have a median of 0.010 and a mean of 0.087 -- 90% of studies sit
at zero -- so the pairs surviving `min_gap = 0.25` are essentially "the report
mentioned it" against "the report did not", and inside one effusion stratum
that comparison is mostly silence, not synovitis. This is §2.4 of the plan,
which named the risk exactly: the residual task is harder and the labels are
noisier relative to what is left of the signal.

### Three changes, because the weight alone was not the bug

1. **`pairwise_rank_loss` takes `max_margin`**, and `stratified_rank_loss`
   defaults it to 8.0 -- about a 3,000:1 probability ratio, past which a pair
   is not asking for an ordering. A contradictory set can no longer buy scale.
   Default `None` everywhere else, so every earlier run stays bit-identical.
   `tests/test_net.py::TestTheMarginClamp` holds it.
2. **Weight 0.1 -> 0.02.**
3. **`AUX_MODE = "cosine"`.** The Du et al. gate applies the auxiliary
   gradient only while it points downhill for the main task. It has been in
   this repo since 2026-08-26 and would have switched this term off around
   epoch 1. Leaving it at "fixed" while adding a far more lopsided auxiliary
   than the one it was validated against was the error, and it was mine.

### What the run is worth

Not a measurement of the objective -- it never got a fair trial. But it is a
measurement of the *instrument*: stratified pairwise ranking on mined synovitis
labels is unstable for a reason that follows from the label distribution, and
any future version of this idea has to answer it. The conditional-AUC logging
earned its place here: the exact 0.5000 named the failure three epochs in,
where macro alone would have looked like an ordinary bad run.

---

## 2026-08-31 (c) — Stage 1 configured and pre-registered

Notebook 08 is set for the run: **fold 1, 16 x 5, 12 epochs, ConvNeXt-T,
`LAMBDA_STRATIFIED_RANK = 0.1`, caliper 0.15, findings `("Synovitis",)`**,
residual coupling and the decorrelation penalty both off, everything else at
the reference config. One lever, not three.

**Fold 1 is not a free choice.** The 0.630 being beaten is a fold-1 model's
number and per-finding scores move more across splits than the treatment does.
The ensemble is complete (folds 0-4 all trained aux-free, fold 4 at val 0.8697,
read from its own metrics.csv), so nothing is waiting on this slot.

### Why 0.1 and not the 0.2 the ranking term was screened at

That 0.2 spread across all twelve findings. This loss touches one head, so the
same lambda puts roughly twelve times the pressure on it. The 0.2 configuration
is also the one that **collapsed** -- the rank term fell to near-zero within a
few epochs, having memorised its own queue, and gold went 0.861 -> 0.854. At
0.1 the term sits at about parity with the synovitis column's share of the BCE
(~1/12 of it). A judgment call; the plan does not specify a weight.

### Why caliper 0.15 on the probability, checked before spending the session

The setting that can silently make the run a no-op. Measured on the 4,349
out-of-fold effusion predictions, against the pairs that already survive the
`min_gap = 0.25` target filter (25.9% of all pairs):

    caliper   of gapped pairs kept    both-high    both-low
      0.05           8.4%               65.3%       27.5%
      0.15          23.7%               64.0%       23.7%
      0.30          46.3%               58.1%       13.4%
    percentile band 0.15
                    23.9%               80.6%       13.3%

At 0.15 the loss sees roughly 16-32 usable pairs per step against a 256-study
queue -- a real gradient, not a no-op. And the **stratum composition is why the
probability caliper wins over percentile matching**: the gold conditional AUC
is 76% high-effusion pairs and 24% low, and the probability caliper reproduces
that mix (64/24, the rest straddling) where percentile matching skews to
80.6/13.3 and would train a stratum the metric barely weighs.

### Pre-registered, before the run

- **Primary:** conditional AUC on the gold 58, paired against the aux-free
  fold-1 baseline (`failures/gold_predictions.parquet`, 0.630). **Pass at
  >= 0.75.** Notebook 08 now writes `gold_predictions_fold{FOLD}.parquet` and
  prints the reading and the paired difference itself.
- **Higher-power secondary:** out-of-fold conditional on mined labels against
  the current 0.801, paired. 484,654 within-stratum pairs there against 376 on
  gold, so this is what will actually resolve a move of 0.02.
- **Void condition, checked before the AUC:** if `stratified_rank_loss` falls
  to near-zero in the first few epochs the term memorised its queue, and the
  run says nothing about the objective either way.
- **Guard:** macro below the fold-1 reference by more than 0.02 kills it
  regardless of synovitis. One finding is 1/12 of the metric being scored.
- **Known in advance:** the gold interval is +/-0.17, so even a pass will not be
  significant on 58 studies, and two runs of one configuration differ by 0.011.
  Gold is the sanity check; the mined proxy is the discriminator. This is the
  cost of Stage 2 not having happened yet.

---

## 2026-08-31 (b) — Stage 0a ran: the frozen path is not better, it is more gated

The frozen aggregation stack, five folds, 24.8 min, pool grid 4 (6,528 dims
per slice projected to 512). Two runs of the identical configuration exist --
the interactive session and the committed one -- and both are recorded here
because their difference is itself a number worth having.

    gold 58                        conditional            pooled   eff-as-syn
    frozen, run A            0.609 [0.434, 0.778]  P=0.110  0.711     0.746
    frozen, run B (logged)   0.620 [0.443, 0.790]  P=0.089  0.723     0.750
    fine-tuned, fold 1       0.630 [0.457, 0.791]  P=0.071  0.701     0.722

**Run-to-run spread on conditional AUC is 0.011** between two runs of the same
config, against a sampling CI of +/-0.17. Stage 1's target (0.630 -> 0.75) is
eleven times the jitter, so it is readable; anything under ~0.02 conditional is
not, even paired.

### §2.2 is closed: do not ship the frozen path for this finding

Pre-registered rule -- ship only if it wins on *both* the paired pooled
comparison and conditional. Paired on the same 58 studies, same resamples:

    conditional  -0.011   95% CI [-0.144, +0.124]   P(not better) = 0.560
    pooled       +0.022   95% CI [-0.075, +0.120]   P(not better) = 0.344

It loses the half that matters. And the plan's premise for the comparison does
not reproduce: the +0.045 pooled gap it was built on (0.746 vs 0.701) is
+0.010 to +0.022 here. Note where 0.746 actually turns up in this run -- it is
where the frozen path's **effusion** head lands as a synovitis classifier.

### The story about fine-tuning destroying synovitis signal is not supported

That reading has stood since 2026-08-29: "frozen beats tuned on Synovitis
across all three runs (0.746 vs 0.682, 0.694) -- that is what fitting label
noise looks like." On the metric that measures the finding, frozen does **not**
beat tuned. Its advantage is entirely in cross-stratum pairs:

    pos/neg cell        frozen misranked    fine-tuned misranked
    Eff=1 vs Eff=1           43.4%                38.8%
    Eff=1 vs Eff=0           12.4%                18.9%
    Eff=0 vs Eff=1           61.5%                55.4%
    Eff=0 vs Eff=0           21.1%                31.1%

The frozen path is *worse inside* the effusion strata and *better across* them.
It is more gated, not less -- which is exactly what §2.2 warned a pooled-only
comparison would hide, and it is why 61.6% of its error is within-stratum
against the fine-tuned model's 55.6%.

### The finding that changes what Stage 1 is for

**In both paths the effusion head outscores the synovitis head as a synovitis
classifier** -- frozen 0.750 against 0.723, fine-tuned 0.722 against 0.701. A
model with a fixed trunk and a small head, which cannot chase the mined labels
the way a fine-tune can, reproduces the gating anyway and slightly harder.

So the coupling is **not an overfitting artefact**. It survives removing the
capacity that was supposed to be causing it. What that does *not* settle is
whether it comes from the labels (mined synovitis is largely effusion talk) or
from the features (synovitis is genuinely hard to separate from fluid without
contrast, which 0c says this collection does not have). Either way, a better
trunk or a shorter schedule will not undo it: it has to be attacked by
construction, which is what Stage 1's residual parameterisation does.

### The gate, and the high-power second reading

Gold reads **AMBIGUOUS** for both paths -- two semi-independent architectures
landing at 0.609/0.620 and 0.630, each with P(<= 0.5) near 0.07-0.11. Neither
clears significance and both point the same way, which is more consistent with
weak-but-real signal than with one broken model, but n=58 cannot say so.

The free by-product of the run, out-of-fold over all 4,349 mined studies:

    frozen       conditional 0.820 [0.791, 0.848]   pooled 0.900
    fine-tuned   conditional 0.801 [0.773, 0.829]   pooled 0.899
    paired       +0.019 [-0.005, +0.044]  P(not better) = 0.062

Both **CLEAR**, on 484,654 within-stratum pairs. Against extractor agreement
rather than truth, and its positives are the report-mentioned severe cases --
but general-purpose features with a small head *do* order synovitis within an
effusion stratum. §2.4's worry, that the residual task is unlearnable because
the labels are 82.2% silent, is not what this shows; the silent studies are
what the proxy drops, so it stays open on exactly the studies that matter.

### Where that leaves the plan

Per §4's ambiguous row: **Stage 1 at reduced confidence, and prioritise Stage
2.** Nothing here sends us to Stage 4 -- the descope branch needed the frozen
conditional at chance, and it is not, in either run. Stage 3 is not indicated
first either: crops address within-stratum discrimination, which is the right
target, but the plan itself files it as a literature-motivated bet and the
gate did not fail.

Stage 2 is now the binding constraint and it is measured, not asserted: the
conditional CI is +/-0.17 on this set, so a Stage 1 run that genuinely reaches
0.75 will still be reported as ambiguous against 0.630. More gold labels are
what makes the next experiment readable.

---

## 2026-08-31 — the synovitis plan: metric, Stage 0, and the Stage 1 code

`plan.md` implemented. Every §1 figure re-derived from the local artifacts
first, and all of them reproduce exactly: pooled 0.701, conditional 0.630,
the pair decomposition 44.4 / 30.0 / 14.4 / 11.2, the conversions 0.755 and
0.867. `tests/test_conditional.py::TestTheReferenceNumbers` now holds them, so
a change to the metric fails a test rather than surfacing six weeks later.

### The metric (§3), and what it costs to ignore it

    pooled AUC, synovitis head            0.701
    pooled AUC, EFFUSION head             0.722   <- the same studies
    pooled AUC, gold effusion label       0.698
    conditional AUC (gold strata)         0.630   95% CI [0.457, 0.791]
    conditional AUC (predicted strata)    0.617

`src/model/conditional.py`, torch-free so the tools, the notebooks and the
training loop's validation hook all read the same definition. The plan's
"stratify on predicted effusion gives 0.617" is a **median split** of the
effusion score -- that is what reproduces it, and a median is what the module
defaults to, because a fixed 0.70 cut that balances one model's studies puts
80% of another's on one side.

`gold_Synovitis_conditional` and `gold_Synovitis_pooled` are now logged every
epoch beside `gold_macro`, on both validation loaders. Monitoring only.

### 0c — contrast: present in three studies, not misrouted

    7 series in 3 studies of 4,407 name contrast ("+C", all T1 TSE FS)
    routed as fluid-sensitive: 0
    none of the three is in the gold 58
    blind spot: 4,322 series (17.7%) carry no usable description

So the routing branch does not fire and the **non-contrast ceiling applies**:
effusion-synovitis is defined in the literature as fluid distension that
cannot be separated from synovitis without contrast, and this collection is
essentially without it. The plan's warning is honoured rather than waved at --
a description-free control (a repeated T1-like acquisition in one plane, which
is what a pre/post pair looks like without a typed string) finds 269 studies,
an upper bound on what the description scan could be missing, since a repeat
for motion looks identical. `ContrastBolusAgent` is now in `HEADER_FIELDS`; the
existing `index_train.parquet` predates it, so the next scan answers the
question directly instead of by inference.

### 0d — the premise was wrong, the test still ran

`worst05_317032662220` does not carry two series labelled `sag fluid fatsat R`.
It carries `sag t1`, `sag pd fs`, `sag t2 fs`, `cor t2 fs`, `ax t2 fs`. The
two identical panel captions on the failure sheet come from
`inspect.describe_series`, which prints the *metadata vector the model was
conditioned on* -- plane, fluid, fat-sat, side -- so two distinct sequences
sharing those flags print the same caption. Not a duplicated series.

The underlying question is still worth asking, and on 4,349 out-of-fold studies:

    corr(n fluid+fatsat series, synovitis logit)   +0.132
      partial, controlling predicted effusion      +0.108
    the mined label's own correlation              +0.151

The model tracks protocol *no harder than the labels do*, which is what
separates an aggregation artefact from the real fact that sicker knees get
scanned with more sequences. **No series-type-aware aggregation is indicated.**
The gold 58 gives +0.106 against the label's -0.073 and is not readable -- a
correlation on 58 studies carries SE 0.13, and the tool now refuses to read one
below 200 studies rather than printing an eye-catching verdict.

### 0a and 0b — blocked on a GPU session, and now one run each

No frozen DINOv2 predictions on the 58 and no saved silence-reweighting
predictions exist, here or in `models_train/`. No DINOv2 checkpoint either.

**Why 0a could not simply be run:** notebook 06 scored the gold 58 on the
frozen path and printed the per-finding AUCs *from predictions it then threw
away*. A printed macro cannot be compared to another run's predictions paired,
and paired is the only comparison that reads anything at n=58 -- the frozen
path's +0.045 on pooled is 0.66 of a standard deviation unpaired. Notebook 06
now writes `gold_predictions_frozen.parquet` in submission column order and
ends with the gate itself: conditional AUC, its CI, `P(<= 0.5)`, the reading
from `conditional.gate_reading`, and -- if the fine-tuned run's predictions are
attached -- the paired difference on both metrics. The cutoffs moved into
`src/model/conditional.py` so the notebook, the tool and the training loop read
one definition; `tests/test_notebook.py::TestTheFrozenPathEmitsTheGateArtifact`
holds the file in place.

So 0a is one frozen-path run, which is minutes on the aggregation stack rather
than hours, and 0b is one notebook-11 scoring of a `SILENCE_REWEIGHT = True`
checkpoint if one survives -- a 1.9 h retrain if it does not. For a fine-tuned
checkpoint the path is unchanged: point `CHECKPOINT` at it, take
`gold_predictions.parquet`, then:

    python tools/synovitis_gate.py --predictions frozen_gold.parquet \
        --baseline failures/gold_predictions.parquet

which prints both readings paired on the same 58 studies. Until then the gate
is undecided and the reference run reads **AMBIGUOUS** -- the expected outcome
at n=58, and the plan's own instruction for that branch is Stage 1 at reduced
confidence with Stage 2 prioritised.

### §6's interim proxy, which is not blocked

Out-of-fold predictions over all 4,349 mined studies, stratified on predicted
effusion:

    conditional 0.801   95% CI [0.773, 0.829]   P(<= 0.5) = 0.000
    pooled      0.899   (the effusion head alone scores 0.874)

Reads CLEAR, with 484,654 within-stratum pairs behind it instead of 376. What
it is worth: it measures agreement with the extractor, and its 251 positives
are the report-mentioned cases, which are the severe ones. What it is still
evidence for: the trunk *can* order synovitis within an effusion stratum when
the labels it is scored against are high-precision. §2.4's worry -- that the
residual task is unlearnable because the labels are 82.2% silent -- is not
supported by the one reading available, but the reading cannot rule it out
either, because the silent studies are exactly the ones this proxy drops.

### Stage 1 — the code, all of it off by default

A reference fold is unchanged; every knob below is inert at its default and
the ensemble-config check in notebook 08 flags any of them being set.

- **`net.stratified_rank_loss`** -- RankNet over pairs whose *predicted*
  effusion scores lie within a caliper (0.15), restricted to the named
  findings. Conditional AUC is within-stratum pair concordance, so this is the
  metric, differentiably, not an analogy to it. Both ranking losses share one
  cross-batch queue and it is pushed once per step; the queue now carries each
  study's effusion score, because deciding comparability needs the reference
  studies' effusion and they have left the batch.
- **`KneeNet.couple`** -- `syn_logit = beta * stopgrad(eff_logit) + g(features)`,
  built into the model so it **travels in the checkpoint** and a bare
  `KneeNet(backbone, dim)` at a scoring site adopts it. That is the
  input-normalisation lesson applied before it costs a run rather than after.
  The parent is detached: the synovitis objective must not reshape the one
  head on this pair that is reliable.
- **`net.coupling_penalty`** -- aims at an explicit target, and the module
  *raises* if a coupling weight is set without one. Independence is the wrong
  target: gold OR 6.09, p = 0.003, label correlation 0.403 against the model's
  0.65.

Calibrating beta (`conditional.calibrate_coupling`, IRLS, label on parent
logit):

    gold labels, n=58     0.64
    mined labels, n=4349  1.14   (82% silent, biased to severity)
    the model's own self-coupling, logit space   0.93

The last is what the residual head is meant to replace, so it is not the
target; 0.64 is the truth-supported figure and the one to start from, with the
caveat that it rests on 58 studies.

### Pre-registration, before any of this is run

Conditional AUC 0.630 -> **>= 0.75**, tested by paired bootstrap on the same
58 studies (`tools/synovitis_gate.py --baseline`). That is pooled ~0.755. A
pooled-only reading of the same run would show +0.054 and be dismissed as
noise, which is the whole reason the metric changed.

Two things fixed by decision rather than by the plan, both flagged:

1. The plan says "clearly > 0.5", "= 0.5" and "ambiguous" without cutoffs.
   Fixed in `gate_reading`: clear = bootstrap 2.5th percentile > 0.5 and
   P(<= 0.5) < 0.05; chance = point <= 0.55 and P(<= 0.5) >= 0.25; ambiguous
   otherwise. In code so they cannot be chosen after seeing a number.
2. §4's gate table sends a chance reading to Stage 3, §10's decision table
   sends it to Stage 4. The tool prints both and says the plan gives both.

### What shipped

    src/model/conditional.py        the metric, paired bootstrap, beta calibration
    src/model/net.py                stratified rank loss, coupling penalty,
                                    residual coupling in KneeNet
    src/model/lightning.py          the three knobs, shared queue, conditional
                                    AUC logged every epoch
    src/data/dicom_io.py            ContrastBolusAgent in the header scan
    tools/synovitis_gate.py         §3 report, 0a/0b, the pre-registered rule
    tools/series_audit.py           0c and 0d
    notebooks/11                    section 8: the same analysis where the
                                    cache and the index live
    notebooks/08                    LAMBDA_STRATIFIED_RANK, RESIDUAL_BETA,
                                    LAMBDA_COUPLING, all 0/None

62 new tests, 577 pass. Not measured: nothing above has been trained. The
metric change, 0c and 0d are measurements; Stage 1 is code awaiting a run.

---

## 2026-08-29 (d) — the Synovitis fix measures null; seven consecutive nulls

    Synovitis   0.682 -> 0.694   (+0.012)      gate was 0.78

Per-finding gold SE on 58 studies is 0.045-0.08, so this is noise. The
correlation between how much silent weight a finding lost and how its AUC
moved is **-0.08** -- nothing happened, to any finding, in either direction.
macro 0.877 -> 0.870, also noise.

**What this rules out, which is the useful part.** Removing 57% -> 19% of
Synovitis's gradient from silent studies changed nothing. So silent false
negatives dragging the ranking down was *not* the mechanism, and the diagnosis
behind the fix was wrong. What remains: the 18% of reports that *do* mention
synovitis are presumably the severe cases, and a model trained on those cannot
rank the mild ones a radiologist grades positive. **Loss weighting cannot fix
a biased positive sample.** Only image-derived labels for the silent studies
can -- the old plan's Phase E.

Note also that frozen DINOv2 still beats the fine-tuned model on Synovitis
across all three runs (0.746 vs 0.682, 0.694). Not actionable -- picking a
per-finding blend on gold is the fitted-on-gold overfit this project banned --
but it is consistent with fine-tuning destroying general features while
fitting noisy labels.

**Experimental-design error, mine.** The per-finding baseline came from the
16 x 6 run, so this comparison carries the series-cap change as well as the
reweighting. I should have had the 16 x 5 table before running. The reading is
unaffected -- everything is inside noise either way -- but the comparison was
not the clean one I set up.

### P2, third reading

    best mined-val epoch   0.870
    EMA                    0.870   (-0.000)
    SWA                    0.870   (-0.000)

Exactly flat. Three readings, three ties or losses. P2 is closed for good.

### The tally

    change                          val      gold        LB
    P0.1 normalisation travels        -         -         -   (ConvNeXt-invisible)
    P0.2 series linspace              -         -    bundled
    P0.3 checkpoint glob              -         -         -   (hygiene)
    P0.4 MAX_SERIES 6            +0.003    -0.005    -0.003   REVERTED
    P1   20 ep + warmup          +0.002    +0.021    -0.001   killed
    P2   EMA / SWA                    -    -0.003         -   killed
    silence reweighting          -0.003    -0.012         -   REVERTED

Seven changes, no improvement. Best LB remains **0.891** (24 x 5, 2026-08-25),
which predates every one of them.

**The binding constraint is the instrument, not the ideas.** Gold's macro SE is
0.015-0.025, per-finding 0.045-0.08, val has disagreed with the LB twice, and
every remaining item in the plan is worth <= 0.02 macro. Seven nulls in a row
is what running experiments below the noise floor looks like.

---

## 2026-08-29 (c) — the per-finding table, and the Synovitis fix

### Where the 0.053 actually is

Fold 1, 16 x 6, selected checkpoint, gold 58:

    Baker's          0.964    Effusion          0.919
    Medial OA        0.950    Medial Meniscus   0.897
    ACL              0.947    Contusion         0.852
    Lateral OA       0.925    Fracture          0.838
    MCL              0.900    PF OA             0.837
                             Lateral Meniscus  0.809
                             Synovitis         0.682   <- and falling
    macro 0.877

**The resolution hypothesis is refuted in its strong form and is withdrawn.**
It rested on small structures being catastrophic; Lateral Meniscus is 0.809
and Medial Meniscus 0.897. Fixing Lateral Meniscus perfectly is worth +0.012
macro. A 49 GB cache rebuild is not justified by that, and the recommendation
to do one was made before this table existed -- reasoning where a measurement
was available and cheap.

Fixing every finding under 0.87 to 0.95 gives macro 0.877 -> 0.938. **Synovitis
alone is 37% of that headroom.**

### P2 is killed

    best mined-val epoch   0.877
    EMA                    0.874   (-0.003)
    SWA                    0.875   (-0.002)

Gate was "EMA/SWA gold >= best-mined-val gold". Both below; the kill condition
("averaging underperforms argmax on the clean set") fired. Consistent with P1
-- no memorisation drift, so nothing for averaging to correct. **P0 measured
null-to-negative, P1 killed, P2 killed: the entire top of the plan is closed.**

### Synovitis: a label problem, quantified

    82.2% of studies are SILENT about Synovitis
    mined target, overall            0.087
    mined target, when silent        0.011
    TRUE prevalence on gold          0.466

So ~1,666 of the 3,576 silent studies are true positives being trained toward
zero, and because silence is so common here **they carry 57% of the finding's
gradient**. The model is being taught to rank real positives low. Synovitis is
the only finding where the fine-tuned model is WORSE than the retired frozen
path (0.682 vs 0.746) -- more capacity fits the noise harder.

**The fix: `fuse.rescale_silent_weights`.** The silent weight becomes
`WEIGHT_SILENT x (1 - silence_rate(finding))`, with the rate measured on the
4,349 mined studies.

    finding            silent%   weight        silent share of gradient
    Synovitis            82.2%   0.250 -> 0.044        57% -> 19%
    Lateral OA           54.8%   0.250 -> 0.113        28% -> 15%
    ACL                   9.2%   0.250 -> 0.227         3% ->  3%

**Why this is not the rule the module already rejected.** `fuse` removed
per-finding silence priors once before: 24 numbers fitted on 58 gold studies,
0.860 in-sample against 0.812 leave-one-out. This is one rule with no free
parameters, evaluated on the mined corpus, and it touches **weights only** --
fabricating P(positive | silent) as a *target* measured worse on every variant
tried, and that finding stands.

Near-neutral by construction for the 8 findings where reports are thorough:
silence already carries 3-5% of their gradient and still does.

`SILENCE_REWEIGHT = True` in notebook 08; the reference labels are one flag
away. No label regeneration and no re-upload -- the rescale runs at load time
against the existing `train_labels.parquet`.

**Untested.** The change cannot be scored locally: weights do not alter the
targets' ranking, so label gold AUC is identical either way. It needs one
fold-1 training run, read on Synovitis specifically.

---

## 2026-08-29 (b) — P0.4 measures null; the selector verdict replicates

Fold 1, 16 x 6, 20 epochs, A100, 5.0 min/epoch. The series cap was the only
change against the P1 run.

    at each run's selected checkpoint      P1 (16x5)   P0.4 (16x6)   delta
    val                                        0.874        0.877   +0.003
    gold                                       0.882        0.877   -0.005
    min/epoch                                    4.8          5.0     +4%

**Null.** The two signals disagree in sign and both deltas are inside their
noise. My +20% runtime estimate was wrong by a factor of five: only ~17% of
studies have a sixth series to gain, so the mean series/study rises 6.7% and
the wall clock with it, not 6/5.

**The plan's gate technically passes** -- "gold up or flat with no cost
blowup", and flat with +4% is exactly that. The kill needs "no gain AND
inference budget pressure", and there is no pressure. So by the letter of the
plan, `MAX_SERIES` stays at 6. Recorded because the reading is a null and the
gate is generous, not because the change earned anything.

**The series axis is now closed.** The plan's own accounting put the total
remaining headroom at +14%; 5 -> 6 was half of it and measured nothing.

### P1's kill replicates

Second independent run, same conclusion:

    run          pearson r   val peak      gold peak     selection cost
    P1 (16x5)        0.979   0.874 @ep10   0.882 @ep10           +0.000
    P0.4 (16x6)      0.974   0.877 @ep13   0.887 @ep6            +0.010

Two runs, no divergence, r ~ 0.975 both times. The biased-selector hypothesis
is closed on replication rather than on one run.

The 0.010 "selection cost" here is one epoch-6 gold excursion to 0.887 against
a plateau mean near 0.872 -- gold's sd over epochs 10-20 is 0.004 and its macro
SE is ~0.02, so a single reading 0.010 above the plateau is ordinary. **Do not
read a single epoch's gold.** The P1 run's 0.000 and this run's 0.010 are the
same measurement.

### Still unread, twice now

Notebook 08's scoring cell prints the P2 comparison -- selected vs EMA vs SWA
on the gold 58. `avg_fold1_ema.pt` and `avg_fold1_swa.pt` are on disk from both
runs. It is the last free measurement and the only thing standing between P2
and a verdict.

---

## 2026-08-29 — P1 killed; P0.4 configured; the plan's remaining gaps closed

### P1: no divergence. The hypothesis is dead.

Fold 1, ConvNeXt-T, 16 x 5, 20 epochs, fixed warmup, aux-free. 4.8 min/epoch on
one GPU, ~1.6 h.

    val   peak 0.874 at epoch 10, final 0.870
    gold  peak 0.882 at epoch 10, final 0.870
    pearson r over 20 epochs      0.979
    SELECTION COST                0.000

**Val and gold peaked at the same epoch.** The plan asserted mined-val
selection was costing ~0.015 per fold; measured on ConvNeXt it costs nothing,
and the two curves track at r = 0.979 including a shared dip at epoch 8. The
plan's kill criterion -- "no divergence -> selection rule is fine, revisit" --
has fired, and with it the premise of P2, which the plan headed "highest
expected gain **given P1**".

**Why the plan was wrong:** the divergence it diagnosed (val 0.860 -> 0.843
with gold flat) came from the **DINOv2** run. It generalised that curve to
ConvNeXt. ConvNeXt does not reproduce it, so the early-learning/memorisation
argument in section 2.1 was fitted to a trunk this project has retired.

**Do not read gold 0.882 as an improvement** over the 0.861 reference. Val moved
+0.002 at the same geometry while gold moved +0.021; val has ~870 studies and
gold has 58 with a macro SE of ~0.02. The large sample says "same model".

**Second negative result:** epochs 12-20 bought +0.003 gold and nothing on val.
Run length is not a lever; 12 epochs was already converged.

**Still unread:** notebook 08's scoring cell, which prints the P2 comparison
(selected vs EMA vs SWA on the gold 58). Free -- the checkpoints are on disk.

### The LB verdict: 0.888. Gold said +0.021; the leaderboard said -0.001.

Submitted fold 1 solo. **LB 0.888** against the 16 x 5 reference's 0.889.

    gold 0.861 -> 0.882   (+0.021)
    val  0.872 -> 0.874   (+0.002)
    LB   0.889 -> 0.888   (-0.001)

**Val was the better forecaster, by a lot.** It has ~870 studies; gold has 58
with a macro SE of 0.015-0.025, so +0.021 was one standard error and meant
nothing. This was called before the submission -- "treat this as a
reproduction of the reference, not an improvement" -- on exactly that
arithmetic, and it is the first time the plan's Hanley-McNeil number has been
used to predict rather than to explain.

**The gold-forecast rule has now failed twice in five trials, and the newest
miss is the largest:**

    change                        gold      LB     verdict
    8 -> 16 slices              +0.023  +0.024     tracked
    3 -> 5 series               +0.021  +0.020     tracked
    16 -> 24 slices             +0.004  +0.002     tracked
    report loss 0.5             +0.006  -0.009     MISSED
    20 ep + warmup + P0 fixes   +0.021  -0.001     MISSED

The three that tracked were capacity changes with real gold movement behind
them. Both misses had a gold delta inside one SE. **The operative rule: a gold
delta under ~0.03 forecasts nothing. Read val, or accept that you cannot
tell.**

**What this run actually bundled.** Against the 0.889 reference it changed
four things at once: 20 epochs instead of 12, absolute warmup, a fixed seed,
and P0.2's series-selection fix in the submission (which changes the images
scored for 32.5% of studies). The bundle is worth -0.001, and no single
component can be attributed. That is a fair read for P1's purpose -- the
divergence question is answered within-run -- but it is not a clean read of
any of the four, and P0.2 in particular went in as a correctness fix whose
effect is now known to be inside noise.

**Run length is closed as a lever.** Epochs 12-20 moved gold +0.003, val
nothing, and the LB -0.001. Twelve epochs was already converged. Future runs
can drop back to 12 for a 40% saving once P0.4 is read.

### P0.4 configured, and a documentation drift found

`MAX_SERIES` 5 -> 6 in notebook 08, `TUNED_MAX_SERIES` with it. The only change
in the next run; compare against the P1 numbers above.

While making it: **notebook 08's markdown still said "24 of 24 slices, 6
series, budget ~5.3 h" while its code ran 16 x 5.** The 2026-08-28 revert
updated the constants, but the script carrying one of three prose blocks failed
mid-way and only part was re-run. The run was correct and the documentation was
not, which is the harder failure to notice. Now guarded by
`test_the_geometry_prose_matches_the_geometry_constants`, which parses both and
compares.

### The plan's remaining implementable gaps, all default-off

- **Augmentation (section 3.4)** -- `src/model/augment.py`. The pipeline had one
  intensity scale. Now: photometric, series-shared noise, cutout with a fixed
  box through the stack, slice reversal, and a study-level laterality flip that
  swaps `LATERAL_LABEL_PAIRS` in the targets. `config.py` has carried that
  constant and its warning since long before anything used it.
- **Flip TTA (3.8)** and **member rank-averaging (3.9)** in notebook 10. The
  rank gap was subtler than "not implemented": `to_ranks` already rank-blended
  frozen vs tuned, but fold members were averaged as probabilities inside
  `tuned_predict` -- so it was never applied across the only ensemble that
  still exists.
- **Checkpoint guard extended.** It separated EMA/SWA from selected but missed
  a fold appearing under two names *within* a family -- `convnext_fold1.pt` and
  `lit_fold1.ckpt` are the same weights and both match `TUNED_PATTERNS`. That
  is the standing trap in PROJECT.md, and in a five-fold ensemble it gives one
  fold 2/6 of the vote. Now keyed on `(family, fold)`.

### A checker that was checking the wrong notebook

`tools/check_submission.py` defaults to `07_submit.ipynb`. Every "submission
notebook ran clean" this session was exercising notebook 07 -- including one
cited as covering notebook 10's new `pick_series`. Notebook 10 has now been run
end to end via `CHECK_NOTEBOOK=10_submit_ensemble.ipynb`, with the switches on
and off. This is the fourth entry in the "a check that passes may be looking at
the wrong thing" pattern the playbook names.

476 tests, from 445.

---

## 2026-08-28 — new plan adopted; P0 correctness fixes and selection-free training

No GPU run this session. `docs/plan.md` was replaced with a model-selection
and training-constraint plan, and the code changes its P0-P3 items call for
were made. **Nothing below has been measured** — the fold-1 20-epoch run that
reads P1 and P2 is the next thing to spend a session on.

### The plan that was replaced

Phases A-E had closed their own levers: A (label delivery) and B (capacity)
done, C (label quality) flat inside the gold set's resolution, C2 (auxiliary
losses) down 0.009 on the leaderboard. The new plan is ordered by expected
value rather than by phase, and its top items are bugs rather than ideas.

### DINOv2 closed

Fine-tuned DINOv2-S measured gold 0.820 against fold 1's 0.861 gate; at 24
slices it converged level with ConvNeXt while losing to its own frozen self on
Lateral OA (0.820 vs 0.899), Lateral Meniscus (0.621 vs 0.680) and PF OA
(0.736 vs 0.772). That is what the literature predicts, not a surprise:
DINOv2 transfers "on par or slightly better" than supervised CNNs on medical
classification (Bassi et al., arXiv:2312.02366), and even DINOv3 only
overtakes ImageNet initialisation at 512px, where ConvNeXt-B still beat
ViT-B/16 (arXiv:2510.07191). Both notebooks are back on ConvNeXt-T, and a
test asserts no ViT name can reappear in either.

### P0 — four correctness fixes, all silent failures

1. **Normalisation now travels in the checkpoint.** `input_mean`/`input_std`
   were `persistent=False`, so a strict load restored every weight and left
   normalisation off. `KneeNet._load_from_state_dict` now adopts a file's own
   statistics and still accepts files written before the change. This is the
   leading explanation for gold 0.864 logged in training against 0.823 from
   the same weights reloaded.
2. **The submission picks series by `np.linspace`**, matching
   `StudyIndex.series_for`. It took `volumes[:MAX_SERIES]`, a head truncation,
   which selected different series for the 32.5% of studies carrying more than
   five.
3. **Checkpoint recovery is no longer alphabetical.** `sorted(glob(...))[0]`
   put `lit_fold1-v1.ckpt` ahead of `lit_fold1.ckpt`.
4. **`MAX_SERIES` 5 -> 6** -- applied, then **reverted to 5 the next day** and
   deferred to its own run. It lifts mean series/study 6.7% and cuts
   truncation from 32.5% of studies to 16.7% (the whole remaining headroom on
   this axis is +14%), but it adds capacity, and P1 has to be read against
   fold 1's baseline at 16 x 5. One change at a time.

### P1/P2 — the selector is the suspect

Mined val rose to 0.860 at epoch 9 and fell to 0.843 by epoch 20 while gold
held flat at 0.864-0.869 across eleven epochs. That is the
early-learning-then-memorisation signature (Arpit ICML 2017, Liu ELR NeurIPS
2020), and because mined val shares the training labels' ~18% systematic
disagreement with image-derived truth, it makes mined val a **biased**
selector rather than a noisy one.

- Warmup is now an absolute step count rather than OneCycleLR's 0.3 fraction,
  so a 12- and a 20-epoch run put the learning-rate peak in the same place and
  are comparable. `schedule="cosine"` is available and removes the restretch
  entirely.
- `knee_lightning.WeightAverage` accumulates EMA (decay 0.999) and SWA (last
  25% of steps) and writes both from inside `fit` — read back off disk,
  because under `ddp_notebook` the object that accumulated them is in a forked
  child. A run now ships three candidates and notebook 08 scores all three
  against gold.
- Folds are seeded identically, which is what makes a cross-fold weight soup
  well-posed at all. `src/model/averaging.py` holds the soup maths; notebook
  10 has a `SOUP` switch, defaulted off because it changes submission
  semantics and every LB number on record came from probability averaging.

### Geometry: reverted to the reference, 2026-08-29

The P1 run was first configured at 24 x 6 -- 24 was left over from the DINOv2
run and kept on the grounds that it holds the best LB (0.891 vs 0.889). Wrong,
and not only on cost:

- **The plan says otherwise.** Section 3.1: "past ~16-24 slices, expect
  flattening ... treat slice count as roughly saturated and spend compute
  elsewhere". P1 item 5 says "rerun **reference** ConvNeXt", and the reference
  is 16 x 5.
- **24 slices confounds the experiment.** The two 24-slice runs behaved
  oppositely on gold. ConvNeXt at 24 x 5: val peaked at epoch 7 and fell to
  0.871 while **gold fell 0.865 -> 0.842** -- both curves down, which is
  overfitting. The DINOv2 24-slice run: val fell while **gold held flat at
  0.864-0.869 for eleven epochs** -- which is the biased-selector signature the
  plan's section 2.1 is built on. P1 asks ConvNeXt to distinguish those two,
  and at 24 slices ConvNeXt has already produced the wrong one. At 16 its late
  decay was mild (0.854 -> 0.849, epochs 9-12, with an aux loss on).

Both notebooks are back at 16 x 5, ~3 h for twenty epochs instead of ~5.3.

**Agreed with the user 2026-08-29:** hold `MAX_SERIES` at 5 for the P1 run,
then **follow the plan as written** from there -- P0.4 (5 -> 6) is the next
change, run alone so its gate can be read. The two departures taken silently
the day before (24 slices, and MAX_SERIES 6 inside the P1 run) both had to be
unwound; the rule now is to flag a departure and agree it, not to substitute
reasoning for the plan. `docs/plan.md` opens with this.

### Also landed

- Gated attention pooling (Ilse et al. eq. 9) as `gated_pool=True`. Note the
  plan's premise was wrong on this point: the aggregation has always used
  attention pooling, never mean pooling, so eq. 8 -> eq. 9 is the real move
  and the expected gain is a fraction of what the plan quoted.
- `label_smoothing`, `pos_weight` and `symmetric_beta` on `weighted_bce`, all
  inert by default. **MixUp was not implemented**: at `BATCH_STUDIES = 1` a
  within-batch permutation is the identity, and FiLM's per-series metadata
  cannot be interpolated across studies. Recorded in `docs/plan.md`.
- 428 tests, from 371.

### Open, deliberately

- **The metric is not confirmed from the official Evaluation tab.** Multiple
  public trackers report macro ROC-AUC; RSNA 2024 used mean weighted log loss.
  If it is log loss, calibration and per-finding weighting become first-order
  and plan sections 3.5 and 3.9 flip. Verify before the final submission.
- **Grouped-by-site folds are implemented but off.** `make_folds(groups=...)`
  and `scanner_key` (manufacturer + model + field strength, from notebook 01's
  index, since site is not given) are in, with `GROUPED_FOLDS` in notebook 08.
  Off by default: turning it on re-cuts every fold and invalidates fold 1's
  baseline, which P1 is being read against. Adopt it when the five-fold
  ensemble is rebuilt, and expect validation to *drop* — that drop is the
  ~0.053 of inflation coming out.
- **A larger clean validation set** is the ceiling on every decision above.

---

## 2026-08-26 (m) — fold 3 scores LB 0.877 solo: the first cross-split LB measurement

**Fold 3, submitted alone, scored public LB 0.877** (submission 55801109,
2026-08-26 18:27:48). The run was clean: one checkpoint, correct geometry.

**Read from the kernel log, not from recollection — and the recollection was
wrong.** The run was first reported as `tuned folds : 3` / `tuned:
convnext_fold3.pt`, which would have meant the standing glob trap had fired
and averaged three models. It had not. `kaggle kernels output
pssv27/submission-2` returns the actual log:

```
  tuned: lit_fold3.ckpt
frozen folds: 0 (pool grid 1)  [skipped, BLEND=1.0]
tuned folds : 1
```

`kernel-metadata.json` lists exactly two dataset sources (`knee-convnxt`,
`knee-labels`), and `knee-convnxt` holds exactly one file, `lit_fold3.ckpt`.
The notebook shipped `TUNED_SLICES = 16` / `TUNED_MAX_SERIES = 5`, matching
fold 3's training geometry on both axes. No stale checkpoint, no duplicate
load, no geometry mismatch. **0.877 is fold 3, honestly.**

*Method worth keeping:* the whole verification is local and takes seconds —
`kaggle kernels output <ref>` for the log, `kaggle kernels get <ref> -m` for
the attached sources, `kaggle datasets files <ref>` for what a dataset holds,
`kaggle competitions submissions -c <slug>` to tie a score to a run. Never
settle a "which checkpoint scored this?" question by recollection again.

**What the number means: the first measurement of how much the LB moves
between splits at a fixed configuration.**

| fold | val | gold | LB |
|---|---|---|---|
| 1 (16 × 5, aux-free) | 0.872 | 0.861 | **0.889** |
| 3 (16 × 5, aux-free) | 0.869 | 0.867 | **0.877** |

Same configuration, same geometry, same labels, different split: **0.012
apart**, mean 0.883. Two points, so treat 0.012 as an order of magnitude and
not a variance estimate — but it is the only such measurement the project
has, and it costs an LB entry to get another.

**Gold anti-predicted it.** Fold 3 has the *higher* gold (0.867 vs 0.861) and
the *lower* LB. This is not a new failure of gold — it is exactly why the
standing rule says never to read one fold's gold against another's — but it
is the first time the rule has been confirmed on the leaderboard rather than
asserted.

**The consequence for Phase D, which changes a number in the plan.**
Ensembling lifts above the *mean* member, not above the *best* member. With
members averaging ~0.883, five folds at the historical +0.005–0.010 land at
**~0.888–0.893** — straddling the current best of 0.891 rather than clearing
it. The plan's stated expectation ("+0.005–0.010 on 0.889") was anchored to
the best member and is too optimistic; it is corrected in `docs/plan.md`.

**This does not cancel the ensemble, it changes its justification.** With
equivalent models landing 0.012 apart on the public set, holding 0.891 as
"the best config" is partly a bet on which fold drew the friendlier split —
i.e. selection on the public LB, which is the one set that does not decide
the competition. The five-fold average is the variance-reduction play for
the **private** LB, and that is the reason to spend the remaining ~15.5 h,
not a public-score climb. A five-fold result of ~0.890 should be recorded as
the expected outcome, not read as a failure and not used to reopen anything.

**Unchanged by this:** the C2 verdict. That comparison (0.880 vs 0.889) held
the split fixed and varied only the loss, so cross-split spread does not
apply to it.

## 2026-08-26 (l) — fold 3 banked: the ensemble's first new member

**Fold 3, 16 × 5, aux-free: val 0.869, gold 0.867 at epoch 7. Verified from
the checkpoint, not from the log** — `lit_fold3.ckpt` stores
`lambda_report: 0.0`, `lambda_rank: 0.0`, carries zero non-`model.` keys, and
its `metrics.csv` has no `report_cosine`/`report_gate` columns. It was
trained on the reference configuration and is a valid ensemble member.
`total_steps` 5232 (436 × 12), matching fold 0's split size rather than
fold 1's 5220.

**Correction to how this run was first read.** When its numbers arrived they
were compared against the fold-1 references (gold 0.861, LB 0.889) and called
"a null" — that comparison was invalid, because fold 3 is a different split
with no baseline of its own. The conclusion not to submit it stood for a
better reason than the one given: **a single fold is not a submission at all,
it is one member of an average.** The notebook now prints exactly this
warning whenever `FOLD != 1`, added earlier the same day for this reason.
Judge ensemble members by the five-fold leaderboard score; do not read a
fold's gold against fold 1's.

**Ensemble status: 2 of 5 in hand.**

| fold | source | state |
|---|---|---|
| 0 | — | **next run** |
| 1 | `lit_fold1_old.ckpt` (val 0.8718) | have, needs `.pt` export |
| 2 | — | outstanding |
| 3 | `lit_fold3.ckpt` (val 0.8686) | **done, verified clean** |
| 4 | — | outstanding |

Three runs left at ~2.3 h each (~6.9 h), then the five-fold submission
(~8.6 h inference, inside the 9 h cap). Notebook 08 now ships `FOLD = 0`.

## 2026-08-26 (k) — gradient gating measured: no collapse, no gain. C2 closes for good

**`AUX_MODE = "cosine"`, `LAMBDA_REPORT = 1.0`, fold 1, 16 × 5: val 0.874,
gold 0.856 at the selected epoch.** Third auxiliary-loss configuration, third
failure to beat the aux-free reference.

| config (fold 1, 16 × 5) | val peak | gold at selected | LB |
|---|---|---|---|
| **reference, no aux** | 0.872 | **0.861** | **0.889** |
| report 0.2 + rank 0.2 | 0.868 | 0.854 | — |
| report 0.5, fixed λ | 0.881 | 0.867 | 0.880 |
| **report 1.0, cosine gate** | 0.874 | **0.856** | — |

Du et al.'s guarantee held exactly as advertised and was worth exactly what
it promised: **no collapse** — nothing like the screen's rank-loss
degeneracy — and **no gain**. Propositions 1–2 promise convergence to the
main task's critical points, not a better optimum, and that is what arrived.

**The run also produced the cleanest evidence yet that validation is
compromised for this loss family.** Within the run, val and gold move
*apart*:

| epoch | val | gold |
|---|---|---|
| 7 | 0.872 | **0.870** |
| 9 (selected) | **0.874** | 0.856 |

Val rose 0.002 while gold fell 0.014. Checkpoint selection is on val — which
is correct and must stay that way, since selecting on gold is the leak fixed
back in August — so **with a report-derived auxiliary loss, correct
selection systematically picks a worse model.** The best gold epoch was 7,
and val had no way to know. That is a structural objection to the whole
family here, independent of how the loss is weighted: the metric used to
choose the checkpoint is mined from the same reports the loss trains on.

**C2 is closed for good.** Both lambdas revert to 0.0 and `AUX_MODE` to
`"fixed"` in the generator, with the three results recorded at the switch so
nobody re-runs them. The code stays: the reduced-overfitting signature was
real, and if a future phase reaches for capacity again a small report lambda
remains a candidate regulariser — gated on the LB, never on val.

**Trap created by this run:** it wrote `/kaggle/working/convnext_fold1.pt`
(220 tensors, aux head correctly stripped) — but that file is the **cosine
aux model at gold 0.856**, not an ensemble member. The ensemble's fold-1
entry is `lit_fold1_old.ckpt` (16 × 5 reference, val 0.8718, LB 0.889). Two
different models now claim the name `convnext_fold1.pt`; notebook 10 globs
that pattern and would average the wrong one without complaint.

Notebook 08 is reset to the ensemble configuration: `FOLD = 3`,
`LAMBDA_REPORT = 0.0`, `AUX_MODE = "fixed"`. 342 tests.

## 2026-08-26 (j) — costing the fix: the 32B cannot make it, and the ceiling is unreachable

**Asked what fixing the extractor's errors would be worth, and whether the
32B could do the fixing. Answers: +0.004 and no.**
`tools/score_categorical_merge.py` computes both — gold-blind merge rules
that could ship, and gold-using ceilings that only say whether the work is
worth starting.

| configuration | cat-only | fused | delta | |
|---|---|---|---|---|
| 14B (shipped) | 0.838 | 0.891 | — | baseline |
| 32B | 0.824 | 0.885 | −0.006 | measured |
| merge: fill_silence (32B fills 14B's silences) | 0.842 | 0.892 | **+0.000** | noise |
| merge: mean_severity | 0.856 | 0.893 | +0.002 | noise |
| merge: both_agree (assert only when both do) | 0.817 | 0.881 | **−0.010** | hurts |
| CEILING: fix in-text silences | 0.855 | 0.895 | +0.004 | bound |
| CEILING: fix said-no | 0.850 | 0.902 | +0.011 | bound |
| CEILING: fix false positives | 0.916 | 0.934 | **+0.043** | bound |
| CEILING: fix all reachable | 0.942 | 0.950 | +0.059 | bound |
| CEILING: perfect extraction | 1.000 | 0.997 | +0.106 | bound |

**The 32B cannot fix the 14B, because it makes the same mistakes.** 78 of the
14B's 97 false positives are also the 32B's (**80% shared**); it calls YES on
only 7 of the 14B's 32 silent positives. Every gold-blind merge lands inside
the ±0.005 noise floor, and demanding agreement *hurts* by 0.010 — it removes
true positives along with false ones.

**The one big ceiling is false positives (+0.043), and it is not reachable.**
Three findings, on reading the actual cases:

- **It is concentrated nowhere.** Per finding, fixing *every* false positive
  perfectly is worth at most **+0.008** (Contusion, 18 cases), then Effusion
  +0.008, Synovitis +0.004, Lateral OA +0.004, MCL +0.003. The +0.043 exists
  only if all twelve are fixed at once, using gold to know which they are.
  Even an oracle-grade fix of the single best finding is below the 0.01
  measurability bar.
- **Most of them are not errors.** For Effusion the *same phrase appears on
  both sides of the gold*: "Geringer Gelenkerguss" (slight joint effusion)
  under a positive label, "Begleitend geringer Gelenkerguss" under a
  negative one; likewise "Leve derrame articular" and "Mild effusion" among
  the negatives. The text does not distinguish them, so no extractor can.
  These are the gold's own severity threshold, not extraction defects.
- **Where a threshold *is* legible, severity already handles it.** MCL's
  positives are complete ruptures, grade-2 sprains and loss of continuity;
  its false positives are "grade I sprain", "low-grade partial-thickness
  tear", periligamentous oedema. That pattern is learnable — and MCL already
  scores **0.959**, because the extractor grades those at severity 0.35 and
  AUC ranks them below the real tears. Nothing left to win.

Supporting measurement: among all YES calls, precision by stored severity
rises monotonically — minimal (≤0.35) 56% over 134 calls, higher (>0.35) 77%
over 166. Severity is genuinely informative, which is why fusion absorbs the
false positives; but the two classes overlap far too much for any threshold
to separate them (suppressing all minimal calls would destroy 75 true
positives to remove 59 false ones).

**Decision: not attempted.** The proposal was to fix the misses with the 32B;
that is worth +0.004 against a ±0.005 floor. The larger prize is a class that
is partly irreducible, partly already handled, and nowhere concentrated
enough to measure. Phase C stays closed. Note also that the previous attempt
to sharpen definitions from radiological reasoning cost 0.020 macro and 0.152
on MCL and was reverted — the precedent for editing definitions on inspection
is bad, and this inspection does not overturn it.

## 2026-08-26 (i) — reading the 14B's actual mistakes: most "errors" are reports that never said it

`tools/inspect_extraction_errors.py` prints every disagreement between the
14B categorical extractor and the gold 58 with the model's own evidence
span, the sentences the lexicon's anatomy patterns flag, and the 32B's call
on the same study. First pass over all twelve findings:

| finding | n+ | AUC | positives mentioned in report | unreachable | miss in-text | said no | false pos |
|---|---|---|---|---|---|---|---|
| Synovitis | 27 | 0.706 | **37%** | 11 | 0 | 0 | 5 |
| Fracture | 18 | 0.752 | 67% | 4 | 2 | 1 | 4 |
| Effusion | 35 | 0.757 | 80% | 0 | 0 | 0 | 14 |
| Contusion | 19 | 0.760 | 68% | 1 | 1 | 0 | 18 |
| Lateral Meniscus | 23 | 0.804 | 78% | 0 | 1 | 3 | 8 |
| PF OA | 21 | 0.826 | 67% | 4 | 2 | 0 | 5 |
| Lateral OA | 11 | 0.828 | 73% | 2 | 0 | 0 | 11 |
| Medial OA | 15 | 0.869 | 80% | 1 | 1 | 1 | 8 |
| Baker's | 12 | 0.911 | 75% | 1 | 0 | 0 | 5 |
| ACL | 24 | 0.935 | 88% | 0 | 0 | 0 | 4 |
| Medial Meniscus | 26 | 0.944 | 69% | 1 | 0 | 0 | 5 |
| MCL | 9 | 0.959 | 89% | 0 | 0 | 0 | 10 |
| **total** | | | **73%** | **25** | **7** | **5** | **97** |

**Of 37 missed positives, 25 (68%) were never written down.** The
radiologist marked the finding from the images and the report never
discusses the anatomy at all. **Report coverage predicts extraction AUC at
r = +0.65** (Spearman +0.62): findings whose positives are mentioned <72% of
the time average AUC 0.797, those above average 0.866. Synovitis is the
extreme and explains its long-standing weakness — **only 37% of synovitis
positives are mentioned in their report**, so its 0.706 is close to a
text ceiling, not a prompt failure. Caveat kept in the tool's docstring:
coverage is measured with the lexicon's anatomy patterns, so it is a lower
bound — unusual phrasing counts as unmentioned.

**The `said no` misses are mostly the report contradicting the gold**, not
misreading. Read one by one, three of the five cite explicit normality:
"Lateral menisküs normal." (Turkish), "Normaal voorkomen menisci." (Dutch),
"δεν σημειώνονται παθολογικά ευρήματα" (Greek) — each against a *positive*
gold label, each extracted at confidence 1.0. The extractor read the
sentence correctly; the text and the radiologist disagree.

**So the genuinely fixable misses number about seven, not thirty-seven** —
in-text silences, and the 32B caught several of them (e.g. "small horizontal
tear at the anterior horn of the lateral meniscus", which the 14B missed
entirely and the 32B called at 0.17).

**The dominant error is over-assertion: 97 false positives**, and Contusion
shows the mechanism cleanly. Its definition says bone marrow oedema
"excluding oedema that is purely degenerative subchondral change", yet the
extractor fires on: `subchondral bone edema is detected` (the excluded case
verbatim), `aumento de señal de la inserción distal del tendón cuadricipital`
(tendon, not bone), and `edema along the inferior margin of the popliteal
cyst` (soft tissue). Every one of them is scored at severity 0.35
("minimal"), which is why AUC absorbs them better than the count suggests —
the extractor hedges correctly even when it fires wrongly.

**What this does and does not license.** It is the clearest account yet of
*why* label quality plateaued: roughly a quarter of positive labels are not
recoverable from text at all, which caps any extractor and explains why four
levers of Phase C moved nothing. It does **not** license another prompt
round — (h) established that the 58-study gold set cannot resolve a change
below ~0.01, and tightening Contusion's oedema scope would move a 19-positive
finding by less than that. Recorded as understanding, not as a queued run.

## 2026-08-26 (h) — Phase C closes: the 32B fails, and a replication shows the sweep was measuring noise

**Two results arrived together, and the second one retracts a claim I made
in (d).**

### The 32B categorical upgrade fails (lever 3)

`llm_labels_gold__32b.parquet`, scored through the identical fusion against
the shipped 14B frame:

| | cat-only macro | fused macro |
|---|---|---|
| 14B (shipped) | **0.838** | **0.891** |
| 32B | 0.824 | 0.885 |

**The bigger model extracts categorically *worse*, not better** — and the
fused frame breaks the Synovitis floor (0.749 < 0.75). It wins ACL
(0.935 → 0.963), Lateral Meniscus, Medial OA and Lateral OA; it loses MCL
(0.959 → 0.902), Medial Meniscus (0.944 → 0.894), Effusion, Synovitis,
Contusion and Fracture. On the categorical ground — the six findings the
probability mode cannot reach — the fused deltas are Medial Meniscus −0.044,
MCL −0.011, PF OA −0.003 against ACL +0.016 and Baker's +0.009. It loses
where it had to win.

*Unresolved confound, recorded rather than hand-waved:* the shipped 14B
corpus was extracted 2026-08-22 and the prompt has been edited since (the
MCL/meniscus definitions were sharpened, measured worse, and reverted). If
the 14B ran under a different prompt revision, this compares model size and
prompt version together. It does not change the decision — swapping the 32B
frame in makes the labels worse either way — but it is not clean evidence
that 32B < 14B at extraction.

### The replication: the sweep's differences were sampling noise

A second, independent draw of all five probability configurations
(`data/meta/gold_sweep_latest/`) reshuffles the ordering completely:

| config | draw 1 | draw 2 | delta |
|---|---|---|---|
| baseline_n3 | 0.890 | 0.886 | −0.004 |
| baseline_n5 | 0.885 | 0.889 | +0.005 |
| correlates_n5 | 0.884 | 0.890 | +0.006 |
| granular_n5 | 0.888 | 0.883 | −0.005 |
| reason_first_n5 | 0.882 | 0.886 | +0.005 |

**Mean |delta| between *identical* configurations is 0.005; the entire
spread *between different* configurations within one draw is 0.007–0.008.**
The between-config signal is the same size as the within-config noise. Draw
1's best variant (`granular`) is draw 2's worst; draw 2's best
(`correlates`) was draw 1's worst.

**Retraction of (d).** That entry concluded "sampling noise on the fused
macro is small, so the deltas above are mostly real rather than resampling
luck", inferred from a single pair (shipped 0.891 vs a re-draw 0.890). One
pair agreeing to 0.001 was luck, not evidence. Per-run sd is ~0.004.

**And the jackknife was measuring the wrong noise source.** Ensembles, draw
1 → draw 2: `phrasings3` 0.893 (jackknife win rate **0.76**) → 0.892
(**0.00**); `all_n5` 0.890 (**0.00**) → 0.895 (**0.96**). The delete-8
jackknife resamples *studies* while holding one extraction fixed, so it
reports high confidence about a difference that evaporates when the
*extraction* is redrawn. Any future gold-set comparison of two extractions
must average over draws, or resample the extraction, or it is measuring
nothing.

### Verdict

**Phase C closes on all four levers.** The deeper finding is not that each
lever failed but that **the 58-study gold set cannot resolve a label change
smaller than ~0.01, while the gate asks for 0.005** — the instrument is less
precise than the decision it was built to make. Combined with the standing
history (labels +0.037 gold → −0.002 LB; C2 +0.006 gold → −0.009 LB), label
quality is done as a lever until either the gold set grows or a change is
large enough to clear the noise. Budget goes to Phase D.

*Tooling:* both scorers now take a sweep directory as `argv[1]`, so two
draws can be compared without moving files. `tools/check_notebook.py` runs
the notebook's logic cells in a temporary directory — it had been writing
`llm_labels_gold__32b.parquet` and friends into the repo root, and one such
stub was nearly scored as the real 32B result. Its tell: every confidence
exactly 1.0, which three identical stub samples produce and real 3-sample
voting cannot. 330 tests.

## 2026-08-26 (g) — notebook 08 armed for the aux experiment, and made unable to hide a collision

Notebook 08 now ships `AUX_MODE = "cosine"`, `LAMBDA_REPORT = 1.0` (guarded
by `if reports is not None`, so a missing `report_embeddings.parquet` still
disables it rather than lying in the hparams), `LAMBDA_RANK = 0.0`.

**`FOLD` moved 3 → 1 with them, deliberately.** The notebook serves two
mutually exclusive runs and the aux settings were about to be armed on the
fold reserved for the ensemble. An ensemble fold trained with an auxiliary
loss is worse than useless — it makes one member of the average a different
model, and averaging hides it — while the same run on any fold but 1 has no
baseline to be read against (fold 1: val 0.872 / gold 0.861 / LB 0.889).
Both configurations are now written at the top of the notebook:

| run | FOLD | LAMBDA_REPORT | AUX_MODE |
|---|---|---|---|
| ensemble fold | 0 / 2 / 3 / 4 | 0.0 | fixed |
| aux experiment | 1 | 1.0 | cosine |

A banner prints the run's identity before training: `AUX EXPERIMENT`,
`ENSEMBLE FOLD n`, or `MISCONFIGURED` when the lambdas are on with any fold
but 1. All three branches exercised against the generated notebook. 330 tests.

## 2026-08-26 (f) — gradient-similarity gating: the aux-loss weight is not a number

**Du et al., "Adapting Auxiliary Losses Using Gradient Similarity"
(arXiv:1812.02224), read and implemented. Its thesis is that C2 asked the
wrong question: there is no best fixed `LAMBDA_REPORT`, because a constant
weight is itself the failure mode.**

The paper's argument, and how exactly it matches what we measured:

- Section 2 names two problems with `L_main + λ·L_aux`: (i) λ is constant
  and cannot adapt when the auxiliary's usefulness changes during training,
  and (ii) task relatedness is ambiguous to quantify up front. C2 tuned λ
  (0.2 → 0.5) and read the improvement as "more of a good thing", which is
  precisely the reading the paper says the constant hides.
- Their fix (Algorithm 1, the *unweighted* variant they use throughout):
  apply the auxiliary gradient to the shared parameters **only while
  `cos(∇θL_main, ∇θL_aux) ≥ 0`** — only while it is a descent direction for
  the main task. Propositions 1–2 prove convergence to the main task's
  critical points. They keep only the *sign* of the cosine because minibatch
  estimates are too noisy to trust further.
- **Figure 3(c) is our result.** With unrelated tasks, the fixed-λ multi-task
  baseline sits *below* single-task for the whole run while the gated version
  blocks the transfer and recovers single-task performance. Our report loss
  at a fixed 0.5: val 0.881 / gold 0.867 but **LB 0.880 against the aux-free
  0.889**. The rank loss collapsing to 0.026 by memorising its own queue is
  the same disease with the weight stuck on.
- The honest limit, in their words: the method "is not a silver bullet that
  guarantees positive transfer, but it can drop the worst-case scenarios."
  It promises no-worse-than-baseline, not better.

**Implemented in `src/model/lightning.py`** as `aux_mode="cosine"`
(`aux_gate="binary"` = Algorithm 1, `"weighted"` = Algorithm 2's max(0,cos)).

- **The gate is nearly free here because of an architectural accident.**
  `forward_pooled` returns `head(pooled), pooled`, so *every* shared
  parameter is reached only through `pooled` — both shared gradients are
  `Jᵀa` and `Jᵀb` for the same Jacobian. Taking the cosine at that
  bottleneck asks the same question for two backward passes through the two
  small heads instead of two through ConvNeXt. Recorded as a proxy, not an
  identity: `Jᵀ` can rotate the pair, so this is sign-agreement in
  representation space rather than arithmetically the paper's θ-space cosine.
- Probe gradients bypass Lightning's GradScaler, so both losses are scaled by
  1024 before `autograd.grad` — under `16-mixed` an unscaled fp16 probe can
  flush to zero, which reads as cosine 0, reads as "aligned", and silently
  restores a fixed λ.
- The cosine is all-reduced across ranks (via `torch.distributed`, not
  `self.all_gather`, which needs an attached Trainer): DDP averages the two
  ranks' gradients, so a per-rank binary decision would quietly become a
  half-strength one whenever the ranks disagreed.
- `report_cosine` and `report_gate` are logged per epoch — the diagnostic is
  worth more than the gate, because it says whether C2's report loss was ever
  aligned at all.
- Six tests, both gate directions driven directly (aux = +main → cos +1,
  applied; aux = −main → cos −1, blocked to exactly zero). 330 tests.
  **`aux_mode` changes nothing while the lambdas are 0**, so the fold
  ensemble is untouched — that is a test, not a hope.

Mechanism check on the toy fixture (12 batches): an auxiliary target
unrelated to the labels gives mean cosine −0.001 with the gate firing 6/12 —
correctly coin-flip, blocking half the harmful updates — while a
label-bearing target gives +0.045 firing 8/12. Live and non-degenerate; the
real distribution needs a real run.

**The answer to "what is the best value": `AUX_MODE = "cosine"` with
`LAMBDA_REPORT = 1.0`** (which reproduces Algorithm 1 exactly — the paper
applies the auxiliary at full strength when it applies it at all), rank
still 0.0. One fold-1 run at 16 × 5, ~1.8 h, the probes costing under a
percent. Read `report_gate` first: mostly-1 means the fixed λ was not the
problem and C2's failure was the circular validation; oscillating or
mostly-0 means the fixed λ really was applying a harmful gradient and this
is the fix. **Gate on the LB regardless** — val labels are mined from the
same reports the loss embeds, so validation will flatter it again.

## 2026-08-26 (e) — two tracks armed: fold sweep on Kaggle, 32B categorical on Colab

**The decision from (d) resolved into a plan that does not force the
either/or: lever 3's measurement is cheap (gold checkpoint, minutes of
Colab) — only its payoff is expensive, so it runs in parallel with the
fold ensemble instead of competing with it for Kaggle hours.**

Track 1 — fold ensemble (Kaggle, user): notebook 08 now ships `FOLD = 3`
(user's pick for the next session); the config is the 16 × 5 reference with
both aux lambdas at 0.0 — the generated notebook needs no edits. Remaining
sessions sweep 0, 2, 4. Five folds fit the 9 h submission cap at 16 slices.
The full pre-flight checklist applies, plus the standing trap: `knee-convnxt`
holds the 0.880 report-loss ckpt and must end up holding exactly the fold
`.pt` exports (never `.pt` and `.ckpt` of the same fold — notebook 10 globs
both and double-counts).

Track 2 — 32B categorical gold measurement (Colab, user): notebook 03 gains
section 6b — `run_categorical` (the module's original categorical prompt and
parser, untouched and still tested) over the gold 58 with whatever model
section 1 picked, `CAT_MAX_NEW_TOKENS = 900` because the categorical JSON
carries severity/certainty/evidence. Writes `llm_labels_gold__32b.parquet`
in the shipped frame's exact schema. `tools/score_categorical_upgrade.py`
(local, seconds) swaps the 32B frame into the identical full fusion next to
the 14B and applies the standard gate; it prints the six categorical-ground
findings separately because that is where an upgrade must win or it wins
nowhere. Verified: the 14B frame passed through it reproduces fused 0.891
and cat-only 0.838 (the recorded 14B numbers), and an identity copy fails
the gate. A PASS earns the ~1 h categorical corpus re-run; a fail closes
Phase C with every extraction lever measured.

Guard-rail updated rather than deleted: the notebook test that forbade
`parse_response` in the run loop (it caught a real regression once) now
asserts the tighter invariant — the probability path parses probabilities
only, the categorical parser lives only in `run_categorical`. 325 tests.

## 2026-08-26 (d) — Phase C prompt levers all fail the gate

**All five dumps scored. Nothing passes; every variant fuses at or below the
shipped 0.891.** Prompt iteration (lever 1), more self-consistency samples
(lever 2) and prompt ensembling (lever 4) are closed.

| configuration | raw macro | **fused macro** | vs shipped |
|---|---|---|---|
| shipped labels | 0.860 | **0.891** | — |
| baseline@3 (re-draw) | 0.860 | 0.890 | −0.001 |
| baseline@5 | 0.853 | 0.885 | −0.006 |
| correlates@5 | 0.856 | 0.884 | −0.007 |
| granular@5 | 0.870 | 0.888 | −0.003 |
| reason_first@5 | **0.877** | **0.882** | −0.009 |

- **The raw screen inverted against the fused truth**, exactly as the
  leverage table predicted the day it was measured. `reason_first` led the
  raw table by +0.017 and finished *last* fused — and it is the only
  configuration to breach the 0.75 floor (Synovitis 0.732). Its raw wins
  were in MCL and Medial Meniscus, where the probability mode contributes
  +0.007 and +0.001 to fusion; its losses were in Synovitis, Lateral OA and
  Effusion, three of the six findings that carry the mode's whole value.
  **Standing rule: never rank a prompt candidate on raw macro.**
- The prediction recorded in (c) before the dumps arrived — reason_first at
  or below baseline, granular the best of the three, none clearing +0.005 —
  held on all three counts.
- `correlates` again moved its targets and paid for them elsewhere
  (fused PF OA 0.862 → 0.883, Synovitis 0.792 → 0.783, Lateral OA
  0.862 → 0.830).

**Lever 4 tested for free** (`tools/score_prompt_ensembles.py`, seconds
locally — the five frames are all an ensemble needs). Four combinations
fixed by structure before scoring, never selected on score:

| ensemble | fused macro | jackknife win rate |
|---|---|---|
| baseline_pair | 0.889 | 0.00 |
| **phrasings3** (correlates+granular+reason_first) | **0.893** | **0.76** |
| all_n5 | 0.890 | 0.00 |
| all5 | 0.891 | 0.01 |

- Ensembling works in the textbook way — three individually *worse* frames
  (0.884, 0.888, 0.882) average to 0.893, above any member and above
  shipped — and it survives the delete-8 jackknife 76% of 200 draws against
  shipped's 23%. It is nonetheless **+0.002, less than half the gate**, and
  it costs 3× the corpus extraction (~3 h of Colab) to produce. Declined on
  the trade, not on the sign.
- Adding baseline back to the ensemble erases the gain (all_n5 0.890,
  all5 0.891), which is the tell that all of this sits inside the noise the
  gate exists to ignore.

**Phase C's remaining lever is 3: the categorical mode is still the 14B.**
The leverage measurement points straight at it — the six findings the
probability mode cannot improve (Medial Meniscus +0.001, ACL +0.002, PF OA
+0.004, MCL +0.007, Lateral Meniscus +0.011, Baker's +0.013) are exactly the
ones the categorical extractor carries, and that extractor has never been
re-run on the 32B. Prompt work cannot reach that half of the label space;
only re-extraction can.

**But the premise deserves re-checking before the spend.** Phase A closed
with labels moving 0.854 → 0.891 on gold (+0.037) while the LB moved −0.002,
and C2 has since added a second case of a gold gain that the LB refused. The
open question is not "can labels get better" — it is whether label gold has
ever bought leaderboard. Recorded for the next decision rather than answered
here.

## 2026-08-26 (c) — the sweep's screen, and why raw macro misleads

**The gold sweep ran on Colab. Four of the five dumps were lost in transit
(see the defect below); the fused verdict landed the same day in (d).** What
the raw table and one local measurement established first:

Raw macro per configuration (58 gold studies, from the Colab table):

| config | raw macro |
|---|---|
| baseline@3 | 0.860 |
| baseline@5 | 0.853 |
| correlates@5 | 0.856 |
| granular@5 | 0.870 |
| reason_first@5 | **0.877** |

- **The pipeline reproduces.** baseline@3 is an independent re-draw of the
  shipped configuration at temperature 0.7, and it scored raw macro 0.860
  against the shipped 0.860, fused **0.890 against the shipped 0.891**. Two
  runs one thousandth apart: sampling noise on the fused macro is small, so
  the deltas above are mostly real rather than resampling luck. Individual
  findings still swing (fused Lateral OA 0.862 → 0.846).
- **More self-consistency samples did not help.** baseline@5 is 0.007 *below*
  baseline@3 — lever 2 of Phase C looks worthless, and the variants must be
  read against baseline@5, their own control, not against baseline@3.
- **Raw macro is the wrong weighting, and this is now measured.** How much
  fused gold AUC the probability mode adds over categorical-only, per
  finding: Fracture +0.111, Effusion +0.091, Synovitis +0.091, Contusion
  +0.049, Medial OA +0.043, Lateral OA +0.042 — and then Baker's +0.013,
  Lateral Meniscus +0.011, MCL +0.007, PF OA +0.004, ACL +0.002, Medial
  Meniscus +0.001. **Half the findings are already carried by the categorical
  extractor and the lexicon**, so a probability-side gain there buys nothing
  the labels do not already have.
- **That makes the raw leader suspect.** `reason_first`'s +0.024 over its
  control is concentrated in MCL (0.816 → 0.984) and Medial Meniscus
  (0.784 → 0.905) — the two findings with the *least* leverage (+0.007,
  +0.001) — while it loses on Effusion (−0.037), Lateral OA (−0.048) and
  Synovitis (−0.027), three of the six that matter. Prediction on the record
  before the dumps arrive: **reason_first fuses at or below baseline**, and
  `granular` is the better candidate (Synovitis +0.035, Lateral OA +0.019
  against its control, against Effusion −0.031). None obviously clears
  +0.005 fused.
- `correlates` did move its targets — Synovitis +0.020 and PF OA +0.050 over
  baseline@5 — but gave it back on Effusion (−0.041 vs baseline@3). Telling
  the model synovitis accompanies effusion appears to have flattened
  effusion's own discrimination. The idea works; the trade is not yet a win.

**Defect found and fixed:** section 6 called `files.download()` once per
dump, and the browser blocks a burst of downloads from one cell — four of
five were lost silently, since the failure is browser-side and nothing in
the notebook raises. The cell now writes one `gold_sweep.zip` and downloads
that. Verified end to end by `tools/check_notebook.py`, which executes the
real cell and produced a five-dump archive. Section 6's markdown now carries
the leverage table so the raw screen is not read as a verdict.

Resolved in (d) above: the dumps were recovered from the live session and
scored.

## 2026-08-26 (b) — Phase C harness: prompt-variant gold sweep, fused gate

**Levers 1–2 of extraction v3 are implemented and tested (325 tests); the
next step is one Colab session.** No numbers yet — this entry records the
instrument, not a result.

- `src/labels/llm_extract.py`: `PROBABILITY_PROMPT_VARIANTS` — the baseline
  plus three candidates, each one idea so a delta is attributable, built by
  anchored substitution with import-time asserts (rewording the baseline
  without updating a variant fails loudly). `correlates` gives the model
  named co-occurrence priors for silent findings (synovitis↔effusion/OA,
  cross-compartment OA, contusion↔acute injury) — aimed at Synovitis 0.792
  and Lateral/PF OA, the fused labels' weakest. `granular` pushes precise
  non-round integers against tie blocks. `reason_first` asks for two or
  three sentences of sorting prose before the JSON (the parser already
  finds a JSON object after prose; `MAX_NEW_TOKENS` 400 → 512 for the
  headroom).
- Notebook 03 section 6 is now a five-configuration sweep over the gold 58:
  baseline@3, baseline@5, and the three variants@5 — baseline@5 is the
  control that separates the sample-count lever from the wording levers.
  `run_extraction` takes `variant`/`n_samples`; one dump per configuration
  (`llm_probs_gold__<variant>_n<n>.parquet`), all downloaded.
- `tools/score_prompt_variants.py` (local, seconds): scores every dump in
  `data/meta/gold_sweep/` as the **full fusion** — score_average with the
  categorical frame, lexicon blend on top — next to the shipped reference,
  and applies the plan's gate: fused macro > shipped + 0.005, no finding
  below 0.75. Verified to reproduce the shipped 0.891 exactly, and to fail
  a byte-identical copy of the shipped dump (margin logic works). Raw AUC
  prints as context; the fused number gates, because fusion is what trains.
- Notebook 03 section 7 takes `PROMPT_VARIANT` / `N_SAMPLES_CORPUS`, re-runs
  the gold 58 at the chosen configuration so section 8's file pair is
  consistent, and is explicitly forbidden until the local gate passes.

Next (user, Colab): run notebook 03 through section 6 on an A100 (~minutes),
drop the dumps into `data/meta/gold_sweep/`, run
`python3 tools/score_prompt_variants.py`, and bring the table back. Corpus
(~1 h) only on a PASS; if nothing passes, lever 3 (32B categorical
re-extraction) is next in line.

## 2026-08-26 — report loss on the LB: **0.880**, and the gold forecast breaks

**The C2 winner submitted (fold 1, 16 × 5, `LAMBDA_REPORT = 0.5`): public LB
0.880, against 0.889 for the aux-free twin on the identical geometry, fold,
and notebook.** Gold said +0.006, val said +0.009, the forecast said ~0.895;
the LB said −0.009. First miss after three hits.

**It was not plumbing — each link was verified before believing the number:**

- `knee-convnxt` held exactly one file, `lit_fold1.ckpt`, byte-identical in
  size to the escalation checkpoint (uploaded 02:20, submitted 02:25, scored
  0.880). Replaced, not joined — no two-model averaging.
- The checkpoint's own stored hparams read `lambda_report: 0.5,
  lambda_rank: 0.0`, selected epoch 8 — the escalation run, not the screen.
  (The `metrics (1).csv` on the Desktop is the *screen's* log — its
  hparams.yaml says 0.2/0.2 — a red herring; the submitted weights are the
  right ones.)
- Export is clean: the ckpt's 4 `report_head.*` keys sit outside the
  `model.` prefix, `load_knee_state` drops them, and notebook 10's tuned
  path loads strict — a key mismatch would have crashed, and the run scored.

**Reading: the aux loss taught the model to predict reports, and the
report-derived evaluations rewarded it for that.** Val labels are mined from
the same reports the aux target embeds, so pulling the pooled embedding
toward the report embedding buys val accuracy partly by circularity, not
generalisation. Gold's +0.006 sits inside the 58-study ±0.02 floor. The
public LB is the only large, report-independent evaluation in the loop, and
it puts report 0.5 at −0.009 real AUC on this geometry. The forecast rule
survives with a rider: **gold forecasts the LB when the training change is
label- or capacity-shaped; a loss that points at the same text the val
labels came from can move gold and val without moving the LB.**

**Decisions:**

- C2 closes with both lambdas at 0.0 as the shipped default (generator and
  notebook 08 updated, 321 tests pass). The code stays — the
  reduced-overfitting signature was real (val flat through epoch 12), so the
  report loss may still earn a place if a later phase reaches for capacity,
  e.g. a smaller lambda at 24 slices. Not at 16 slices, where it is measured
  as a net loss.
- The plan's "24 slices + report loss" follow-up was gated on the LB
  confirming; the gate failed and the run is cancelled.
- Phase C (extraction v3, Colab) is now the lever, as the plan already
  ordered.
- **Trap left standing: `knee-convnxt` currently holds the 0.880
  checkpoint, and notebook 10 ships `TUNED_SLICES = 16`.** The LB-best
  artefacts are the 24-slice fold-1 ckpt (Desktop `lit_fold1_oldd.ckpt`,
  LB 0.891, needs `TUNED_SLICES = 24`) and the 16-slice reference (Desktop
  `lit_fold1_old.ckpt`, LB 0.889, the efficiency-track candidate). Restore
  the matching pair before any future best-config submission.

## 2026-08-25 (f) — report loss alone: val 0.881, gold 0.867. C2 has a winner

**`LAMBDA_REPORT = 0.5`, `LAMBDA_RANK = 0.0`, fold 1, 16 × 5. Selected epoch 8
on val 0.881; gold there 0.867 — +0.009 val and +0.006 gold over the
reference.** The rank loss was the problem, and dropping it is worth +0.013
gold on its own:

| config (fold 1, 16 × 5) | val peak | gold at selected |
|---|---|---|
| reference, no aux losses | 0.872 | 0.861 |
| report 0.2 + rank 0.2 (the screen) | 0.868 | 0.854 |
| **report 0.5 alone** | **0.881** | **0.867** |

- The screen's net −0.007 was one loss helping and the other hurting, exactly
  as the plan's borderline branch predicted. Rank's own loss had collapsed to
  0.026 — it was ranking its own queue, not learning to rank studies.
- **This matches the 24-slice run's numbers at 16-slice cost** (that run:
  val 0.880, gold 0.865, LB 0.891). Same accuracy for ~30% less training and
  ~33% less inference — strictly better on the efficiency track, where the
  aux loss costs nothing because it is dropped at export.
- **The overfitting signature is gone.** At 24 slices val peaked at epoch 7
  and fell to 0.871 while gold fell 0.865 → 0.842. Here val holds flat
  (0.881, 0.880, 0.880, 0.879, 0.878) and gold *rises* late, reaching 0.884
  at epoch 10. Read that rise as noise — +0.017 on 58 studies is inside the
  ±0.02 floor — but the absence of decay is real, and it is what the
  regression target was predicted to buy.
- 12.2 min/epoch against ~11 for the reference: the report loss costs ~11% of
  training time and nothing at inference.

Next (user, Kaggle): submit this checkpoint (~2 h). Notebook 10 already ships
`TUNED_SLICES = 16`, so the geometry matches with no edit — but
`knee-convnxt` still holds the 24-slice `lit_fold1.ckpt`, which must be
**replaced, not joined**, or notebook 10 globs both and averages two models
at one geometry. Gold has forecast the LB three times running, so expect
roughly 0.895.

## 2026-08-25 (e) — aux-loss screen: gold 0.854, below the 0.861 reference

**The C2 screen (fold 1, 16 × 5, both lambdas 0.2) selected at gold 0.854 —
−0.007 against the reference.** Not submitted. Full training set confirmed
(three shards, 4,349 studies); `hparams.yaml` and the printed
`aux losses: report 0.2 (dim 384)  rank 0.2` line confirm both losses live.

- Both losses **trained**: `report_loss` 0.170 → 0.044, `rank_loss`
  0.652 → 0.026 over 12 epochs. Not the broken branch.
- But `rank_loss` collapsing to 0.026 means the model near-perfectly ranks
  its own detached queue — memorisation, and the prime suspect for the
  deficit. `report_loss` plateauing at 0.044 cosine distance says the
  embedding target is easier than hoped.
- The regularisation signature is visible: late-epoch gold decays
  0.854 → 0.849 (epochs 9 → 12), far milder than the 24-slice run's
  0.865 → 0.842. Val holds flat at 0.867–0.868 through epoch 12.
- Epoch time 11.4 min (vs ~9 min without the losses); the run cost ~2.3 h.

**Branch taken (plan's "flat but trained", adjusted):** one escalation run
with `LAMBDA_REPORT = 0.5` and `LAMBDA_RANK = 0.0` — report alone and
stronger, rank dropped as the likely negative. Generator and notebook 08
updated. If it clears 0.861 with the reduced-decay signature, the follow-up
is 24 slices + report loss; if flat, C2 closes and Phase C is the lever.

Tooling, same day: `BatchProgress` callback (plain-text tqdm over training
batches, rank zero, 15 s refresh) replaces Lightning's widget bar, which
never reached the Kaggle log — the Trainer now sets
`enable_progress_bar=False` and epochs show batch-level progress. Tests
321. An external defect-inspection project doc was screened for
transferable ideas; verdict recorded in `docs/plan.md` (fp16 already in
place, serving stack inapplicable, OHEM and `torch.compile` parked as
unmeasured candidates).

## 2026-08-25 (d) — two auxiliary objectives implemented, ready for a fold-1 run

**With capacity spent, two training-time losses now attack what the pipeline
structurally ignored: the report text beyond the 12 extracted numbers, and
the fact that the metric ranks while BCE calibrates.** Both are auxiliary
only — the exported state dict keeps just the `model.`-prefixed keys, so
notebook 10 and the submission are untouched.

- **Report-embedding regression.** `src/labels/report_embed.py` embeds every
  report with a local multilingual encoder (`intfloat/multilingual-e5-small`,
  384 dims, 512-token window — the reports are Spanish/Dutch/German and
  long). Output `data/meta/report_embeddings.parquet`. In training, a head on
  the *Lightning wrapper* (deliberately not on `KneeNet`) regresses the
  pooled study embedding onto the report embedding by cosine distance,
  weight `LAMBDA_REPORT = 0.2`. `KneeNet.pool_study`/`forward_pooled` expose
  the representation; the head vanishes at export by construction.
- **Pairwise ranking loss.** `net.pairwise_rank_loss`, RankNet-style, weight
  `LAMBDA_RANK = 0.2`. One study per rank means no in-batch pairs, so the
  module keeps a cross-batch queue (256 studies, per rank, reset each epoch)
  of detached recent scores and ranks the current study against it. Pairs
  count only when soft targets differ by ≥ 0.25, weighted by the gap and both
  label weights — ordering studies the extractor barely distinguishes would
  train on extraction noise.
- Plumbing: `StudyIndex(..., reports=...)` → `collate_studies` emits
  `report`/`report_mask`; notebook 08 loads the parquet if attached and
  prints a notice (and disables the loss) if not. The head always joins the
  graph — DDP errors on a parameter that misses a backward pass.
- Notebook 08 returned to **16 slices** (the iteration geometry; 24 remains
  a one-line change for a final AUC-slot run). Constants at the top of the
  train cell; either loss ablates by setting its lambda to 0.0.
- Tests 302 → 320: rank-loss ordering/gap/graph properties, report plumbing
  and masking, export dropping the head, DDP-safety of a reportless batch,
  the artefact contract of `report_embeddings.parquet`.

Next (user, Kaggle): upload `report_embeddings.parquet` into `knee-labels`,
re-upload notebook 08, run fold 1 (~1.8 h). Gate in `docs/plan.md` Phase C2:
gold must beat 0.861 meaningfully at the selected epoch.

## 2026-08-25 (c) — twenty-four slices: **LB 0.891**, and the ladder ends

**+0.002. The third capacity rung is the last one: gold predicted it, the cost
does not justify it, and on the efficiency metric it is a regression.**

    epoch    1     2     3     4     5     6     7     8     9    10    11    12
    val    0.773 0.817 0.835 0.863 0.862 0.873 0.880 0.874 0.879 0.875 0.873 0.871
    gold   0.737 0.807 0.819 0.856 0.860 0.868 0.865 0.856 0.862 0.857 0.845 0.842

Same fold (1), same labels, only `SLICES_PER_SERIES` 16 -> 24, so this is a
clean head-to-head against the 0.889 run:

| | val peak | gold at selected |
|---|---|---|
| 16 slices, 5 series | 0.872 | 0.861 |
| **24 slices, 5 series** | **0.880** | **0.865** |

- Selected epoch 7 on val 0.880; gold there 0.865. The move is **+0.008 val,
  +0.004 gold**, against +0.023 and +0.021 gold on the two rungs before it,
  which returned +0.024 and +0.020 on the LB. The slice axis is flattening.
- **Gold cannot settle this.** +0.004 is well inside the ~±0.02 noise floor of
  58 studies. Val's +0.008 on ~870 mined labels is the better signal and is
  still small. Only the leaderboard decides; gate stays **LB > 0.889**.
- **The curve shape changed, and that is the more interesting result.** At 16
  slices val was flat within 0.002 from epoch 7 through 12 and gold was still
  climbing at epoch 8. Here val peaks at 7 and falls to 0.871 by 12 while gold
  falls 0.865 -> 0.842. That is overfitting onset: more capacity per study
  against the same 4,349 studies. Twelve epochs is now past useful rather than
  merely flat, though val still selects the peak on its own, so nothing needs
  changing before the LB reports.
- 17.3 min/epoch against 11 at 16 slices — a ratio of 1.57 against the 1.50 the
  geometry implies. Same independent check that validated the series change:
  the extra slices are being read, not silently dropped.
- **Five folds here would cost 17.3 h of training plus a submission run**, call
  it ~20 h against a 25 GPU-h/week quota — one week, one attempt, no room for a
  mistake. The earlier ~7 GPU-h estimate predates two capacity rungs. If the LB
  confirms the flattening, that hour budget belongs to ConvNeXt-S rather than to
  five folds of a rung worth +0.008 val.

### The LB reported 0.891

- **Gold tracked the LB a third time.** gold +0.023 -> LB +0.024; gold +0.021
  -> LB +0.020; now gold +0.004 -> **LB +0.002**. The small local signal was
  not noise hiding a real gain -- it was an accurate forecast of a small one.
- Capacity ladder, complete: 0.845 -> 0.869 -> 0.889 -> **0.891**. Two rungs
  paid +0.044; the next two paid +0.002 between the last pair. **The slice
  axis is exhausted.**
- **The rung is net-negative on the Efficiency Prize.** With
  `Benchmark - maxAUC` ~= -0.45, an AUC point is worth ~2.22 on the efficiency
  score and an hour of runtime costs 0.111. So +0.002 AUC bought -0.004 while
  +0.94 h of runtime cost +0.081: **the efficiency score got 0.077 worse.**
  General rule, worth keeping: **an extra hour of submission runtime needs
  +0.050 AUC to break even.** No capacity change this project has ever made
  clears that bar. The two tracks pull in opposite directions and the answer is
  the two distinct final submissions already planned, not a compromise config.
- **Experiments should run at 16 slices from here.** 24 costs +57% training and
  +48% inference for +0.002; it is not the geometry to iterate in.

### Two corrections to what was written before this run

- **`docs/plan.md` said the submission loop is bound by per-study DICOM decode
  "since doubling slices barely moved it". That is falsified.** Fitting the
  three measured runtimes (16x3 1.29 h, 16x5 1.95 h, 24x5 2.89 h) gives ~0.30 h
  of shared setup and ~0.0206 h per slice-series unit -- the loop is bound by
  GPU forward passes, not decode. The fit predicts 2.78 h for the point it was
  not given, against 2.89 h measured. Efficiency work should attack forward
  cost.
- **Fold ensembling no longer fits the 9 h cap.** `tuned_predict` builds
  volumes once and multiplies only the forward pass, so five folds at 24 slices
  is ~13.3 h and gets killed; at 16 slices it is ~8.6 h, 24 minutes of slack.
  Two folds at 24 slices (~5.5 h) is the practical ceiling. The "five folds,
  +0.005-0.010" plan was written for a cheaper geometry and no longer holds.

---

## 2026-08-24 (e) — a missing input became a submission of NaN

The sixteen-slice submission failed its own validator with "missing
predictions". Cause was not the model: `knee-convnxt` was not attached, so
`find_inputs("convnext_fold*.pt")` returned nothing and every tuned
prediction was None.

- The guard asked whether *any* checkpoint had been found. With `BLEND = 1.0`
  the attached frozen weights satisfied it, the frozen path was then skipped
  by design, and `pd.DataFrame(None, columns=...)` produced an empty frame
  that the id insert padded into rows of NaN -- several cells downstream of
  the actual problem, and printed as `tuned folds : 0` where it was easy to
  scroll past.
- Guards now check the paths the blend actually uses, and a `blended` of None
  refuses before the frame is built. `check_submission.py` gained
  `CHECK_OMIT_TUNED=1`, which withholds the tuned checkpoints and asserts the
  notebook refuses; verified both ways. Four tests.
- The reason this reached a submission at all: notebook 08's section 7 scores
  gold *from the exported checkpoint* and would have caught an export
  problem, but the earlier reporting crash meant it never ran. Export
  verification is not optional before notebook 10.

---

## 2026-08-25 (b) — five series: **LB 0.889**, new best

+0.020 over 0.869, and it tracked the gold gain (+0.021) closely enough to
retire the fold-1 confound: the capacity effect was real and fold variance did
not meaningfully distort it.

| configuration | gold | LB |
|---|---|---|
| 8 slices, 3 series | 0.827 | 0.845 |
| 16 slices, 3 series | 0.840 | 0.869 |
| **16 slices, 5 series** | **0.861** | **0.889** |

Runtime scaled 1.29 h -> 1.95 h, near the 5/3 ratio of the change, which was
independent evidence at submission time that the extra series were actually
being read rather than silently dropped.

Two capacity rungs, +0.044 between them, with labels held constant throughout.

---

## 2026-08-25 — five series: gold 0.840 -> 0.861

**Second capacity rung lands about as hard as the first. Submission pending.**

    epoch    1     3     5     7     8    10    12
    val    0.769 0.831 0.865 0.870 0.871 0.872 0.870
    gold   0.740 0.823 0.850 0.856 0.864 0.861 0.858

- Selected epoch 10 on val 0.872; **gold there 0.861**, against 0.840 for the
  three-series run. Capacity has now moved gold 0.817 -> 0.840 -> 0.861 while
  labels held constant.
- **Confounded by fold.** This was `FOLD = 1`, not 0. Gold is measured on the
  same 58 studies regardless of fold so the comparison is meaningful, but the
  delta carries fold-to-fold variance, which has never been measured here.
  +0.021 is large enough that variance is unlikely to explain it.
- Selection cost more than usual: gold peaked at epoch 7 (0.873) while val
  selected epoch 10 (gold 0.861), a 0.012 gap against the 0.002 seen before.
  Val is flat within 0.002 from epoch 7 onward, so the choice among those
  epochs is arbitrary from val's side and gold's bounce is 58-study noise.
  Not a selection fault -- selecting on gold remains off the table.
- Val and gold now sit ~0.011 apart, against ~0.02 in the eight-slice runs.
- Converged: val plateaus from epoch 7, so twelve epochs remains right and
  more would not obviously help.
- Cost 2.2 h on two T4s at ~11 min/epoch.

---

## 2026-08-24 (f) — sixteen slices: **LB 0.869**, new best

**+0.024 over the previous best of 0.845, and the largest move the project
has made.** Gold 0.840 -> LB 0.869, an offset of +0.029 against the +0.011 to
+0.020 seen previously.

Every submission to date:

| configuration | honest gold | LB |
|---|---|---|
| frozen DINOv2-S | 0.770 | 0.787 |
| frozen + tuned, 50/50 | - | 0.839 |
| 8 slices, categorical labels | 0.825* | 0.845 |
| 8 slices, dual-mode soft | 0.816 | 0.827 |
| 8 slices, dual-mode sharpened | 0.827 | 0.843 |
| **16 slices, dual-mode sharpened** | **0.840** | **0.869** |

\* inflated by the gold-selection leak, since fixed.

The label work was not wasted -- the 16-slice run trains on the dual-mode
sharpened labels -- but on its own it moved nothing, and capacity moved
everything. Slices were the binding constraint, exactly as the elimination
argument predicted once the two delivery bugs were ruled out.

Two bugs paid for reaching this number, both now guarded: a missing input
became a submission of NaN, and the checkpoint export sat behind a reporting
cell that crashed.

---

## 2026-08-24 (d) — 16 slices: honest gold 0.817 -> 0.840

**Outcome: the capacity change is the largest single gain measured on this
project. Submission pending.**

Per-epoch, two T4s, effective batch held at eight studies:

    epoch   1     4     7     9    12
    val   0.727 0.847 0.860 0.858 0.854
    gold  0.700 0.835 0.840 0.842 0.839

- Selected epoch 7 (val peak 0.860), gold there **0.840**. Gold's own peak was
  0.842 at epoch 9, so validation-fold selection cost 0.002 -- consistent with
  every prior run and further evidence the selection signal is sound.
- Epoch 1 val 0.727 sat inside the 0.73-0.75 band predicted from the 8-slice
  runs, which was the agreed check that the DDP port had not silently broken
  something. It had not.
- Honest gold by configuration: 0.827 (8 slices, sharpened) -> 0.817 (plus
  silence weights) -> **0.840** (16 slices). The capacity hypothesis, which
  the label experiments pointed at by elimination, is confirmed on gold.
- Converged rather than truncated: val flat from epoch 7, so twelve epochs is
  right and more would not obviously help. The next capacity rung is series
  (3 of ~5.5), not epochs.
- Two bugs paid for on the way: `torch.cuda.is_available()` poisoning the
  fork, and the metrics read sitting in front of the checkpoint export.

---

## 2026-08-24 (c) — Phase B step 1: 16 slices, and Lightning for T4 x2

**Outcome: notebook 08 trains at 16 slices under Lightning DDP across both
Kaggle GPUs. Not yet run.**

- `SLICES_PER_SERIES` 8 → 16. `BATCH_STUDIES` 4 → 2 keeps the per-step image
  count at 96, and `ACCUM_STEPS` is now derived from
  `8 // (BATCH_STUDIES * DEVICES)` so the effective batch is eight studies on
  one GPU or two, which also keeps the OneCycleLR step count fixed.
- **`TUNED_SLICES` in notebook 10 raised to 16 to match.** Nothing enforces
  this: the attention pool masks over the slice axis, so a checkpoint trained
  at 16 and scored at 8 runs clean and just scores worse. Notebook 09 stays
  at 8 deliberately (its job is fitting Colab's ~5.5 GB transfer ceiling) and
  the dataset now raises when the cache supplies fewer slices than configured,
  rather than training quietly on short volumes.
- New `src/model/lightning.py` holding the dataset, LightningModule, an
  epoch reporter, and the checkpoint exporter. It lives in `src/` rather than
  a notebook cell because dataloader workers and the spawn start method
  pickle: a class defined in a cell belongs to `__main__` and does not
  survive that. (`ddp_notebook` itself forks -- corrected below.) Same rule that
  put the cache's pool worker in `src/data/cache.py`.
- Three DDP hazards handled explicitly, each of which fails silently:
  memmaps are opened lazily per process and dropped in `__getstate__` (so
  ~29 GB never gets serialised into a worker); AUC is computed over
  all-gathered predictions, since averaging per-rank AUCs is not the AUC of
  the set; and the checkpoint is exported by reading the file rather than a
  module in memory, because the process that calls `fit` is not the process
  that trained.
- **A test caught a real bug before it reached Kaggle**: the prediction buffer
  was named `self._buffers`, which collides with `nn.Module`'s reserved
  attribute and corrupts `state_dict`, `load_state_dict`, and `.to(device)`.
  Renamed. Lightning was installed locally so these tests run rather than
  skip; 16 new tests, suite at 278.
- Gold remains monitor-only: the checkpoint callback monitors `val_macro`,
  and a test asserts `monitor="gold_macro"` appears nowhere in the generator.

---

## 2026-08-24 (b) — sharpened fold-0 LB 0.843; silence weighting found retired

**Outcome: sharpening recovered +0.016 (0.827 → 0.843) but still ties the
old-label baseline 0.845. Selection exonerated. Root cause of the remaining
gap identified as a second half of the same bundled change: the probability
extractor silently retired the silence down-weighting.**

- Sharpened fold-0: gold 0.827 at the selected epoch, **LB 0.843**. Three
  fold-0 points now exist: old labels 0.845, soft dual 0.827, sharpened dual
  0.843 — labels improved 0.854 → 0.891 on gold (+0.037) and the LB moved
  −0.002. The label gain is not reaching the model.
- **Selection exonerated.** Val peaked epoch 10 (0.846, gold 0.827); gold's
  own peak was epoch 7 (0.829). Val's pick costs 0.002 against an oracle
  selecting on gold, and both curves flatten after epoch 10, so 12 epochs is
  the right length. The validation signal is working as designed.
- **Root cause found by auditing the weights, not the targets.** The
  probability mode answers for every finding, so `_informative` was true
  everywhere and `WEIGHT_SILENT` never fired: 34.5% of slots (both
  extractors silent) went from 0.25 to 0.7+, a **2.8x increase in training
  mass on the least trustworthy third of the corpus**. Mean weight 0.63 →
  0.78. This contradicted a documented design decision ("silence is weighted
  down, not imputed") and was never a deliberate change — the label swap
  moved targets *and* weights together, the bundling the playbook warns of.
- Fix: `_informative` now tracks the categorical mode alone, which is the
  only extractor with a mention signal. The probability estimate still
  orders studies inside the silent block (the tie-breaking gain is kept);
  it just no longer asserts certainty. Verified: weights are now bit-for-bit
  identical to the old pipeline (mean 0.625, 34.5% at WEIGHT_SILENT), gold
  AUC unchanged at 0.891, targets still dual-mode and 75% extreme.
- Three new tests, including the invariant that would have caught this:
  adding a probability frame must change targets and never weights. Suite
  at 262.
- **Consequence for the standing hypotheses:** the capacity hypothesis has
  *not* had a clean test — the model has never trained on the new labels
  with correct weighting. Capacity (Phase B) waits one more fold-0 run.

---

## 2026-08-24 — fold-0 LB 0.827: soft targets diagnosed, sharpening added

**Outcome: the fold-0 gate fired (LB 0.827 vs old 0.845); the regression was
traced to target-shape dilution, and `fuse.sharpen_targets` now restores the
crisp shape while keeping the 0.891 gold AUC bit-for-bit.**

- Fold-0-only submission scored **0.827** — below the old-label fold-0's
  0.845 despite labels improving 0.854 → 0.891. LB tracked honest gold in
  both cycles (0.825→0.845, 0.816→0.827), so the result is real.
- Local diff of old vs new targets found the mechanism: `score_average`
  pulled confident calls toward the middle. Extreme targets (<0.1 or >0.9)
  fell 73% → 60% of slots; Medial Meniscus mid-zone grew 35% → 63%,
  Synovitis 12% → 36%. Probability mode is never silent, so the old
  silence down-weighting vanished too: mean sample weight 0.63 → 0.78.
  Net: more mid-range targets, counted harder, with weaker BCE gradients —
  and the fine ordering that raised label AUC lives in differences too
  small to learn at n=4,349.
- Fix: `sharpen_targets` quantile-maps each finding's targets onto the
  categorical fusion's marginal, plus an epsilon rank term keeping the map
  strictly increasing. Ordering (and therefore gold AUC 0.891) is preserved
  exactly — verified per finding — while the written labels return to 75%
  extreme (old: 73%). Weights untouched (one change at a time). Four new
  tests; suite at 259.
- Capacity (Phase B) demoted as suspect: a capacity limit caps gains, it
  does not regress 0.018.
- Next: upload sharpened labels, rerun fold 0, gate at LB > 0.845.

## 2026-08-23 → 08-24 — dual-mode labels, leak fix, fold-0 retrain

**Outcome: training labels 0.854 → 0.891 on gold; fold 0 retrained honestly
at gold 0.816; fold-0-only submission run, LB score pending.**

### Label fusion (local)

- Built `src/labels/mode_fusion.py`: combines categorical and probability
  extraction modes under five strategies, each with ≤1 global parameter and
  nothing fitted per finding. Built `src/labels/eval_fusion.py`: gold scoring,
  leave-one-out for the parameterised strategy, delete-8 jackknife, plateau
  scans.
- Extended `fuse.py` with a score-frame path and a probability-aware
  `build()`; verified the no-dumps fallback reproduces the previous
  `train_labels.parquet` bit for bit. Test suite 234 → 255, all passing.
- Gold checkpoint with the 32B extractor (user, Colab): probability mode
  0.810 → **0.860** macro; MCL 0.705 → 0.839, Medial Meniscus 0.576 → 0.765.
- Strategy comparison: `score_average` 0.889, `rank_average` 0.889 (jackknife
  split 0.54/0.47 → tie → simplest wins), both above the 0.885 per-finding
  oracle. Froze `score_average`; it also keeps a probability-like scale
  (mean target 0.33 vs gold prevalence 0.35), which matters for BCE targets.
- `LEXICON_WEIGHT` 0.30 → 0.10: the old weight cost ~0.010 against the
  stronger LLM; 0.00–0.15 is a plateau (0.889–0.891).
- Full 32B corpus run (user, ~1 h): `llm_probs.parquet`, 0% missing. Rebuilt
  `train_labels.parquet`: **gold 0.891**, weakest finding Synovitis 0.792,
  all twelve above the 0.70 kill threshold, 300–975 distinct target values
  per finding (the categorical ties are gone).

### Training pipeline (generator)

- Notebook 08: **gold-selection leak fixed** — best epoch now selected on the
  validation fold's mined labels binarised at 0.5; gold prints as monitoring
  only. `FOLD` is a per-session constant so five sessions produce
  `convnext_fold{0..4}.pt`.
- Notebook 10: `BLEND = 1.0` (leaderboard's verdict: frozen 0.787, blend
  0.839, tuned 0.845); tuned path averages all fold checkpoints it finds.
- Notebook 03: extractor ladder now prefers Qwen2.5-32B-Instruct-AWQ at
  ≥22 GB VRAM, AWQ-safe dtype handling, autoawq install on the transformers
  fallback.
- Fixed `tools/check_notebook.py`, which still asserted the categorical
  schema and had been failing silently since the probability switch; it now
  drives the probability aggregation path. `check_submission.py` fakes two
  tuned folds to exercise multi-checkpoint averaging.

### Fold-0 retrain (user, Kaggle)

- First attempt caught running on **one shard** (tell: 2.3 min/epoch ≈ the
  one-shard speed; 1,490 studies, 21 gold). Stopped, attached all three
  shards, reran: 4,349 studies, 58 gold, 6.1 min/epoch.
- Result: best val macro 0.821, **gold 0.816 honest** (old run: 0.825 *with*
  the selection leak). Per finding vs frozen baseline: Fracture 0.865 vs
  0.683, Baker's 0.924, Medial OA 0.949; weak spots Synovitis 0.699 and
  Lateral OA 0.809 (both below frozen).
- Fold-0-only submission run clean: `frozen=False tuned=True weight=1.0`,
  projected 2.16 h / 1,500 studies (efficiency penalty 0.240). **Decision
  gate on the LB score**: ≥ ~0.85 → launch folds 1–4; ≤ 0.845 → 16-slice
  capacity experiment first. See `docs/plan.md`.

### Documentation

- `PROJECT.md` updated through each state change; `docs/plan.md` (forward
  plan with gates) and `docs/agent_playbook.md` (decision method for whoever
  continues) written; `*.pt` gitignored.
