# Worklog

Chronological session record. `PROJECT.md` holds current state; this holds
what happened, with numbers. Newest entry first.

---

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
