# Agent playbook — how to think about this project

Written for any model or person continuing this work. `PROJECT.md` records
*what* is true; `docs/plan.md` records *what's next*; this file records *how
decisions get made here*, because the method is what has produced the gains
and the recorded failures are what happens when it is skipped.

---

## 1. The frame: this is a labels project wearing a vision costume

4,407 studies, 58 real labels, 4,349 free-text reports. The imaging model is
a commodity — anyone can fine-tune a ConvNeXt. The manufactured label matrix
is the product. The model tracks its labels almost mechanically (labels 0.854
→ model LB 0.845; labels are now 0.891). When deciding where the next hour
goes, default to the text side unless there is direct evidence the model is
the bottleneck (model trailing its labels by ≫0.02 under honest evaluation).

## 2. The epistemics: three signals, in order of trust

1. **The leaderboard** — the only ground truth. It has already vetoed two
   ideas that gold endorsed (the frozen+tuned ensemble; implicitly, the gold
   figure inflated by selection). When gold and LB disagree, LB wins.
2. **The gold 58** — the only honest local signal, and it is *small*. Its
   noise floor is ~±0.02 macro. Rules that keep it honest:
   - Never train on it. Never let it select an epoch, a weight, or a
     per-finding choice. `splits.assert_no_gold_leakage` guards training;
     nothing guards *you* — selection-by-gold is a leak you can commit in a
     print statement.
   - Never fit per-finding on it. 12 findings × 58 studies has already
     produced one overfit (fitted fusion weights: 0.860 in-sample, 0.812
     leave-one-out). Global rules with ≤1 parameter only; pick parameters
     from plateaus, never argmaxes; when strategies tie under a jackknife,
     take the simplest.
   - Treat per-finding differences of <0.05 on gold as weak evidence.
3. **The validation fold** (~870 mined labels) — noisy truth but not the
   number the project is judged by, which is exactly why it is allowed to
   select checkpoints and gold is not.

## 3. The loop discipline: cheap before expensive

The cost ladder is: local script (seconds) → gold checkpoint on Colab
(minutes) → corpus extraction (~1 h) → training fold (~1.5 h) → five folds +
submission (~8 h). Never buy a rung without a measurement from the rung
below. History: two prompt changes reasoned from radiology domain knowledge
both measured *worse* at the gold checkpoint. Domain reasoning generates
hypotheses; only measurement promotes them.

Related: change one thing at a time. The grid-4 embedding failure is recorded
with its confound (features and projection changed together) — that
ambiguity was the price of bundling changes.

## 4. Verify against the real artefact

Several bugs here survived checks that examined the wrong thing: a notebook
validated by joining source the same wrong way it was written; a checker that
asserted the *old* output schema and had been failing since the pipeline
changed; an import "confirmed" by finding a filename in text. When a check
passes, ask what it actually looked at. When output looks wrong, suspect the
data plumbing first — the worst bugs here (slice order reversed by
`InstanceNumber`, sagittal mirroring flipping the wrong axis) raise no errors
and just cost 0.03 silently.

## 5. Mechanics that are easy to get wrong

- **Notebooks are generated.** Edit `src/` or `tools/make_notebook.py`, run
  `python tools/make_notebook.py`, never edit a notebook directly. A test
  fails if they drift.
- **The traps list in `PROJECT.md` is paid-for knowledge.** Read it before
  touching Kaggle/Colab plumbing; every entry cost real hours or a lost
  shard.
- **Kaggle**: 25 GPU-h/week, one session; T4 not P100; Save & Run All, never
  Quick Save; check *all three* cache shards are attached (one-shard runs
  look healthy and train on a third of the data — this happened).
- **Rules**: reports are Competition Data and may not leave the machine — no
  hosted APIs for extraction; internet off for submission runs.
- **Datasets by pattern-glob**: notebook 10 averages every
  `convnext_fold*.pt` it can see. Stale checkpoints attached alongside new
  ones get silently averaged in.

## 6. How to proceed when picking this up cold

1. Read `PROJECT.md` end to end, then `docs/plan.md`.
2. Establish which artefacts exist locally (`data/meta/`) and which step is
   blocked on a human running Colab/Kaggle — most steps are; say so rather
   than working around it.
3. Before proposing anything, check whether it is already in the
   tried-and-failed list. The failures are documented *with reasons* so they
   don't get re-run wearing a new hat.
4. Prefer the smallest measurement that could kill your idea. If it survives
   the gold checkpoint, scale it one rung.
5. Record what you did in `PROJECT.md` — numbers, not adjectives — and keep
   commit messages explanatory; the git log is the project's lab notebook.
6. Conventions: neutral voice everywhere, tests accompany every non-trivial
   module, nothing fitted per-finding on gold, plateaus not argmaxes.

## 7. Current standing hypotheses (falsifiable, not settled)

- The 0.94 cluster's edge is primarily extraction quality, not architecture.
  Falsified if: labels reach ~0.93 on gold and the model still can't clear
  ~0.90 LB.
- The model can run ~0.01 below its labels' quality. Falsified if the gap
  stays ≫0.03 under honest selection — then capacity (slices, series,
  backbone) becomes the priority, in that order.
- Gold underestimates LB by ~0.02 (observed once). One data point; do not
  lean on it.
