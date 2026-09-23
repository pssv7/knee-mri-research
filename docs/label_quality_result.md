# Label quality: result of the one-person programme

Plan: `.cross-plan/label-quality-solo/PLAN.final.md`, adapting `docs/label_quality_plan.md`
to one person. Decisions D1–D7: `docs/label_quality/decisions.md`. Audit numbers:
`docs/label_audit_v2.md`. This document holds counts and rates only; no report text.

## Outcome

**Stopped after the audit (decision D7).** The audit found a real, measurable problem in the
PF OA labels and none worth repairing in Lateral Meniscus or Contusion. No repaired labels
were built and no model was trained or submitted, because the only independent evaluation
available to one person, the private leaderboard's macro AUC, cannot resolve an effect on
one finding of twelve.

## What the audit measured

One reviewer (the participant) read 432 study–finding pairs blind to labels and scores, then
re-read 226 of them. Rates come from the random controls only; 95% Wilson intervals.

| Finding | Labels judged wrong (50 controls) | Of which missed positives | Agreed corrections (both passes) |
|---|---|---|---|
| Lateral Meniscus | 4.0% [1.1%, 13.5%] | 2.0% | 1 |
| Contusion | 2.0% [0.4%, 10.5%] | 2.0% | 0 |
| **PF OA** | **22.0% [12.8%, 35.2%]**; 23.0% [15.8%, 32.2%] with the 50 extra controls | **all of them** | **11** (5 missed mentions, 6 severity threshold) |

- **PF OA labels under-call positives.** Every error among its controls was a positive the
  label missed; none was a false positive. Two mechanisms:
  - reports stating patellofemoral cartilage disease in wording no extractor recognised
    (label near 0.01);
  - findings the extractor scored mild, left just under 0.5 by fusion, which the signed
    rubric counts as present from chondromalacia grade II (D2).
- **Model disagreement points at PF OA misses.** Of 50 negative PF OA labels selected
  because the model disagreed, the reviewer judged 34 wrong.
- **Contusion is mostly undecidable from text.** 36 Contusion readings were `unresolved`,
  mostly oedema of unstated cause. The rubric forbids acting on those.
- **Agreement between passes:** 91.3% on readings (κ 0.87) and 97.3% on dispositions
  (κ 0.84) over the 150 controls. All 28 selected proposed changes were agreed a second
  time.

## Limits on these numbers

- **One reviewer.** Agreement between passes is consistency, not independence. No MSK
  reader, adjudicator or second reviewer took part.
- **The 7-day delay before pass 2 was waived (D6).** Agreement, and so the agreed-correction
  count behind the go decision, partly reflects recall. PF OA's go rests on 11 agreed errors.
  The 5 missed mentions alone reach the threshold of 5 exactly; the other 6 depend on the
  reviewer's grade-II rule.
- **Report text, not images.** Every judgement is about what the report says. Nothing here
  measures image truth, and report omissions are invisible to it.
- **Extra PF OA controls weren't blind to role:** the reviewer knew all 48 were controls.
- **Prior exposure** is known only from `label_review_v1` and the gold 58. Reports read
  informally during earlier development were not logged.

## Step 7: the proposal rule, measured and not applied

`src/labels/repair.py` proposes raising a PF OA label when a clause names the patellofemoral
compartment and cartilage disease, is not negated or hedged, and is not graded below II.
It extends the lexicon with spellings it misses (German *Chondromalazie*, Croatian
*hondromalacija*, Turkish *kondromalazi*, Spanish *rótula*) and with cartilage-lesion words.

- On reviewed PF OA pairs with target < 0.5, it finds 20 of 36 agreed corrections (56%).
  About 57% of its proposals are right: missed mentions 7 of 11, severity cases 13 of 26.
- Across the corpus it proposes 648 changes outside the reserve (211 missed mentions, 437
  severity cases). Two-pass review would have taken about 1,300 reads.
- A macro gain the private leaderboard could resolve (~0.01) would need about +0.12 PF OA
  AUC from roughly 0.816. That is why the programme stopped here (D7).

The rule is kept as a record of the wording gaps, which are the concrete lead for any later
change to the PF OA extractor. Nothing in the pipeline imports it.

## Cost

- **Review:** 658 two-stage reviews (432 first pass, 226 second pass), all on 2026-09-23.
- **GPU:** about 0.55 Kaggle GPU-hours, all for three translation runs of 142, 88 and 29
  reports on private kernels, with Qwen2.5-14B-Instruct-AWQ on 2×T4. No training hours.
- **Data handling:** report text stayed local or in private Kaggle datasets and kernels.
  Codex, used for plan review, saw only a redacted copy of the repository.

## Source requirements and their final state

| Source requirement (`docs/label_quality_plan.md`) | Final state |
|---|---|
| Gold 58 kept as historical diagnostics, out of training | done |
| Version labels, weights, report hashes, split, reference predictions | done (`docs/label_quality/baseline_v1.json`) |
| MSK readers reconcile definitions; severity and borderline rules | adapted: solo text rubric, `docs/report_review_rubric.md` (D2) |
| Report / extractor / image questions kept separate | adapted: the image question recorded, never answered |
| Membership assigned before inspecting findings | done (sealed `docs/label_quality/manifest_v1.json`) |
| Reviewed cases out of a locked set; patient grouping; duplicate images | not executable: no locked image set |
| Folds 0/2/4 for discovery, reserved studies excluded; priority first | done |
| Other pending v1 checks as a secondary workstream | not run: 183 pending, stopped by D7 |
| 150 pairs per priority finding, controls first, strata, shortfalls | done: 384 pairs; Lateral Meniscus and Contusion missed-positive strata short (11, 23) |
| Disagreement only selects; coverage by language, scanner, weight, extraction mode | done |
| Blinded reader view | done |
| Ledger fields, seven error classes, final vs intermediate, silence not negative | done |
| Second reviewer before adjudication | adapted: blind self re-review; 7-day delay waived (D6) |
| Original-language expertise | adapted: local translation aid, quotes from the original |
| Counts and intervals by finding, direction, stratum; enlarge controls | done (D4 extra batch for PF OA) |
| 1,000-study image set, pilot, two readers, adjudicator, capture fields, agreement by site | not executable: no readers |
| Leakage checks for new partitions; custodian | not executable / adapted in plan; moot after D7 |
| Power planning for +0.005 / +0.02 / +0.003 | adapted: the resolvable effect estimated at ~0.01 macro; this is what stopped the programme |
| 5.1 narrow evidence-backed rules | done as a measured proposal rule, not applied (D7) |
| 5.2 fresh validation on a reserved sample | not run (D7); the 120-pair reserve is untouched |
| 5.3 versioned candidate labels and correction ledger | not run (D7) |
| 5.4 matched training pair | not run (D7) |
| 5.5 registration and locked readout; confirmation gate | not run (D7) |

## If this is picked up again

- The PF OA wording gaps are the lead: a narrower change to the extractor's PF OA vocabulary,
  checked on the untouched 120-pair reserve before it touches any label.
- A result that could confirm a label effect needs a per-finding evaluation, which means
  image-labelled data beyond the gold 58. The leaderboard can't provide it.
