# Report-review rubric (solo text review), v1

**What this is.** These are the rules one reviewer applies when reading a knee MRI *report* to
decide what the report states about a finding, and whether the training label represents it
correctly. It is step 2 of the one-person label-quality plan
(`.cross-plan/label-quality-solo/PLAN.final.md`), adapting `docs/label_quality_plan.md` §1–2.

**What this is not.** It is **not an MSK image-annotation standard**, and it was not
reconciled or signed by qualified musculoskeletal readers. None took part, so the source
plan's reader reconciliation is recorded as not executable. A report review establishes what
the report says. It never establishes what the images show. The rubric was written without
looking at the 58 gold labels, and no rule in it may be tuned on them.

**Status.** Signed 2026-09-23: every rule marked **Proposed** was confirmed as written (see §7).
Questions marked **Unresolved** cannot be settled from text under the available definitions.
Cases that hit them are recorded, never corrected. After sign-off, this file's hash is sealed
into the manifest (plan step 3), and any later change is a new version.

## 1. The three questions, kept apart

For every study–finding pair, the ledger records three separate answers:

1. **What the report states** (`report_states`): decided with this rubric, from the original
   report, *before* the reviewer sees any target, score or extractor output.
2. **Whether the extraction represents it** (`extractor_correct`, `final_target_correct`):
   decided after the blinded judgement is saved, by comparing it with the intermediate
   extractor calls and with the final fused target.
3. **What the images establish** (`image_establishes`): always left empty. No image review
   exists in this plan. A suspected disagreement between report and image is recorded as an
   *unresolved image question*.

## 2. Assertion states

Each pair gets exactly one `report_states` value.

| State | Meaning | Examples of wording (any language) |
|---|---|---|
| `present` | The report asserts the finding in the named structure, without hedging. | "tear of the …", "rupture", "there is …", "consistent with …" |
| `present_below_threshold` | Asserted, but at a grade the finding's rule (§5) does not count. | "grade I chondromalacia" under a PF OA rule that starts at grade II |
| `negated` | The report explicitly states the structure is normal/intact or the finding absent. | "no tear", "intact", "without abnormality", "normal signal" |
| `uncertain` | The report hedges. | "possible", "suspicious for", "cannot be excluded", "likely", "questionable", "vs." |
| `unmentioned` | The report is silent on this finding or structure. | no reference at all |
| `unattributable` | The report mentions the finding, but the structure or compartment it names does not determine this finding. | "meniscal tear", with no side; "knee osteoarthritis", with no compartment |

Rules for applying the states:

- **Negation scope.** A negation covers only what it names. In "no tear of the medial
  meniscus; the lateral meniscus shows a radial tear", Lateral Meniscus is `present` and
  Medial Meniscus is `negated`. A blanket normal statement ("menisci and ligaments without
  abnormality") negates every structure it names, and only those.
- **Contradiction inside one report.** If the findings section and the impression disagree,
  the more specific statement wins. If neither is more specific, the state is `uncertain`.
- **Prior studies.** "Known", "unchanged" or "persistent" findings are `present`. "Healed",
  "resolved" or "no longer seen" findings are `negated` for the current study.
- **Silence is never a negative.** `unmentioned` never supports a change to a negative target,
  and never supports a positive one.
- **Language.** The decisive phrase is judged in the original language. The local translation
  (plan step 4) is only an aid. If the decisive phrase can't be interpreted with confidence,
  the state is `uncertain` and the note says "language". The evidence quote in the ledger is
  always an exact substring of the original report. `src/labels/lexicon.py` lists the
  multilingual stems the extractors use and is a useful vocabulary reference.

## 3. Certainty and severity fields

- `certainty`: `definite` for `present` and `negated`; `probable` for "likely" or "consistent
  with, probable"; `possible` for "possible", "cannot be excluded" or "suspicious".
- `severity`: the report's own grade or adjective, recorded as written ("grade II",
  "mild", "small", "partial-thickness"), plus a normalised value: `minimal` / `moderate` /
  `severe` / `not stated`.
- **How the extractor scores these:** severity `minimal` = 0.35, `moderate` = 0.85,
  `severe` = 1.0, multiplied by certainty `definite` 1.0 / `probable` 0.7 / `possible` 0.5
  (`src/labels/llm_extract.py`). The fused target then averages the categorical and
  probability modes (`src/labels/mode_fusion.py`). So **a mild finding stated plainly gets a
  target below 0.5 by design.** That is a severity/definition question (§5, §6), not by itself
  an extraction error.

## 4. When the final target is judged correct

The judgement is made on the **final fused target**. An intermediate extractor call that is
wrong while the final target is right is recorded (`extractor_correct = false`) and never
corrected (`final_target_correct = true`).

| `report_states` | Final target counted correct when | If not |
|---|---|---|
| `present` | target ≥ 0.5 | candidate error: missed positive |
| `negated` | target < 0.5 | candidate error: false positive |
| `present_below_threshold` | target < 0.5 | candidate error: false positive (severity/definition) |
| `uncertain` | always: no judgement from text | keep |
| `unmentioned` | always: no judgement from text | keep (report omission, if any, is invisible to text review) |
| `unattributable` | always: no judgement from text | keep; note the structure the report names |

- A target below 0.5 on its own is **never** evidence of an error.
- A candidate error only becomes a proposed correction if it passes the §6 rules and both
  review passes agree (the plan's two-pass rule).

## 5. Finding rules

The competition's definitions are one line per label (Kaggle dataset description). The
project's are `FINDING_DEFINITIONS` in `src/labels/llm_extract.py`. Neither sets severity
thresholds.

### 5.1 Priority findings

#### Lateral Meniscus

- Competition: "lateral meniscus tear".
- Project: "Tear of the lateral (external) meniscus. Degeneration without a tear does not count."

**Proposed rules:**

- **Tear (`present`):**
  - an explicit tear or rupture of the lateral (external) meniscus, of any pattern: radial,
    horizontal, longitudinal, flap, complex, bucket-handle, root;
  - a displaced meniscal fragment;
  - "grade III signal" or signal "reaching/extending to the articular surface" of the
    lateral meniscus, even without the word tear. This is the conventional MRI tear
    criterion.
- **Not a tear:**
  - degeneration, mucoid/myxoid degeneration, or grade I–II intrasubstance signal "not
    reaching the surface": `negated` for tear if the report also says no tear; otherwise
    `present_below_threshold`;
  - meniscal extrusion alone;
  - discoid meniscus alone.
- **Laterality.** The finding must be attributed to the lateral or external meniscus
  (anterior horn, body, posterior horn, root). "Meniscal tear" with no side is
  `unattributable`. "Both menisci torn" makes both `present`. A lateral tear credited to
  Medial Meniscus, or the reverse, is an **anatomy** error.
- **Postoperative:**
  - "status post partial meniscectomy" alone: not a tear;
  - "re-tear" or "new tear": `present`;
  - "postoperative change, re-tear cannot be excluded": `uncertain`.
- **Parameniscal cyst** on the lateral side without a stated tear: `uncertain` (cysts often
  accompany horizontal tears, but the text doesn't assert one).
- **Missed tear phrases across languages.** When a non-English report states a lateral tear
  and the target is negative, record the decisive phrase and the language in the note. The
  audit report (plan step 6) groups `missed_mention` errors by language, so a phrase the
  extractors consistently miss shows up as a repair category rather than as scattered cases.

**Unresolved:** whether grade II signal described as "possibly reaching the surface" should
count. It stays `uncertain`.

#### Contusion

- Competition: "bone contusion / bone bruise".
- Project: "Bone contusion or bone marrow oedema (excluding oedema that is purely degenerative
  subchondral change)."

**Proposed rules:**

- **`present`:**
  - bone contusion, bone bruise, trabecular microfracture/"bone bruise pattern", or bone
    marrow oedema located in a named bone (femoral condyle, tibial plateau, patella, fibular
    head) with a traumatic context or a described traumatic pattern (e.g. lateral femoral
    condyle plus posterolateral tibial plateau; medial patella plus lateral femoral condyle
    after patellar dislocation);
  - the report calls the oedema post-traumatic.
- **Not contusion:**
  - soft-tissue contusion or oedema: subcutaneous, prepatellar, Hoffa fat pad, muscle,
    tendon or ligament oedema. If the report mentions only soft tissue and nothing about
    bone, the state is `unmentioned`; if it says the bones are unremarkable, `negated`.
    Crediting soft-tissue contusion to Contusion is an **anatomy** error;
  - subchondral oedema that the report ties to cartilage loss, osteophytes or osteoarthritis
    (degenerative): `negated` for Contusion when the report attributes it that way.
- **Uncertain cause.** Bone marrow oedema with no stated cause and no traumatic or
  degenerative context is `uncertain`: keep, never corrected.
- **Coexistence with fracture.** Contusion and Fracture are not mutually exclusive.
  - Oedema described around a fracture line alone: Contusion is `uncertain`.
  - A separately described bruise elsewhere: Contusion is `present`.
  - Neither label is corrected on the assumption that one excludes the other.

**Unresolved:**

- Oedema in insufficiency or SONK-type lesions: stays `uncertain` for Contusion. Fracture
  follows its own rule.
- Oedema reported as "reactive" with no further context: stays `uncertain`.

#### PF OA

- Competition: "patellofemoral osteoarthritis".
- Project: "Osteoarthritis, chondromalacia, or cartilage loss in the PATELLOFEMORAL
  compartment (patella or trochlea)."

The two definitions differ. The project counts any chondromalacia; the competition says
osteoarthritis.

**Proposed rules:**

- **`present`:**
  - patellofemoral osteoarthritis or arthrosis;
  - patellar or trochlear osteophytes;
  - cartilage loss, thinning, fissuring, fibrillation or a (partial- or full-thickness)
    defect of the patellar or trochlear cartilage;
  - chondromalacia grade II or higher.
- **`present_below_threshold`:** chondromalacia grade I, cartilage "signal change",
  "softening" or "mild irregularity" without a surface defect.
- **Attribution:**
  - the lateral patellar facet or lateral trochlea is patellofemoral: PF OA;
  - crediting it to Lateral OA is an **anatomy** error;
  - "lateral compartment" alone means lateral tibiofemoral;
  - "tricompartmental" osteoarthritis makes PF OA, Medial OA and Lateral OA all `present`.
- **Unspecified compartment.** "Knee osteoarthritis", "gonarthrosis" or "degenerative
  changes" with no compartment is `unattributable` for PF OA, and equally for Medial OA and
  Lateral OA.

**Unresolved:** where the competition's cut-off lies between chondromalacia and
osteoarthritis. The proposed grade II threshold is a reviewer decision, not a known
competition criterion.

### 5.2 Other findings (secondary workstream)

These apply when the pending v1 full-report checks are completed (plan step 6). The general
rules in §2–4 hold throughout.

| Finding | Competition | Project | Proposed rule | Unresolved |
|---|---|---|---|---|
| ACL | ACL injury | ACL tear, rupture or significant injury; not intact ACL, not PCL | Complete or high-grade partial tear, rupture, non-visualised ACL: `present`. Intact reconstruction graft: `negated`. Graft tear: `present`. PCL findings never count. | Mucoid degeneration, low-grade partial "sprain": `uncertain` |
| MCL | MCL injury | MCL tear, rupture or sprain | Sprain or tear of any grade: `present` (the project definition includes sprain). Medial retinaculum or MPFL injury: not MCL (**anatomy**). | Grade 1 sprain vs the competition's "injury"; periligamentous oedema only: `uncertain` |
| Medial Meniscus | medial meniscus tear | medial tear; degeneration does not count | As Lateral Meniscus, for the medial (internal) meniscus | As Lateral Meniscus |
| Medial OA | OA of the medial tibiofemoral compartment | OA, cartilage loss or thinning, medial femorotibial | Medial femoral condyle or medial tibial plateau cartilage loss, OA or osteophytes: `present`. Patellofemoral findings never count. | Grade of cartilage change that counts (as PF OA) |
| Lateral OA | OA of the lateral tibiofemoral compartment | same, lateral | As Medial OA, lateral side. Lateral patellar or trochlear disease: never (**anatomy**). | As Medial OA |
| Effusion | joint effusion / excess fluid | joint effusion or increased intra-articular fluid | Effusion described as present, at any stated size: `present`. "Physiological" or "normal amount of fluid": `negated`. | Whether "small" or "minimal" effusion counts: stays by severity (§3) |
| Synovitis | inflammation of the joint lining | synovitis or synovial thickening/enhancement | Synovitis or synovial thickening stated: `present`. Periligamentous or fat-pad oedema: not synovitis (**anatomy**). | "Effusion-synovitis" wording; synovial "irritation" |
| Baker's | Baker's cyst | Baker's or popliteal cyst | Baker's, popliteal or semimembranosus–gastrocnemius bursa cyst: `present`. Other cysts (parameniscal, ganglion near ligaments, tibiofibular): not Baker's (**anatomy**). | A cyst in the popliteal region not named as Baker's or popliteal: `uncertain` |
| Fracture | fracture | any fracture, incl. stress, avulsion, insufficiency | Cortical or trabecular fracture line, avulsion (e.g. Segond), osteochondral fracture, stress or insufficiency fracture: `present` | Healed or old fracture; "trabecular microfracture" wording (counts as Contusion; for Fracture, `uncertain`) |

## 6. Error classes and dispositions

A candidate error from §4 gets one of the source plan's seven classes:

| Class | Use when |
|---|---|
| `anatomy` | The finding belongs to a different structure or compartment than the label credits (lateral vs medial, PF vs tibiofemoral, soft tissue vs bone, PCL vs ACL, retinaculum vs MCL). |
| `negation` | The report negates the finding and the target is positive, or the negation's scope was misread. |
| `missed_mention` | The report states the finding plainly and the target is negative. |
| `severity_definition` | The disagreement turns on grade or threshold (§3, `present_below_threshold`, "mild"). |
| `fusion_weighting` | The intermediate calls are right but fusion or the sample weight produces the wrong final target or weight. |
| `report_image_disagreement` | The report text is clear, but there is reason to think the images differ. Recorded, never acted on. |
| `unresolved` | Anything this rubric cannot settle. |

**Disposition rules:**

- **`correct_target`** is allowed only for `anatomy`, `negation` and `missed_mention` errors
  with an exact original-language evidence quote, and only when both passes agree.
- **`severity_definition`** errors are corrected only where §5 gives a **confirmed**
  threshold. If the threshold is unresolved, the disposition is `keep`.
- **`fusion_weighting`** may propose `correct_weight`, with the reason stated. The values are
  set by the candidate step (plan step 7), not here.
- **`report_image_disagreement`, `unresolved`, and every `uncertain`, `unmentioned` or
  `unattributable` pair** get `keep`.
- **Report omissions** (a finding the images might show but the report never mentions) are
  invisible to text review. They are never inferred.

## 7. Reviewer sign-off

The reviewer confirms each **Proposed** rule, or replaces it, before the sampling manifest is
sealed. After that, the rules are fixed for this audit.

| Item | Decision |
|---|---|
| Lateral Meniscus: grade III / surface-reaching signal counts as tear | ☑ confirmed |
| Lateral Meniscus: grade I–II degeneration without surface extension does not count | ☑ confirmed |
| Contusion: oedema tied to OA/cartilage loss is negated; unstated cause is uncertain | ☑ confirmed |
| PF OA: chondromalacia threshold at grade II (grade I = below threshold) | ☑ confirmed |
| Other findings' proposed rules (§5.2) | ☑ confirmed |

Reviewer: participant (sole reviewer) · Date: 2026-09-23 · Rubric version: v1 · Recorded as decision D2 in `docs/label_quality/decisions.md`
