"""Audit of the label tributary: reproduction, properties, and language reach.

`train_labels.parquet` is the other input to every fold the 0.906 ensemble is
made of, and step 1 verified only the image side. This does for the label
matrix what `verify_cache.py` did for the cache -- rebuild it from the code
that claims to produce it and compare -- and then asks the question step 2
asked of the side classifier: what population is the reported accuracy
measured on, versus the population the artifact is applied to?

Four parts, each independently useful:

  A. REPRODUCTION.  Rebuild the shipped matrix from `fuse.build()` and the
     extractor dumps, and compare every cell. Cheap enough (~10 s) to cover
     100% of the corpus rather than the 2% sample `verify_cache.py` could
     afford, because nothing here decodes a pixel.

  B. PROPERTIES.  The numbers the docs quote about the artifact -- 75% extreme
     targets, mean weight 0.625, 34.5% at WEIGHT_SILENT -- read off the file
     rather than off the worklog. Step 2 found the side classifier's
     documented accuracy belonged to a run that no longer existed; this is the
     same check for the labels.

  D. AGAINST THE GOLD IMAGES.  Miss rate and false-alarm rate against the 58
     radiologist-labelled studies -- the only external truth there is. Kept
     apart from the AUC, which measures ordering and cannot see that these
     labels are precise and incomplete.

  C. LANGUAGE REACH.  The corpus is multilingual and the lexicon is one of the
     two signals whose *informativeness* sets every sample weight. If its
     reach varies by language, then studies are weighted by what language they
     were written in, which is not a property anybody chose. Reported as a
     length-matched comparison, because non-English reports are also shorter
     and the confound would otherwise carry the result.

    python tools/audit_labels.py
    python tools/audit_labels.py --sweep      # also sweep LEXICON_WEIGHT

Exit status is non-zero if the shipped matrix does not reproduce, or if a
documented property has drifted. The language section never fails the run --
it is a measurement, not a defect, and what to do about it costs five fold
retrains.
"""

import argparse
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.labels import calibrate, fuse, mode_fusion

# ---------------------------------------------------------------------------
# Language cues
# ---------------------------------------------------------------------------
# No language detector is installed and none is worth a dependency for this:
# every report in this corpus names the knee, the menisci and a side, so
# anatomy and laterality words identify the language with far higher precision
# than general-purpose stopwords would. Anything short of a clear margin is
# left "unknown" rather than guessed -- an audit that mislabels its own strata
# is worse than one that admits a gap.
CUES = {
    "en": ["knee", "meniscus", "medial", "lateral", "tear", "effusion",
           "the ", "and ", "with", "joint"],
    "es": ["rodilla", "menisco", "izquierd", "derech", "articulaci",
           "tecnica", "los ", "del "],
    "pt": ["joelho", "menisco", "direito", "esquerdo", "articula", "ossea"],
    "it": ["ginocchio", "menisco", "destro", "sinistro", "articolazione",
           "legamento"],
    "fr": ["genou", "menisque", "droit", "gauche", "articulation",
           "ligament croise"],
    "de": ["knie", "meniskus", "rechts", "links", "gelenk", "kreuzband",
           "erguss", "nachweis"],
    "nl": ["knie", "meniscus", "rechter", "linker", "gewricht", "kruisband",
           "het ", "van de"],
    "tr": ["diz", "menisk", "sag ", "sol ", "eklem", "kemik", "bag ",
           "izlenmi"],
    "hr": ["koljen", "meniskus", "desno", "lijevo", "zglob", "ligament",
           "prikaz", "promjen"],
}
_GREEK = re.compile(r"[Ͱ-Ͽἀ-῿]")
_CYRILLIC = re.compile(r"[Ѐ-ӿ]")


def detect_language(text) -> str:
    """One of the CUES keys, a script code, 'unknown', or 'empty'."""
    if not isinstance(text, str) or not text.strip():
        return "empty"
    lowered = fuse.lexicon._norm(text)
    length = len(lowered)

    if len(_GREEK.findall(lowered)) > 0.10 * length:
        return "el"
    if len(_CYRILLIC.findall(lowered)) > 0.10 * length:
        # Serbian Cyrillic carries ј љ њ ћ џ; Bulgarian leans on щ ъ;
        # Russian on ы э ё. One of the three is usually unambiguous.
        counts = [(sum(lowered.count(c) for c in "јљњћџ"), "sr"),
                  (sum(lowered.count(c) for c in "щъ"), "bg"),
                  (sum(lowered.count(c) for c in "ыэё"), "ru")]
        best = max(counts)
        return best[1] if best[0] else "cyrillic?"

    scores = sorted(((sum(lowered.count(c) for c in cues), name)
                     for name, cues in CUES.items()), reverse=True)
    top, second = scores[0], scores[1]
    if top[0] < 2 or top[0] < second[0] * 1.5:
        return "unknown"
    return top[1]


# ---------------------------------------------------------------------------
# A. Reproduction
# ---------------------------------------------------------------------------

def check_reproduction(shipped: pd.DataFrame, rebuilt: pd.DataFrame,
                       header: str = "A. REPRODUCTION") -> bool:
    """Every cell of the shipped matrix, against a rebuild from source."""
    print(header)
    ok = True
    if list(shipped.columns) != list(rebuilt.columns):
        print("   [FAIL] column set differs")
        return False
    if not shipped[config.ID_COL].equals(rebuilt[config.ID_COL]):
        print("   [FAIL] study order differs")
        return False

    left = shipped.set_index(config.ID_COL)
    right = rebuilt.set_index(config.ID_COL).loc[left.index]
    worst = 0.0
    differing = 0
    for column in left.columns:
        delta = np.abs(left[column].to_numpy(float)
                       - right[column].to_numpy(float))
        worst = max(worst, float(delta.max()))
        differing += int((delta > 1e-12).sum())

    cells = left.shape[0] * left.shape[1]
    if differing:
        print(f"   [FAIL] {differing:,} of {cells:,} cells differ, "
              f"max |delta| {worst:.3e}")
        ok = False
    else:
        print(f"   [PASS] {cells:,} cells identical across "
              f"{len(left):,} studies and {len(left.columns)} columns")
    return ok


# ---------------------------------------------------------------------------
# B. Documented properties
# ---------------------------------------------------------------------------

DOCUMENTED = {"extreme_share": 0.75, "mean_weight": 0.625, "silent_share": 0.345}


def check_properties(labels: pd.DataFrame, tolerance: float = 0.01,
                     header: str = "B. DOCUMENTED PROPERTIES") -> bool:
    """The numbers the docs quote, read off the artifact itself."""
    print("\n" + header)
    targets = labels[config.LABELS].to_numpy(float)
    weights = labels[[f"{c}_weight" for c in config.LABELS]].to_numpy(float)
    measured = {
        "extreme_share": float(np.mean((targets < 0.1) | (targets > 0.9))),
        "mean_weight": float(weights.mean()),
        "silent_share": float(np.mean(np.isclose(weights, fuse.WEIGHT_SILENT))),
    }
    ok = True
    for name, expected in DOCUMENTED.items():
        got = measured[name]
        status = "PASS" if abs(got - expected) <= tolerance else "FAIL"
        ok &= status == "PASS"
        print(f"   [{status}] {name:14s} {got:.4f}   documented {expected}")
    return ok


# ---------------------------------------------------------------------------
# C. Language reach
# ---------------------------------------------------------------------------

def language_reach(reports: pd.DataFrame, unlabelled: pd.DataFrame,
                   deciles: int = 10, header: str = "C. LANGUAGE REACH"):
    """Per-language informativeness of each of the two extractor signals.

    The weight of every slot is set by two booleans -- did the lexicon say
    anything, did the LLM say anything -- so this is the quantity that decides
    how much a study is allowed to teach, independent of what it says.
    """
    print("\n" + header)
    language = reports.set_index(config.ID_COL)["language"]
    groups = unlabelled[config.ID_COL].map(language).to_numpy()

    mined = (pd.read_parquet(config.META_DIR / "llm_labels.parquet")
             .set_index(config.ID_COL)
             .loc[unlabelled[config.ID_COL]].reset_index())
    scores = mode_fusion.combine(
        mined, pd.read_parquet(config.META_DIR / "llm_probs.parquet"))
    lex = fuse.lexicon_frame(unlabelled["Report"], unlabelled[config.ID_COL])

    lex_ok = np.column_stack([lex[c].map(fuse._informative).to_numpy()
                              for c in config.LABELS])
    llm_ok = np.column_stack([scores[f"{c}_informative"].to_numpy(bool)
                              for c in config.LABELS])
    chars = unlabelled["Report"].str.len().to_numpy()

    rows = []
    for name in pd.Series(groups).value_counts().index:
        mask = groups == name
        rows.append({
            "lang": name,
            "n": int(mask.sum()),
            "lexicon %": 100 * lex_ok[mask].mean(),
            "llm %": 100 * llm_ok[mask].mean(),
            "both %": 100 * (lex_ok[mask] & llm_ok[mask]).mean(),
            "median chars": int(np.median(chars[mask])),
        })
    table = pd.DataFrame(rows)
    print("\n   informative share of the 12 slots, per language\n")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.1f}"))

    # The confound: non-English reports are also shorter, and a shorter report
    # genuinely mentions fewer findings. Comparing inside corpus-wide length
    # deciles removes it.
    bins = pd.qcut(chars, deciles, labels=False, duplicates="drop")
    big = [row["lang"] for _, row in table.iterrows()
           if row["n"] >= 80 and row["lang"] not in ("unknown", "empty")]
    matched = pd.DataFrame(index=[f"d{i}" for i in range(deciles)])
    for name in big:
        mask = groups == name
        matched[name] = [
            100 * lex_ok[mask & (bins == b)].mean()
            if (mask & (bins == b)).sum() >= 10 else np.nan
            for b in range(deciles)]
    print("\n   length-matched: lexicon informative %, by report-length decile\n")
    print(matched.to_string(float_format=lambda x: f"{x:.1f}"))
    return table


def stem_reach(reports: pd.DataFrame):
    """Where a stem list misses a language, name the word it missed.

    A statistical gap is an observation; the stem that does not fire is a
    cause. Multi-word stems are the exposure -- they require the two words
    adjacent and uninflected, which Latin word order gives for free and
    Slavic definite-article suffixes and coordinated noun phrases do not.
    """
    print("\n   stem lists that do not reach their own language")
    print(f"   {'finding':18s} {'stems':>5s} {'multi-word':>11s}")
    for name in ("Medial Meniscus", "Lateral Meniscus", "ACL"):
        stems = fuse.lexicon.FINDINGS[name]["anatomy"]
        multi = sum(" " in s for s in stems)
        print(f"   {name:18s} {len(stems):5d} {multi:10d}")

    print()
    probes = [("bg", "мениск", "Medial Meniscus"),
              ("hr", "menisk", "Medial Meniscus"),
              ("el", "μηνισκ", "Medial Meniscus")]
    for lang, probe, finding in probes:
        subset = reports[reports["language"] == lang]
        if subset.empty:
            continue
        text = subset["Report"].map(fuse.lexicon._norm)
        pattern = fuse.lexicon._stem_pattern(
            fuse.lexicon.FINDINGS[finding]["anatomy"])
        bare = 100 * text.str.contains(probe, regex=False).mean()
        hit = 100 * text.map(lambda t: bool(pattern.search(t))).mean()
        print(f"   {lang}: reports containing '{probe}' {bare:5.1f}%   "
              f"matched by the {finding} stems {hit:5.1f}%")


# ---------------------------------------------------------------------------
# F. Against the gold images
# ---------------------------------------------------------------------------

WEIGHT_MEANING = {
    fuse.WEIGHT_AGREE: "both signals agree",
    fuse.WEIGHT_SINGLE: "only one signal spoke",
    fuse.WEIGHT_DISAGREE: "the two signals disagree",
    fuse.WEIGHT_SILENT: "neither signal spoke",
}


def check_against_gold(gold: pd.DataFrame, fused_gold: pd.DataFrame,
                       threshold: float = 0.5,
                       header: str = "F. AGAINST THE GOLD IMAGES") -> pd.DataFrame:
    """Miss rate and false-alarm rate against radiologist image labels.

    The AUC in Part D says how well the targets *order* studies. It cannot say
    whether they are right, and the two come apart here: these labels are
    precise and incomplete, because a report is not an inventory of what the
    image contains. A finding the radiologist saw and did not write down is
    absent from the report, and the pipeline can only write a zero.

    So misses and false alarms are reported separately -- a pooled error rate
    hides this entirely, since most slots are true negatives and calling them
    all negative already scores well. Broken out by the pipeline's own weight,
    which is the interesting cut: the weight is assigned before anything looks
    at truth, so it either predicts error or it does not.
    """
    print("\n" + header)
    labels = config.LABELS
    truth = np.column_stack([gold[n].to_numpy().astype(int) for n in labels])
    score = np.column_stack([fused_gold[n].to_numpy(float) for n in labels])
    weight = np.column_stack([fused_gold[f"{n}_weight"].to_numpy(float)
                              for n in labels])
    call = score > threshold

    def rates(mask):
        t, c = truth[mask], call[mask]
        positive, negative = t == 1, t == 0
        return (int(mask.sum()), float(positive.mean()) if mask.sum() else np.nan,
                float((~c[positive]).mean()) if positive.sum() else np.nan,
                float(c[negative].mean()) if negative.sum() else np.nan)

    print(f"   {'weight':>7s} {'meaning':26s} {'slots':>6s} {'truly +':>8s}"
          f" {'missed':>8s} {'false +':>8s}")
    for value in sorted(WEIGHT_MEANING, reverse=True):
        n, positive, miss, false_alarm = rates(np.isclose(weight, value))
        if not n:
            continue
        print(f"   {value:7.2f} {WEIGHT_MEANING[value]:26s} {n:6d}"
              f" {100 * positive:7.1f}% {100 * miss:7.1f}% {100 * false_alarm:7.1f}%")
    n, positive, miss, false_alarm = rates(np.ones_like(call, dtype=bool))
    print(f"   {'':7s} {'ALL':26s} {n:6d} {100 * positive:7.1f}%"
          f" {100 * miss:7.1f}% {100 * false_alarm:7.1f}%")

    rows = []
    for i, name in enumerate(labels):
        t, c = truth[:, i], call[:, i]
        if t.sum() == 0:
            continue
        rows.append({"finding": name, "positives": int(t.sum()),
                     "missed %": 100 * float((~c[t == 1]).mean()),
                     "false + %": 100 * float(c[t == 0].mean())
                     if (t == 0).sum() else np.nan})
    columns = ["finding", "positives", "missed %", "false + %"]
    # No positives anywhere means every miss rate is undefined rather than
    # zero, so the table is empty by construction and must not be sorted into
    # a KeyError. Reachable with a small or skewed gold set.
    table = (pd.DataFrame(rows).sort_values("missed %", ascending=False)
             if rows else pd.DataFrame(columns=columns))
    print("\n   per finding, worst first (n is small -- read the order, not"
          " the decimals)\n")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.1f}"))

    # Silence is the design decision this measures, so it gets stated outright
    # rather than left for the reader to derive from the table.
    silent = np.isclose(weight, fuse.WEIGHT_SILENT)
    hidden = int((silent & (truth == 1)).sum())
    if silent.sum():
        print(f"\n   {hidden} of {int(silent.sum())} silent slots are truly"
              f" positive and are written as zero by construction.")
        print("   That is the cost of 'silence means absent, weighted low' --"
              " a design choice,")
        print("   measured here rather than assumed.")
    return table


def sweep_lexicon_weight(gold: pd.DataFrame):
    """Gold macro against the lexicon's share of the target.

    Two things are worth reading off this: whether the shipped 0.10 is on a
    plateau, and whether the sample weights move at all. They do not -- weights
    are set by informativeness, not by this constant -- so the target half and
    the weight half of the lexicon's influence have to be decided separately.
    """
    print("\nD. LEXICON_WEIGHT SWEEP (gold n=58, macro SE ~0.015-0.025)\n")
    original = fuse.LEXICON_WEIGHT
    rows = []
    try:
        for weight in (0.00, 0.05, 0.10, 0.15, 0.20, 0.30):
            fuse.LEXICON_WEIGHT = weight
            fuse.LLM_WEIGHT = 1.0 - weight
            fused, fused_gold, _ = fuse.build()
            aucs = [calibrate._auc(gold[n].to_numpy().astype(int),
                                   fused_gold[n].to_numpy())
                    for n in config.LABELS]
            weights = fused[[f"{c}_weight" for c in config.LABELS]].to_numpy(float)
            rows.append({"LEXICON_WEIGHT": weight,
                         "gold macro": float(np.nanmean(aucs)),
                         "mean weight": float(weights.mean())})
    finally:
        fuse.LEXICON_WEIGHT = original
        fuse.LLM_WEIGHT = 1.0 - original
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if table["mean weight"].nunique() == 1:
        print("\n   sample weights are identical across the sweep: this knob "
              "moves targets only.")
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default=None,
                        help="shipped train_labels.parquet; defaults to META_DIR")
    parser.add_argument("--sweep", action="store_true",
                        help="also sweep LEXICON_WEIGHT against gold")
    parser.add_argument("--out", default=None,
                        help="write the per-language table to this csv")
    args = parser.parse_args()

    shipped_path = Path(args.labels or (config.META_DIR / "train_labels.parquet"))
    shipped = pd.read_parquet(shipped_path)
    print(f"shipped  {shipped_path}  {shipped.shape}\n")

    rebuilt, fused_gold, _ = fuse.build()
    ok = check_reproduction(shipped, rebuilt)
    ok &= check_properties(shipped)
    check_against_gold(calibrate.load_gold(config.TRAIN_CSV), fused_gold)

    reports = pd.read_csv(config.TRAIN_CSV)
    reports["language"] = reports["Report"].map(detect_language)
    unlabelled = calibrate.load_unlabelled(config.TRAIN_CSV)
    table = language_reach(reports, unlabelled)
    stem_reach(reports)

    if args.sweep:
        sweep_lexicon_weight(calibrate.load_gold(config.TRAIN_CSV))
    if args.out:
        table.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")

    print("\n" + ("ALL CHECKS PASSED" if ok else "CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
