"""Tests for the label-matrix audit, on a synthetic corpus with known answers.

The audit's whole value is that it can fail. On the real corpus every check
passes, which proves nothing about the checks themselves -- a function that
returns True unconditionally would look identical. So the fixtures here are
built with the answer known in advance, and every check is exercised twice:
once on data that should pass, once on data broken in the specific way that
check exists to catch.

The synthetic corpus is deliberately multilingual and deliberately silent in
places, because those are the two properties the audit measures.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config  # noqa: E402
from src.labels import fuse  # noqa: E402
from tools import audit_labels  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

REPORTS = {
    # One report per language the detector claims to separate. Each is written
    # the way that language actually writes it -- the Bulgarian carries the
    # definite-article suffix, the Croatian coordinates the two menisci --
    # because those are exactly the constructions the stem lists miss.
    "en": "MRI of the right knee. Tear of the medial meniscus posterior horn. "
          "Small joint effusion. The ACL is intact.",
    "es": "RMN de rodilla derecha. Rotura del menisco medial. "
          "Derrame articular leve. Los ligamentos cruzados sin alteraciones.",
    "bg": "ЯМР на дясна колянна става. Установява се увреда на задния рог на "
          "медиалния менискус. Ставният хрущял е изтънен. Налице е ставен излив.",
    "hr": "MR desnog koljena. Medijalni i lateralni menisk uredne morfologije. "
          "Prikazuje se manji izljev u zglobu.",
    "el": "Μαγνητική τομογραφία δεξιού γόνατος. Ρήξη του έσω μηνίσκου. "
          "Παρουσία αρθρικής συλλογής.",
    "tr": "Sag diz MR. Medial menisk arka boynuzunda yirtik izlenmistir. "
          "Eklem icinde efuzyon mevcuttur.",
}


def fake_ids(n):
    return [f"1.2.826.0.1.3680043.8.498.{i:051d}" for i in range(n)]


def fake_corpus(n_per_language=4, n_gold=6, seed=0):
    """A train.csv-shaped frame: reports, plus labels on the gold rows only.

    `calibrate.load_gold` splits on whether the first finding column is null,
    so the gold rows are the ones carrying integers and the mined rows are the
    ones carrying NaN. Reproducing that convention here is the point -- a
    fixture that got it backwards would test nothing.
    """
    rng = np.random.default_rng(seed)
    languages = list(REPORTS)
    rows = []
    for i in range(n_per_language * len(languages)):
        rows.append({"Report": REPORTS[languages[i % len(languages)]]})
    frame = pd.DataFrame(rows)
    frame.insert(0, config.ID_COL, fake_ids(len(frame)))
    for name in config.LABELS:
        frame[name] = np.nan
    # The gold rows get integer labels; everything else stays NaN.
    for i in range(n_gold):
        for name in config.LABELS:
            frame.loc[i, name] = int(rng.random() < 0.35)
    return frame


def fake_label_matrix(ids, seed=0, silent_share=0.345, mean_weight=0.625):
    """A train_labels-shaped frame hitting the documented properties on purpose.

    Built to satisfy `check_properties` so that the failing case below is
    unambiguous: if the passing fixture did not pass, a later failure would be
    the fixture's fault rather than the check's.
    """
    rng = np.random.default_rng(seed)
    n = len(ids)
    out = {config.ID_COL: list(ids)}
    for name in config.LABELS:
        # 75% of targets extreme, matching the shipped shape.
        extreme = rng.random(n) < 0.75
        target = np.where(extreme,
                          np.where(rng.random(n) < 0.5, rng.random(n) * 0.05,
                                   0.95 + rng.random(n) * 0.05),
                          0.1 + rng.random(n) * 0.8)
        # Weights drawn so the mean lands near 0.625 with ~34.5% at 0.25.
        draw = rng.random(n)
        weight = np.where(draw < silent_share, fuse.WEIGHT_SILENT,
                          np.where(draw < 0.60, fuse.WEIGHT_SINGLE,
                                   np.where(draw < 0.66, fuse.WEIGHT_DISAGREE,
                                            fuse.WEIGHT_AGREE)))
        out[name] = target
        out[f"{name}_weight"] = weight
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# A. Reproduction
# ---------------------------------------------------------------------------

class TestReproduction:
    def test_identical_frames_pass(self):
        frame = fake_label_matrix(fake_ids(50))
        assert audit_labels.check_reproduction(frame, frame.copy()) is True

    def test_one_changed_cell_fails(self):
        """The check exists for this: a single drifted number, nothing else."""
        frame = fake_label_matrix(fake_ids(50))
        broken = frame.copy()
        broken.loc[7, "ACL"] = broken.loc[7, "ACL"] + 1e-6
        assert audit_labels.check_reproduction(frame, broken) is False

    def test_changed_weight_fails(self):
        """Weights are half the artifact and are easy to forget."""
        frame = fake_label_matrix(fake_ids(50))
        broken = frame.copy()
        broken.loc[3, "Effusion_weight"] = fuse.WEIGHT_SILENT
        assert audit_labels.check_reproduction(frame, broken) is False

    def test_reordered_studies_fail(self):
        frame = fake_label_matrix(fake_ids(50))
        assert audit_labels.check_reproduction(
            frame, frame.iloc[::-1].reset_index(drop=True)) is False

    def test_missing_column_fails(self):
        frame = fake_label_matrix(fake_ids(50))
        assert audit_labels.check_reproduction(
            frame, frame.drop(columns=["ACL_weight"])) is False


# ---------------------------------------------------------------------------
# B. Properties
# ---------------------------------------------------------------------------

class TestProperties:
    def test_documented_shape_passes(self):
        frame = fake_label_matrix(fake_ids(4000))
        assert audit_labels.check_properties(frame) is True

    def test_lost_silence_weighting_fails(self):
        """The 2026-08-24 regression: silence stopped being down-weighted.

        It moved the mean weight 0.625 -> 0.78 and nothing else visible. This
        is the check that would have caught it in ten seconds.
        """
        frame = fake_label_matrix(fake_ids(4000))
        for name in config.LABELS:
            column = f"{name}_weight"
            frame[column] = frame[column].replace(fuse.WEIGHT_SILENT,
                                                  fuse.WEIGHT_SINGLE)
        assert audit_labels.check_properties(frame) is False

    def test_diluted_targets_fail(self):
        """Targets pulled toward the middle -- the soft-target dilution bug."""
        frame = fake_label_matrix(fake_ids(4000))
        for name in config.LABELS:
            frame[name] = 0.5 + (frame[name] - 0.5) * 0.2
        assert audit_labels.check_properties(frame) is False


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

class TestLanguageDetection:
    @pytest.mark.parametrize("code", sorted(REPORTS))
    def test_each_language_is_identified(self, code):
        assert audit_labels.detect_language(REPORTS[code]) == code

    def test_empty_and_missing_are_not_guessed(self):
        assert audit_labels.detect_language("") == "empty"
        assert audit_labels.detect_language(None) == "empty"
        assert audit_labels.detect_language(float("nan")) == "empty"

    def test_no_cue_returns_unknown(self):
        """Refusing to guess matters: a mislabelled stratum is worse than a gap."""
        assert audit_labels.detect_language("12345 ... ---") == "unknown"


# ---------------------------------------------------------------------------
# The stem gap the audit exists to surface
# ---------------------------------------------------------------------------

class TestStemReach:
    def test_english_meniscus_is_matched(self):
        pattern = fuse.lexicon._stem_pattern(
            fuse.lexicon.FINDINGS["Medial Meniscus"]["anatomy"])
        assert pattern.search(fuse.lexicon._norm(REPORTS["en"]))

    def test_bulgarian_meniscus_is_missed(self):
        """Not a wish -- this is the defect, pinned so a fix is visible.

        The report says "медиалния менискус"; the stem list holds the Russian
        "медиальн мениск". If someone repairs the stems, this test fails and
        the worklog entry claiming the gap needs updating.
        """
        text = fuse.lexicon._norm(REPORTS["bg"])
        assert "мениск" in text
        pattern = fuse.lexicon._stem_pattern(
            fuse.lexicon.FINDINGS["Medial Meniscus"]["anatomy"])
        assert pattern.search(text) is None

    def test_croatian_coordination_is_missed(self):
        text = fuse.lexicon._norm(REPORTS["hr"])
        assert "menisk" in text
        pattern = fuse.lexicon._stem_pattern(
            fuse.lexicon.FINDINGS["Medial Meniscus"]["anatomy"])
        assert pattern.search(text) is None


# ---------------------------------------------------------------------------
# F. Against the gold images
# ---------------------------------------------------------------------------

def gold_and_targets(truth_rows, score_rows, weight):
    """Two frames shaped like load_gold() and fuse.build()'s gold output."""
    ids = fake_ids(len(truth_rows))
    gold = {config.ID_COL: ids}
    fused = {config.ID_COL: ids}
    for i, name in enumerate(config.LABELS):
        gold[name] = [row[i] for row in truth_rows]
        fused[name] = [row[i] for row in score_rows]
        fused[f"{name}_weight"] = weight
    return pd.DataFrame(gold), pd.DataFrame(fused)


class TestAgainstGold:
    def test_perfect_labels_have_no_misses(self):
        truth = [[1] * 12, [0] * 12]
        score = [[0.9] * 12, [0.1] * 12]
        gold, fused = gold_and_targets(truth, score, fuse.WEIGHT_AGREE)
        table = audit_labels.check_against_gold(gold, fused)
        assert (table["missed %"] == 0).all()
        assert (table["false + %"] == 0).all()

    def test_all_silent_labels_miss_everything(self):
        """Silence writes a zero, so every true positive under it is missed.

        This is the arithmetic behind the real corpus's 100% miss rate in the
        0.25 bucket, and it is a property of the design rather than a bug.
        """
        truth = [[1] * 12, [1] * 12]
        score = [[0.0] * 12, [0.0] * 12]
        gold, fused = gold_and_targets(truth, score, fuse.WEIGHT_SILENT)
        table = audit_labels.check_against_gold(gold, fused)
        assert (table["missed %"] == 100).all()

    def test_misses_and_false_alarms_are_counted_separately(self):
        """Four studies per finding: one miss, one false alarm, two correct.

        A pooled error rate would report 50% for both and hide the asymmetry
        that Part F exists to show.
        """
        truth = [[1] * 12, [0] * 12, [1] * 12, [0] * 12]
        score = [[0.1] * 12, [0.9] * 12, [0.9] * 12, [0.1] * 12]
        gold, fused = gold_and_targets(truth, score, fuse.WEIGHT_AGREE)
        table = audit_labels.check_against_gold(gold, fused)
        assert (table["missed %"] == 50).all()
        assert (table["false + %"] == 50).all()

    def test_findings_with_no_positives_are_dropped(self):
        """A miss rate over zero positives is undefined, not zero."""
        truth = [[0] * 12, [0] * 12]
        score = [[0.1] * 12, [0.9] * 12]
        gold, fused = gold_and_targets(truth, score, fuse.WEIGHT_AGREE)
        assert audit_labels.check_against_gold(gold, fused).empty


# ---------------------------------------------------------------------------
# End to end, on the synthetic corpus
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_language_reach_reports_every_language(self, tmp_path, monkeypatch):
        """The reach table must cover the corpus it was handed, not a subset."""
        corpus = fake_corpus(n_per_language=20, n_gold=6)
        corpus["language"] = corpus["Report"].map(audit_labels.detect_language)
        mined = corpus[corpus[config.LABELS[0]].isna()]

        dumps = tmp_path / "meta"
        dumps.mkdir()
        write_fake_dumps(dumps, mined[config.ID_COL])
        monkeypatch.setattr(config, "META_DIR", dumps)

        table = audit_labels.language_reach(corpus, mined)
        assert set(table["lang"]) == set(corpus.loc[mined.index, "language"])
        assert (table["lexicon %"] <= 100).all()
        # The English arm must out-reach the Bulgarian one on these reports;
        # that is the finding, reproduced on data whose content we control.
        reach = table.set_index("lang")["lexicon %"]
        assert reach["en"] > reach["bg"]


def write_fake_dumps(directory, ids):
    """Minimal llm_labels / llm_probs frames for the mined studies."""
    ids = list(ids)
    rng = np.random.default_rng(1)
    categorical = {config.ID_COL: ids}
    probabilities = {config.ID_COL: ids}
    for name in config.LABELS:
        present = rng.choice([0.0, 1.0, np.nan], size=len(ids),
                             p=[0.45, 0.25, 0.30])
        categorical[name] = present
        categorical[f"{name}_severity"] = np.where(present == 1.0, 0.8, 0.0)
        categorical[f"{name}_confidence"] = np.ones(len(ids))
        categorical[f"{name}_evidence"] = pd.array([None] * len(ids),
                                                   dtype="string")
        probabilities[name] = rng.random(len(ids))
        probabilities[f"{name}_confidence"] = np.ones(len(ids))
    pd.DataFrame(categorical).to_parquet(directory / "llm_labels.parquet")
    pd.DataFrame(probabilities).to_parquet(directory / "llm_probs.parquet")
