"""Validate the generated Colab notebook the way Jupyter actually loads it.

nbformat stores `source` as a list of strings that Jupyter concatenates with no
separator, so every line must carry its own trailing newline. A generator that
splits on "\\n" without restoring it produces a notebook that looks correct in
JSON and collapses to a single line on execution -- and validating it by
joining with "\\n" hides the bug, because that re-inserts exactly what is
missing. These tests join with "" for that reason.
"""

import ast
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

NOTEBOOK = os.path.join(ROOT, "notebooks", "03_extract_labels.ipynb")
MODULE = os.path.join(ROOT, "src", "labels", "llm_extract.py")

pytestmark = pytest.mark.skipif(
    not os.path.exists(NOTEBOOK), reason="notebook not generated"
)


def load_cells():
    with open(NOTEBOOK) as handle:
        return json.load(handle)["cells"]


def cell_source(cell):
    """Concatenate exactly as Jupyter does -- no separator."""
    return "".join(cell["source"])


def code_cells():
    return [c for c in load_cells() if c["cell_type"] == "code"]


def as_python(source):
    """Strip notebook magics so the rest can be parsed as Python.

    Shell and line magics are valid in a cell but not valid Python, and they
    can appear anywhere in it -- not only on the first line, which is what an
    earlier version of this check assumed.
    """
    kept = [line for line in source.splitlines()
            if not line.lstrip().startswith(("!", "%"))]
    return "\n".join(kept)


class TestStructure:
    def test_is_valid_nbformat(self):
        with open(NOTEBOOK) as handle:
            notebook = json.load(handle)
        assert notebook["nbformat"] == 4
        assert notebook["cells"]

    def test_source_lines_keep_their_newlines(self):
        # The actual bug: without trailing newlines every cell becomes one line.
        for index, cell in enumerate(load_cells()):
            lines = cell["source"]
            for line in lines[:-1]:
                assert line.endswith("\n"), (
                    f"cell {index}: source line missing trailing newline -- "
                    f"the cell will collapse to a single line"
                )

    def test_no_multi_statement_cell_collapsed(self):
        # Belt and braces: a real code cell should span several lines.
        multiline = [c for c in code_cells() if cell_source(c).count("\n") > 3]
        assert len(multiline) >= 5


class TestCellsCompile:
    def test_every_code_cell_parses(self):
        for index, cell in enumerate(code_cells()):
            try:
                ast.parse(as_python(cell_source(cell)))
            except SyntaxError as exc:
                pytest.fail(f"code cell {index} line {exc.lineno}: {exc.text}")


class TestEmbeddedModule:
    def test_module_cell_matches_repo_source(self):
        # The notebook embeds llm_extract.py so the two cannot disagree about
        # how 13,000 generations are parsed. Regenerate after editing it.
        with open(MODULE) as handle:
            expected = handle.read().rstrip()
        embedded = [
            cell_source(c).rstrip() for c in code_cells()
            if "def parse_response" in cell_source(c)
        ]
        assert embedded, "no cell embeds the extractor module"
        assert embedded[0] == expected, (
            "notebook is stale -- rerun tools/make_notebook.py"
        )

    def test_module_cell_executes_and_exports(self):
        embedded = [
            cell_source(c) for c in code_cells()
            if "def parse_response" in cell_source(c)
        ][0]
        namespace = {}
        exec(embedded, namespace)  # noqa: S102 - validating generated output
        for name in ("SYSTEM_PROMPT", "build_prompt", "parse_response",
                     "aggregate_samples"):
            assert name in namespace, f"{name} missing from the notebook cell"

    def test_later_cells_only_use_defined_names(self):
        # Cells after the module cell call these; a rename would break the
        # notebook without breaking any test in test_llm_extract.py.
        joined = "\n".join(cell_source(c) for c in code_cells())
        for name in ("build_probability_prompt", "parse_probabilities",
                     "aggregate_probabilities", "SYSTEM_PROMPT"):
            assert joined.count(name) >= 2, f"{name} defined but never used"

    def test_the_categorical_path_is_not_silently_still_wired(self):
        # The notebook was switched to probabilities because the categorical
        # contract tied 53% of label slots onto one value. Phase C lever 3
        # (2026-08-26) reintroduced a *separate* categorical function to
        # measure the 32B, so the invariant tightens rather than disappears:
        # run_extraction -- the function that feeds llm_probs.parquet --
        # must parse probabilities only, and parse_response may appear only
        # inside run_categorical, whose output goes to its own parquet.
        joined = "\n".join(cell_source(c) for c in code_cells())
        start = joined.index("def run_extraction")
        end = joined.index("\ndef ", start + 1)
        run_extraction_body = joined[start:end]
        assert "parse_probabilities" in run_extraction_body
        assert "parse_response(" not in run_extraction_body

        start = joined.index("def run_categorical")
        end = joined.index("\ncat_gold_results", start)
        run_categorical_body = joined[start:end]
        assert "parse_response(" in run_categorical_body
        assert "parse_probabilities" not in run_categorical_body
        # And the corpus cell still calls the probability path.
        corpus = joined[joined.index("mined = run_extraction"):]
        assert "PROMPT_VARIANT" in corpus


class TestBootstrapReloads:
    """The bootstrap must not trust a module already in sys.modules.

    Writing src/*.py does nothing to an import that already happened. Re-running
    the cell in a live kernel then keeps the OLD code with the new file sitting
    unused on disk, and the run looks healthy until something calls a function
    that only exists in the new version -- after the expensive part has already
    run against stale logic.
    """

    def bootstrap(self):
        # The bootstrap lives in the Kaggle notebooks, not the Colab one --
        # only Kaggle needs the repo modules materialised.
        for name in ("01_scan_index.ipynb", "02_build_cache.ipynb"):
            path = os.path.join(ROOT, "notebooks", name)
            if not os.path.exists(path):
                continue
            with open(path) as handle:
                cells = json.load(handle)["cells"]
            for cell in cells:
                if cell["cell_type"] != "code":
                    continue
                source = "".join(cell["source"])
                if "FILES = {" in source:
                    return source
        pytest.fail("no bootstrap cell found in the Kaggle notebooks")

    def test_evicts_previously_imported_modules(self):
        source = self.bootstrap()
        assert "del sys.modules[" in source
        assert "startswith(\"src.\")" in source

    def test_invalidates_the_import_caches(self):
        assert "importlib.invalidate_caches()" in self.bootstrap()

    def test_eviction_precedes_the_imports(self):
        # Compare only within the executable tail. The embedded module sources
        # contain "from src import config" themselves, and matching that copy
        # makes the ordering look wrong when it is not.
        source = self.bootstrap()
        tail = source[source.index('sys.path.insert(0, ".")'):]
        assert tail.index("del sys.modules[") < tail.index("from src import config")


class TestGeneratorIsValid:
    """The generator emitted a broken file and it was committed unnoticed.

    A cell body containing a triple-quoted docstring terminates the
    nb.code('''...''') string that carries it, so the generator itself stops
    parsing. Nothing else in the suite reads the generator, so the failure
    surfaced only on the next run of it.
    """

    def test_generator_parses(self):
        path = os.path.join(ROOT, "tools", "make_notebook.py")
        with open(path) as handle:
            ast.parse(handle.read())

    def test_every_notebook_is_present_and_parses(self):
        directory = os.path.join(ROOT, "notebooks")
        notebooks = [f for f in os.listdir(directory) if f.endswith(".ipynb")]
        assert len(notebooks) >= 4, notebooks
        for name in notebooks:
            with open(os.path.join(directory, name)) as handle:
                cells = json.load(handle)["cells"]
            for index, cell in enumerate(cells):
                if cell["cell_type"] != "code":
                    continue
                try:
                    ast.parse(as_python("".join(cell["source"])))
                except SyntaxError as exc:
                    pytest.fail(f"{name} cell {index}: {exc}")


class TestSubmissionGuardsTheActivePath:
    """A missing checkpoint must refuse, not produce a submission of NaN.

    With `BLEND = 1.0` the frozen path is skipped by design. If the tuned
    checkpoints are also absent, a guard that merely asks "did any checkpoint
    turn up?" is satisfied by the frozen ones, every prediction comes back
    None, and `pd.DataFrame(None, columns=...)` becomes an empty frame that
    the id insert pads into rows of NaN. That reached a real submission once.

    `python tools/check_submission.py` with CHECK_OMIT_TUNED=1 exercises the
    same thing end to end.
    """

    def _ensemble_source(self):
        source = open(os.path.join(ROOT, "tools", "make_notebook.py")).read()
        return source[source.index("def build_ensemble_notebook"):]

    def test_guard_is_conditioned_on_the_active_path(self):
        ensemble = self._ensemble_source()
        assert "if USE_TUNED and not tuned_models:" in ensemble
        assert "if USE_FROZEN and not frozen_models:" in ensemble

    def test_the_old_any_checkpoint_guard_is_gone(self):
        ensemble = self._ensemble_source()
        assert "if not frozen_models and not tuned_models:" not in ensemble

    def test_empty_predictions_refuse_before_the_frame_is_built(self):
        ensemble = self._ensemble_source()
        guard_at = ensemble.index("no predictions to write")
        frame_at = ensemble.index("submission = pd.DataFrame(blended")
        assert guard_at < frame_at


class TestColabFinetuneNotebook:
    """The Colab copy of notebook 08 is generated from the same source.

    Its whole value is that a result from it is comparable with a Kaggle run,
    which holds only while the cells that decide the numbers are identical.
    """

    def _code(self, name):
        path = os.path.join(ROOT, "notebooks", name)
        with open(path) as handle:
            cells = json.load(handle)["cells"]
        return ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]

    def test_both_notebooks_exist(self):
        for name in ("08_finetune.ipynb", "08_finetune_colab.ipynb"):
            assert os.path.exists(os.path.join(ROOT, "notebooks", name))

    def test_the_training_cell_is_identical(self):
        # The bootstrap cell embeds lightning.py, which also defines the
        # class, so match the construction call rather than the name.
        kaggle = [c for c in self._code("08_finetune.ipynb")
                  if "knee_lightning.KneeLitModule(" in c]
        colab = [c for c in self._code("08_finetune_colab.ipynb")
                 if "knee_lightning.KneeLitModule(" in c]
        assert len(kaggle) == 1 and len(colab) == 1
        assert kaggle[0] == colab[0], (
            "the Colab copy's training cell has drifted -- its results are no "
            "longer comparable with a Kaggle run")

    def test_the_oof_dump_is_identical(self):
        kaggle = [c for c in self._code("08_finetune.ipynb")
                  if "oof_fold" in c]
        colab = [c for c in self._code("08_finetune_colab.ipynb")
                 if "oof_fold" in c and "drive" not in c]
        assert kaggle[0] == colab[0]

    def test_the_competition_is_never_downloaded_whole(self):
        # A bare competition_download pulls the 570 GB DICOM archive; only the
        # two CSVs are needed and `path=` is what fetches one file.
        for cell in self._code("08_finetune_colab.ipynb"):
            for line in cell.split("\n"):
                if "competition_download(" in line and "def " not in line:
                    call = cell[cell.index("competition_download("):]
                    assert "path=" in call.split(")")[0] + call.split(")")[1], (
                        "competition_download without path= downloads 570 GB")

    def test_the_cache_comes_from_datasets_not_notebook_outputs(self):
        # notebook_output_download has no Colab resolver and would copy
        # ~9.7 GB per shard instead of mounting it.
        staging = [c for c in self._code("08_finetune_colab.ipynb")
                   if "CACHE_DATASETS" in c][0]
        assert "dataset_download" in staging
        for handle in ("shard-1", "shard-2", "shard-3"):
            assert handle in staging
        # The mount is tried first; the download is the fallback, not the path.
        assert staging.index("dataset_download(") < staging.index(
            "notebook_output_download(")

    def test_the_cache_folder_is_not_pointed_at_drive(self):
        staging = [c for c in self._code("08_finetune_colab.ipynb")
                   if "CACHE_DATASETS" in c][0]
        assert 'os.environ.pop("KAGGLEHUB_CACHE"' in staging

    def test_credentials_are_checked_before_anything_is_fetched(self):
        # Private shards and the competition both 401/404 without auth, and
        # neither failure names authentication as the cause.
        staging = [c for c in self._code("08_finetune_colab.ipynb")
                   if "CACHE_DATASETS" in c][0]
        assert "get_kaggle_credentials()" in staging
        # kagglehub.login() must not be called: in a notebook it renders an
        # ipywidgets form and returns, storing the token from a callback that
        # cannot run until the cell has finished, so the cell never sees it.
        # Compare code lines only: the comment documenting kagglehub's
        # credential order legitimately names login().
        code = "\n".join(line for line in staging.split("\n")
                         if not line.strip().startswith("#"))
        assert "kagglehub.login()" not in code, (
            "login() is asynchronous in a notebook; use getpass instead")
        assert "getpass" in staging
        # The token must reach the environment too, or a re-import loses it.
        assert 'os.environ["KAGGLE_API_TOKEN"]' in staging
        assert "whoami()" in staging, "authentication is never verified"
        check = staging.index("get_kaggle_credentials() is None")
        for call in ("dataset_download(", "competition_download(",
                     "notebook_output_download("):
            assert staging.index(call) > check, (
                f"{call} runs before the credential check")

    def test_drive_is_copied_to_local_disk_not_read_in_place(self):
        staging = [c for c in self._code("08_finetune_colab.ipynb")
                   if "CACHE_DATASETS" in c][0]
        assert "DRIVE_CACHE" in staging and "copy_from_drive" in staging
        # shutil.copy2 onto LOCAL, which must not itself be on Drive.
        assert "shutil.copy2(source, target)" in staging
        local = [l for l in staging.split("\n") if l.startswith("LOCAL = ")][0]
        assert "/content/drive" not in local, (
            "training memmaps these files; the staging target must be local")

    def test_a_short_copy_is_detected(self):
        # A truncated .npy memmaps without complaint and trains on whatever
        # bytes are present, so the copy is size-checked.
        staging = [c for c in self._code("08_finetune_colab.ipynb")
                   if "CACHE_DATASETS" in c][0]
        assert "os.path.getsize(target) != size" in staging

    def test_the_shard_count_is_reported(self):
        # One shard instead of three trains on a third of the data and looks
        # healthy doing it.
        staging = [c for c in self._code("08_finetune_colab.ipynb")
                   if "CACHE_DATASETS" in c][0]
        assert "shards staged" in staging
        assert "len(shards) < 3" in staging


class TestGeneratorHasNoDocstringsInCells:
    """A triple-quoted string in a cell body terminates the generator.

    `nb.code('''...''')` holds the cell source as one string, so a docstring
    inside it closes that string early. PROJECT.md records this as having
    happened twice; it has now happened five times, and the symptom is a
    SyntaxError pointing at the NEXT nb.code call with the message "invalid
    syntax. Perhaps you forgot a comma?", which names neither the file nor the
    cause. This test names both.
    """

    def test_the_generator_still_parses(self):
        import ast
        path = os.path.join(ROOT, "tools", "make_notebook.py")
        with open(path) as handle:
            source = handle.read()
        try:
            ast.parse(source)
        except SyntaxError as exc:
            near = source.splitlines()[max(exc.lineno - 12, 0):exc.lineno]
            raise AssertionError(
                f"tools/make_notebook.py does not parse at line {exc.lineno}. "
                "The usual cause is a docstring inside an nb.code(...) cell "
                "body, which closes the cell string early -- use a comment "
                "instead. Preceding lines:\n" + "\n".join(near)) from None

    def test_every_generated_cell_body_parses_as_python(self):
        import ast
        import json as _json
        for name in sorted(os.listdir(os.path.join(ROOT, "notebooks"))):
            if not name.endswith(".ipynb"):
                continue
            with open(os.path.join(ROOT, "notebooks", name)) as handle:
                cells = _json.load(handle)["cells"]
            for i, cell in enumerate(cells):
                if cell["cell_type"] != "code":
                    continue
                body = "\n".join(
                    line for line in "".join(cell["source"]).splitlines()
                    if not line.lstrip().startswith(("!", "%")))
                try:
                    ast.parse(body)
                except SyntaxError as exc:
                    raise AssertionError(
                        f"{name} cell {i} does not parse: {exc}") from None


class TestTrainingAndSubmissionAgree:
    """The P0 correctness fixes, held in place.

    Each of these ran clean and merely scored worse, which is why each needs a
    test rather than a comment: nothing about them raises, and the cost only
    appears on the leaderboard weeks later.
    """

    def sources(self, name):
        with open(os.path.join(ROOT, "notebooks", name)) as handle:
            notebook = json.load(handle)
        return ["".join(c["source"]) for c in notebook["cells"]
                if c["cell_type"] == "code"]

    def test_the_submission_picks_series_the_way_training_does(self):
        # Training spreads its pick with np.linspace (StudyIndex.series_for);
        # the submission took volumes[:TUNED_MAX_SERIES], a head truncation,
        # which selects a different subset for the 32.5% of studies carrying
        # more than five series.
        joined = "\n".join(self.sources("10_submit_ensemble.ipynb"))
        assert "def pick_series(" in joined
        assert "np.linspace(0, count - 1, TUNED_MAX_SERIES)" in joined
        # The docstring names the old expression, so match the statement.
        assert "picked = volumes[:TUNED_MAX_SERIES]" not in joined

    def test_pick_series_agrees_with_series_for_at_every_count(self):
        """The two rules must select the same positions, not merely both use
        linspace. Extracted from the generated notebook and run against
        `StudyIndex.series_for`'s expression directly."""
        import numpy as np

        joined = "\n".join(self.sources("10_submit_ensemble.ipynb"))
        start = joined.index("def pick_series(")
        end = joined.index("def tuned_predict(")
        namespace = {"np": np, "TUNED_MAX_SERIES": 6}
        exec(compile(joined[start:end], "<pick_series>", "exec"), namespace)
        pick_series = namespace["pick_series"]

        for count in range(1, 16):
            expected = list(range(count))
            if count > 6:
                # src/model/dataset.py, StudyIndex.series_for.
                keep = np.linspace(0, count - 1, 6).round().astype(int)
                expected = np.unique(keep).tolist()
            assert pick_series(count) == expected, count

    def test_no_checkpoint_is_recovered_alphabetically(self):
        # sorted(glob(...))[0] picks lit_fold1-v1.ckpt over lit_fold1.ckpt,
        # because "-" sorts before ".". Lightning writes those -vN suffixes on
        # every name collision.
        for name in ("08_finetune.ipynb", "08_finetune_colab.ipynb"):
            joined = "\n".join(self.sources(name))
            assert "sorted(glob.glob(f\"/kaggle/working/lit_fold" not in joined
            assert "def recover_checkpoint(" in joined
            assert "os.path.getmtime" in joined

    def test_the_averaged_weights_do_not_join_the_fold_glob(self):
        # Notebook 10 averages every convnext_fold*.pt it can see. An EMA
        # named convnext_fold0_ema.pt would silently enter the ensemble as an
        # extra member of a run already in it.
        joined = "\n".join(self.sources("08_finetune.ipynb"))
        assert 'AVERAGE_PREFIX = f"/kaggle/working/avg_fold{FOLD}"' in joined
        assert "convnext_fold{FOLD}_ema" not in joined

    def _checkpoint_identity(self):
        """The real function, lifted out of the generated notebook."""
        import os as _os
        import re as _re
        joined = "\n".join(self.sources("10_submit_ensemble.ipynb"))
        start = joined.index("def checkpoint_identity(")
        end = joined.index("_seen = {}")
        namespace = {"os": _os, "re": _re}
        exec(compile(joined[start:end], "<identity>", "exec"), namespace)
        return namespace["checkpoint_identity"]

    def test_one_fold_under_two_names_is_one_identity(self):
        # The standing trap: convnext_fold1.pt is a bit-exact export of
        # lit_fold1.ckpt, and BOTH match TUNED_PATTERNS. Averaging a fold with
        # itself gives it 2/6 of a five-fold ensemble instead of 1/5.
        identity = self._checkpoint_identity()
        same = ["convnext_fold1.pt", "lit_fold1.ckpt", "lit_fold1-v1.ckpt",
                "lit_fold1_old.ckpt"]
        assert len({identity(n) for n in same}) == 1
        assert identity("convnext_fold1.pt") == ("selected", "1")

    def test_different_folds_stay_distinct(self):
        identity = self._checkpoint_identity()
        folds = [f"convnext_fold{n}.pt" for n in range(5)]
        assert len({identity(n) for n in folds}) == 5

    def test_the_averages_are_their_own_families(self):
        identity = self._checkpoint_identity()
        assert identity("avg_fold1_ema.pt") == ("ema", "1")
        assert identity("avg_fold1_swa.pt") == ("swa", "1")
        # Same fold, three families -- none of them collide.
        assert len({identity("convnext_fold1.pt"), identity("avg_fold1_ema.pt"),
                    identity("avg_fold1_swa.pt")}) == 3

    def test_a_name_without_a_fold_number_only_collides_with_itself(self):
        identity = self._checkpoint_identity()
        assert identity("mystery.pt") != identity("other.pt")

    def test_the_submission_refuses_a_fold_under_two_names(self):
        joined = "\n".join(self.sources("10_submit_ensemble.ipynb"))
        assert "the same fold matched TUNED_PATTERNS under more than one name" \
            in joined

    def test_the_submission_refuses_to_mix_checkpoint_families(self):
        joined = "\n".join(self.sources("10_submit_ensemble.ipynb"))
        assert "more than one family matched TUNED_PATTERNS" in joined

    def test_the_dinov2_axis_is_not_still_wired_anywhere(self):
        # Closed at gold 0.820 against a 0.861 gate. A leftover ViT name in
        # either notebook trains or scores a trunk the project has retired.
        for name in ("08_finetune.ipynb", "10_submit_ensemble.ipynb"):
            for source in self.sources(name):
                for line in source.splitlines():
                    stripped = line.strip()
                    if stripped.startswith(("BACKBONE =", "TUNED_BACKBONE =")):
                        assert "convnext" in stripped, stripped

    def test_the_training_run_accumulates_selection_free_weights(self):
        joined = "\n".join(self.sources("08_finetune.ipynb"))
        assert "knee_lightning.WeightAverage(" in joined
        assert "swa_fraction" in joined
        # And the averages are read back off disk, not off the callback: under
        # ddp_notebook the object that accumulated them is in a forked child.
        assert "AVERAGES = {kind:" in joined

    def test_the_geometry_prose_matches_the_geometry_constants(self):
        """Notebook 08's markdown states the geometry it runs at.

        It drifted once: the constants reverted from 24 x 6 to 16 x 5 and one
        of three prose blocks was left behind, so the notebook told the reader
        "24 of 24 slices, 6 series, budget ~5.3 h" while the code ran 16 x 5.
        The run was correct and the documentation was not, which is the harder
        failure to notice -- nothing errors, and the numbers you plan around
        are wrong.
        """
        import json as _json
        import re as _re

        with open(os.path.join(ROOT, "notebooks", "08_finetune.ipynb")) as fh:
            cells = _json.load(fh)["cells"]
        code = "\n".join("".join(c["source"]) for c in cells
                          if c["cell_type"] == "code")
        prose = "\n".join("".join(c["source"]) for c in cells
                           if c["cell_type"] == "markdown")

        constants = {}
        for name in ("SLICES_PER_SERIES", "MAX_SERIES"):
            match = _re.search(rf"^{name} = (\d+)$", code, _re.M)
            assert match, f"{name} not found in the code cells"
            constants[name] = match.group(1)

        # The knobs table: "SLICES_PER_SERIES  16 of 24" / "MAX_SERIES  6 of ~5.5"
        for name, value in constants.items():
            row = _re.search(rf"^\s+{name}\s+(\d+) of ", prose, _re.M)
            assert row, f"{name} has no row in the knobs table"
            assert row.group(1) == value, (
                f"markdown says {name} is {row.group(1)}, the code says "
                f"{value} -- the notebook is documenting a run it does not do")

    def test_the_submission_switches_default_to_the_measured_behaviour(self):
        # Both change the submission's numbers, and every LB score on record
        # was produced with them off. They ship off so that turning one on is
        # a decision someone made, not a default they inherited.
        joined = "\n".join(self.sources("10_submit_ensemble.ipynb"))
        for switch in ("MEMBER_RANK_AVERAGE = False", "FLIP_TTA = False",
                       "SOUP = False"):
            assert switch in joined, switch

    def test_flip_tta_swaps_the_medial_lateral_findings_back(self):
        # A mirrored knee puts the medial compartment where the lateral one
        # was. TTA that averages the flipped prediction without permuting it
        # corrupts four of the twelve findings and raises nothing.
        joined = "\n".join(self.sources("10_submit_ensemble.ipynb"))
        assert "knee_augment.lateral_swap_order()" in joined
        assert "knee_augment.flip_volume(" in joined

    def test_members_are_combined_once_over_the_whole_test_set(self):
        # Rank-averaging needs every study's score for a member before it can
        # rank any of them, so tuned_predict must NOT average members itself.
        joined = "\n".join(self.sources("10_submit_ensemble.ipynb"))
        assert "averaging.rank_average(list(per_member))" in joined
        assert "transpose(1, 0, 2)" in joined

    def test_the_training_cell_names_its_missing_prerequisites(self):
        """A restarted kernel must be told what to run, not given a NameError.

        The five-fold sweep changes FOLD and re-runs from the training cell,
        which only works while the kernel still holds the cells above. A
        reclaimed Colab runtime loses them and the first symptom was
        `NameError: name 'make_loader' is not defined`, twenty lines into the
        cell, saying nothing about the cause.
        """
        joined = "\n".join(self.sources("08_finetune.ipynb"))
        start = joined.index("_required = {")
        cell = joined[start:joined.index("# The measurement run reached", start)]

        with pytest.raises(SystemExit) as raised:
            exec(compile(cell, "<guard>", "exec"), {})
        message = str(raised.value)
        for name in ("make_loader", "index", "FEATURE_DIM", "DEVICES"):
            assert name in message
        assert "Run all" in message

    def test_the_guard_passes_when_the_kernel_is_warm(self):
        joined = "\n".join(self.sources("08_finetune.ipynb"))
        start = joined.index("_required = {")
        cell = joined[start:joined.index("# The measurement run reached", start)]
        warm = {n: object() for n in
                ("DEVICES", "knee_lightning", "index", "make_loader",
                 "FEATURE_DIM")}
        exec(compile(cell, "<guard>", "exec"), warm)   # must not raise

    def test_the_warmup_is_an_absolute_step_count(self):
        joined = "\n".join(self.sources("08_finetune.ipynb"))
        assert "WARMUP_STEPS = " in joined
        assert "warmup_steps=WARMUP_STEPS" in joined
