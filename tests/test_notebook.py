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
