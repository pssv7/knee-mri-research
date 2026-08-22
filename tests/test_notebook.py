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
            source = cell_source(cell)
            if source.lstrip().startswith("!"):
                continue  # shell magic is not Python
            try:
                ast.parse(source)
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
        for name in ("build_prompt", "parse_response", "aggregate_samples",
                     "SYSTEM_PROMPT"):
            assert joined.count(name) >= 2, f"{name} defined but never used"
