"""Tests for the report-embedding artefact.

The encoder itself is a pretrained model and is not tested; these pin the
artefact's contract -- one row per report, float32, the ID column notebook 08
joins on -- using an injected fake encoder.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config  # noqa: E402
from src.labels import report_embed  # noqa: E402


def fake_encoder(texts):
    return np.stack([np.full(6, float(len(t))) for t in texts])


def make_train_csv(path):
    frame = pd.DataFrame({
        config.ID_COL: ["a", "b", "c"],
        "Report": ["short", "a longer report", None],
    })
    frame.to_csv(path, index=False)
    return path


class TestBuild:
    def test_one_row_per_report_and_none_for_missing(self, tmp_path):
        csv = make_train_csv(tmp_path / "train.csv")
        out = tmp_path / "report_embeddings.parquet"
        result = report_embed.build(csv, out, encoder=fake_encoder)

        written = pd.read_parquet(out)
        assert list(written[config.ID_COL]) == ["a", "b"]
        assert written.shape == (2, 7)  # id + 6 dims
        assert result.equals(written)

    def test_embeddings_are_float32_in_column_order(self, tmp_path):
        csv = make_train_csv(tmp_path / "train.csv")
        out = tmp_path / "report_embeddings.parquet"
        written = report_embed.build(csv, out, encoder=fake_encoder)

        matrix = written.drop(columns=[config.ID_COL])
        assert list(matrix.columns) == [f"e{i}" for i in range(6)]
        assert all(dtype == np.float32 for dtype in matrix.dtypes)
        assert matrix.iloc[0, 0] == len("short")

    def test_row_count_mismatch_raises(self, tmp_path):
        csv = make_train_csv(tmp_path / "train.csv")

        def broken(texts):
            return np.zeros((1, 6))

        try:
            report_embed.build(csv, tmp_path / "out.parquet", encoder=broken)
        except ValueError as exc:
            assert "2 reports" in str(exc)
        else:
            raise AssertionError("a short encoder output must raise")
