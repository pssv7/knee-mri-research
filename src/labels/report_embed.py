"""Sentence embeddings of the radiology reports, for auxiliary supervision.

The pipeline distils each report into twelve soft targets and discards the
text. This module keeps a second, denser summary: one sentence embedding per
report, which the training loop regresses from the pooled study embedding as
an auxiliary loss. Reports exist for train studies and never for test, so
this is a training-time-only signal -- the auxiliary head is dropped at
export and the submission never sees it.

The reports are multilingual (Spanish, Dutch, German, ...), so the default
encoder is a multilingual one, and its 512-token window matters: reports are
long, and a 128-token encoder truncates away the findings that arrive last.
Competition rules forbid sending report text to hosted APIs; this runs the
model locally, and 4,349 reports take a few minutes on CPU.

Usage:  python -m src.labels.report_embed
Writes: data/meta/report_embeddings.parquet  (StudyInstanceUID + e0..eN)
"""

from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from src import config

DEFAULT_MODEL = "intfloat/multilingual-e5-small"
EMBEDDINGS_PARQUET = config.META_DIR / "report_embeddings.parquet"


def embed_texts(texts: Sequence[str], model_name: str = DEFAULT_MODEL,
                batch_size: int = 16, device: Optional[str] = None,
                max_length: int = 512) -> np.ndarray:
    """Mean-pooled, L2-normalised embeddings, (N, dim) float32.

    Imported lazily: transformers is a build-time dependency of this one
    artefact, not of the training pipeline that consumes the parquet.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.device_count() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    # E5 models are trained with an instruction prefix and score measurably
    # worse without one; "passage:" is the model card's choice for documents.
    prefix = "passage: " if "e5" in model_name.lower() else ""

    rows = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            chunk = [prefix + t for t in texts[start:start + batch_size]]
            encoded = tokenizer(chunk, padding=True, truncation=True,
                                max_length=max_length,
                                return_tensors="pt").to(device)
            hidden = model(**encoded).last_hidden_state
            attention = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * attention).sum(1) / attention.sum(1).clamp(min=1)
            rows.append(torch.nn.functional.normalize(pooled, dim=-1)
                        .cpu().float().numpy())
    return np.concatenate(rows)


def build(train_csv: Path = config.TRAIN_CSV,
          out: Path = EMBEDDINGS_PARQUET,
          encoder: Optional[Callable[[Sequence[str]], np.ndarray]] = None,
          **encoder_kwargs) -> pd.DataFrame:
    """Embed every report in train.csv and write the parquet.

    `encoder` is injectable so tests exercise the artefact's shape without
    downloading a model. Gold studies are included -- they never enter
    training, so an unused row is harmless and excluding them here would
    couple this file to the split logic.
    """
    frame = pd.read_csv(train_csv)
    frame = frame[frame["Report"].notna()].reset_index(drop=True)

    encode = encoder or (lambda texts: embed_texts(texts, **encoder_kwargs))
    matrix = np.asarray(encode(frame["Report"].tolist()), dtype=np.float32)
    if matrix.shape[0] != len(frame):
        raise ValueError(f"encoder returned {matrix.shape[0]} rows for "
                         f"{len(frame)} reports")

    columns = {config.ID_COL: frame[config.ID_COL]}
    columns.update({f"e{i}": matrix[:, i] for i in range(matrix.shape[1])})
    result = pd.DataFrame(columns)
    result.to_parquet(out, index=False)
    print(f"wrote {out}: {len(result)} studies x {matrix.shape[1]} dims")
    return result


if __name__ == "__main__":
    build()
