"""Generate notebooks/03_extract_labels.ipynb, embedding llm_extract.py source."""

import json
import pathlib

ROOT = pathlib.Path("/Users/pssv/PycharmProjects/knee_competition")
MODULE_SRC = (ROOT / "src" / "labels" / "llm_extract.py").read_text()

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip().split("\n")})


def code(text):
    cells.append({
        "cell_type": "code", "metadata": {}, "execution_count": None,
        "outputs": [], "source": text.strip("\n").split("\n"),
    })


md("""
# Report Label Extraction — RSNA Knee

Mines the twelve findings from the 4,349 training studies that have a radiology
report but no radiologist labels. Those mined labels are the training signal for
the imaging model; only 58 studies carry real labels, and those are reserved for
validation.

**Runtime:** A100 recommended. Roughly 45–75 min for the full corpus at 3
self-consistency samples per report.

**Data handling:** the model runs locally on this Colab GPU. Report text is
never sent to a hosted API.

### What you do

1. `Runtime → Change runtime type → A100 GPU`
2. Run every cell top to bottom.
3. Cell 6 is a checkpoint: it scores the extractor on the 58 gold studies and
   prints a reliability table. **Read it before running the full corpus** — if
   macro AUC is not comfortably above 0.70, stop and report the numbers back
   rather than spending an hour.
4. The last cell downloads `llm_labels.parquet`. Drop that into `data/meta/`.
""")

md("## 1. Check the GPU and pick a model size")

code("""
import subprocess, torch

print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip())

if not torch.cuda.is_available():
    raise SystemExit("No GPU. Runtime -> Change runtime type -> A100 GPU.")

VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"VRAM: {VRAM_GB:.0f} GB")

# 14B needs ~28 GB in bf16; anything smaller gets the 7B.
if VRAM_GB >= 38:
    MODEL = "Qwen/Qwen2.5-14B-Instruct"
elif VRAM_GB >= 20:
    MODEL = "Qwen/Qwen2.5-7B-Instruct"
else:
    MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"

print("model:", MODEL)
""")

md("""
## 2. Install vLLM

Batched generation over ~13,000 prompts (4,349 reports × 3 samples). vLLM's
continuous batching is roughly an order of magnitude faster than a plain
`transformers` loop here.

Colab usually prompts to restart the runtime after this. **Restart, then skip
this cell and continue from cell 3.**
""")

code("""
!pip install -q vllm
print("installed -- restart the runtime if prompted, then continue from cell 3")
""")

md("""
## 3. Load `train.csv`

Either mount Drive (if you put the file there) or upload it directly. It is
~5.7 MB.
""")

code("""
import os
import pandas as pd

TRAIN_CSV = None

# Try Drive first, then fall back to an upload prompt.
for candidate in ["/content/drive/MyDrive/knee/train.csv", "/content/train.csv"]:
    if os.path.exists(candidate):
        TRAIN_CSV = candidate
        break

if TRAIN_CSV is None:
    from google.colab import files
    print("Select train.csv ...")
    uploaded = files.upload()
    TRAIN_CSV = list(uploaded.keys())[0]

train = pd.read_csv(TRAIN_CSV)
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]

gold = train[train["ACL"].notna()].reset_index(drop=True)
unlabelled = train[train["ACL"].isna()].reset_index(drop=True)
print(f"{len(train)} studies: {len(gold)} labelled, {len(unlabelled)} report-only")
assert len(gold) == 58, f"expected 58 gold studies, found {len(gold)}"
""")

md("""
## 4. Extraction logic

Generated from `src/labels/llm_extract.py` in the repo — **do not edit here**,
edit the module and regenerate, or the notebook and the repo will disagree
about how responses are parsed.
""")

code(MODULE_SRC.rstrip())

md("""
## 5. Load the model

`n=3` in the sampling parameters gives three independent samples per report in a
single pass. Majority vote decides each finding and the vote fraction becomes
the confidence weight used later during fusion — findings the model disagrees
with itself about should carry less weight.
""")

code("""
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

MAX_REPORT_CHARS = 6000   # longest report in the corpus is ~4,750
N_SAMPLES = 3

tokenizer = AutoTokenizer.from_pretrained(MODEL)
llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.90,
          max_model_len=6144, trust_remote_code=True)

sampling = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=1400,
                          n=N_SAMPLES)


def to_chat(report):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(str(report)[:MAX_REPORT_CHARS])},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)


def run_extraction(reports, desc=""):
    prompts = [to_chat(r) for r in reports]
    print(f"generating {len(prompts)} x {N_SAMPLES} samples {desc}...")
    outputs = llm.generate(prompts, sampling)

    results = []
    for output in outputs:
        samples = [parse_response(o.text) for o in output.outputs]
        results.append(aggregate_samples(samples))
    return results
""")

md("""
## 6. Checkpoint — score against the 58 gold studies

**Stop here and read the table.** This is the only honest measurement available.
The rule-based extractor already in the repo scores macro AUC 0.735, and its
weakest findings are the ones this model needs to carry:

| finding | lexicon AUC |
|---|---|
| Synovitis | 0.619 |
| PF OA | 0.646 |
| Contusion | 0.662 |
| Medial Meniscus | 0.678 |
| Lateral OA | 0.680 |

If the model lands below the lexicon on most findings, something is wrong with
the prompt or the parsing — report the table back rather than continuing.
""")

code("""
import numpy as np

gold_results = run_extraction(gold["Report"].tolist(), "(gold 58)")


def auc(y_true, scores):
    y_true = np.asarray(y_true); scores = np.asarray(scores, dtype=float)
    pos = y_true == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    unique, inverse, counts = np.unique(scores, return_inverse=True,
                                        return_counts=True)
    sums = np.zeros(len(unique)); np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


print(f"\\n{'finding':18s} {'n+':>3s} {'ment':>5s} {'prec':>5s} {'rec':>5s} {'AUC':>6s}")
print("-" * 48)
aucs = []
for name in LABELS:
    truth = gold[name].astype(int).to_numpy()
    present = np.array([r[name]["present"] if r[name]["present"] is not None else -1
                        for r in gold_results])
    severity = np.array([r[name]["severity"] if r[name]["severity"] is not None else 0.0
                         for r in gold_results])
    pred = (present == 1).astype(int)
    tp = int(((pred == 1) & (truth == 1)).sum())
    fp = int(((pred == 1) & (truth == 0)).sum())
    fn = int(((pred == 0) & (truth == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    a = auc(truth, severity); aucs.append(a)
    print(f"{name:18s} {truth.sum():3d} {(present != -1).mean():5.0%} "
          f"{prec:5.2f} {rec:5.2f} {a:6.3f}")
print("-" * 48)
print(f"macro AUC: {np.nanmean(aucs):.3f}   (lexicon baseline: 0.735)")
""")

md("""
## 7. Full corpus

Only run this once the table above looks right. This is the long cell.
""")

code("""
import time

start = time.time()
mined = run_extraction(unlabelled["Report"].tolist(), "(4,349 reports)")
print(f"done in {(time.time() - start) / 60:.1f} min")
""")

md("""
## 8. Save and download

One row per study: `<finding>` (1/0/NaN), `<finding>_severity`,
`<finding>_confidence`, and the evidence span. The evidence is what makes a
disagreement between the two extractors auditable instead of mysterious, so it
is kept rather than discarded.
""")

code("""
def to_frame(results, ids):
    rows = []
    for result in results:
        row = {}
        for name in LABELS:
            entry = result[name]
            row[name] = entry["present"]
            row[f"{name}_severity"] = entry["severity"]
            row[f"{name}_confidence"] = entry.get("confidence")
            row[f"{name}_evidence"] = entry.get("evidence")
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.insert(0, "StudyInstanceUID", list(ids))
    return frame


mined_frame = to_frame(mined, unlabelled["StudyInstanceUID"])
gold_frame = to_frame(gold_results, gold["StudyInstanceUID"])

mined_frame.to_parquet("llm_labels.parquet", index=False)
gold_frame.to_parquet("llm_labels_gold.parquet", index=False)

print(mined_frame.shape)
print("\\npositive rate per finding (mined):")
for name in LABELS:
    values = mined_frame[name]
    print(f"  {name:18s} pos {(values == 1).mean():5.1%}  "
          f"neg {(values == 0).mean():5.1%}  unclear {values.isna().mean():5.1%}")

from google.colab import files
files.download("llm_labels.parquet")
files.download("llm_labels_gold.parquet")
""")

md("""
## Next

Both parquet files go into `data/meta/`. Fusion then combines these with the
rule-based extractor: agreement becomes a hard target at full weight,
disagreement becomes a soft target at reduced weight, and silence is filled with
the per-finding `P(positive | unmentioned)` prior measured on the gold studies —
which ranges from 0.05 for MCL to 0.58 for Effusion, and is the reason silence
cannot simply be treated as a negative.
""")

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "A100"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = ROOT / "notebooks" / "03_extract_labels.ipynb"
out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False))
print(f"wrote {out} ({len(cells)} cells)")
