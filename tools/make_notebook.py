"""Generate the Kaggle and Colab notebooks from the repo's own source.

Neither Kaggle nor Colab can import this repository, so the notebooks carry the
modules they need. Rather than reimplementing anything -- which is how a
train/test preprocessing skew gets introduced -- the generator embeds the real
module source and the notebook writes it to disk and imports it normally.

    python3 tools/make_notebook.py

Regenerate after editing anything under src/. tests/test_notebook.py fails if a
notebook is stale.
"""

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
NOTEBOOKS = ROOT / "notebooks"

# Modules the Kaggle notebooks need in order to build the cache.
KAGGLE_MODULES = [
    "src/__init__.py",
    "src/config.py",
    "src/data/__init__.py",
    "src/data/dicom_io.py",
    "src/data/cache.py",
    "src/data/index.py",
]


def read(path):
    return (ROOT / path).read_text()


def lines(text):
    """Split into nbformat source lines, each keeping its trailing newline.

    Jupyter concatenates `source` with no separator, so a plain split("\\n")
    produces a cell whose entire body is one line -- it looks correct in the
    JSON and fails at runtime with a SyntaxError.
    """
    return text.splitlines(keepends=True)


class Notebook:
    def __init__(self, gpu=False):
        self.cells = []
        self.gpu = gpu

    def md(self, text):
        self.cells.append({"cell_type": "markdown", "metadata": {},
                           "source": lines(text.strip())})

    def code(self, text):
        self.cells.append({"cell_type": "code", "metadata": {},
                           "execution_count": None, "outputs": [],
                           "source": lines(text.strip("\n"))})

    def write(self, path):
        metadata = {
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        }
        if self.gpu:
            metadata["accelerator"] = "GPU"
            metadata["colab"] = {"provenance": [], "gpuType": "A100"}

        notebook = {"cells": self.cells, "metadata": metadata,
                    "nbformat": 4, "nbformat_minor": 0}
        path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False))
        print(f"wrote {path.relative_to(ROOT)} ({len(self.cells)} cells)")


BOOTSTRAP_HEAD = """# Materialise the repo modules. Generated from source by
# tools/make_notebook.py -- do not edit here, edit src/ and regenerate.
import pathlib, sys

FILES = {
"""

BOOTSTRAP_TAIL = """
}

for path, source in FILES.items():
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source)

sys.path.insert(0, ".")
from src import config
from src.data import dicom_io, cache, index as index_mod

print("modules ready:", len(FILES), "files")
print("labels:", len(config.LABELS))
print("cache geometry:", config.CACHE_SLICES, "x", config.CACHE_SIZE)
print("data root:", config.DATA_ROOT)

# Fail here rather than several cells later. If the competition data is not
# attached there is nothing to detect, and the local fallback path then fails
# further on with a message that points at the wrong problem.
if not any((config.DATA_ROOT / m).exists() for m in ("train_series", "test_series")):
    root = pathlib.Path("/kaggle/input")
    available = sorted(p.name for p in root.iterdir()) if root.exists() else []
    raise SystemExit(
        "No competition data under " + str(config.DATA_ROOT) + "\\n"
        "Attached inputs: " + str(available) + "\\n"
        "Add it via '+ Add Input' in the sidebar, or set KNEE_DATA_ROOT."
    )

print("data OK:", sorted(p.name for p in config.DATA_ROOT.iterdir())[:8])
"""


def bootstrap_cell():
    """A cell that materialises the repo modules on a machine without the repo.

    Sources are embedded via repr() so quoting and escapes survive intact.
    Built by concatenation rather than an f-string: the embedded code contains
    braces and backslash escapes of its own, and interpolating it was how a
    newline escape became a literal line break in the generated notebook.
    """
    entries = ",\n".join(
        "    {0!r}: {1!r}".format(path, read(path)) for path in KAGGLE_MODULES
    )
    return BOOTSTRAP_HEAD + entries + BOOTSTRAP_TAIL


# ---------------------------------------------------------------------------
# 01 -- header scan
# ---------------------------------------------------------------------------

def build_index_notebook():
    nb = Notebook()

    nb.md("""
# 01 — Header Scan

Reads the DICOM headers of every file in the competition data and writes a
single parquet index. No pixel data is decoded, so this is bound by filesystem
traversal rather than by image decoding.

**Where:** Kaggle, with the competition dataset attached. The ~570 GB is only
mounted here.

**Runtime:** roughly 20–50 min for the train tree, a few seconds for test.

**Output:** `index_train.parquet` and `index_test.parquet` in `/kaggle/working`.
Save the notebook version, then publish the output as a dataset so the cache
build can attach it without rescanning.

The index is what makes the rest cheap: series can be filtered, audited, and
sorted afterwards without touching the imaging again.
""")

    nb.md("## Setup")
    nb.code(bootstrap_cell())

    nb.md("""
## Scan

Threads rather than processes: header reads release the GIL while waiting on
the filesystem, and a process pool would only add pickling overhead.

Set `LIMIT` to a few thousand for a smoke test before committing to the full run.
""")

    nb.code("""
import time
import pandas as pd

LIMIT = None      # e.g. 5000 to smoke-test first
WORKERS = 8

start = time.time()
index_train = index_mod.build_index(
    split="train_series",
    out_path="/kaggle/working/index_train.parquet",
    workers=WORKERS,
    limit=LIMIT,
)
print(f"train scan: {(time.time() - start) / 60:.1f} min")

index_test = index_mod.build_index(
    split="test_series",
    out_path="/kaggle/working/index_test.parquet",
    workers=WORKERS,
)
""")

    nb.md("""
## Audit

The two things that silently corrupt a volume are missing geometry and an
unresolvable knee side. Both are counted here rather than discovered later:
a series without position and orientation tags cannot be ordered reliably, and
a series whose side is unknown cannot be mirrored into a canonical orientation,
which matters because four of the twelve labels are compartment-specific.
""")

    nb.code("""
summary = index_mod.summarize(index_train)
summary.to_parquet("/kaggle/working/series_summary.parquet", index=False)

print(f"series: {len(summary)}")
print(f"studies: {summary['StudyInstanceUID'].nunique()}")
print()
print(f"missing geometry : {(~summary['geometry_ok']).sum()} series "
      f"({(~summary['geometry_ok']).mean():.1%})")
print(f"unknown side     : {summary['laterality'].isna().sum()} series "
      f"({summary['laterality'].isna().mean():.1%})")
print()
print("slice counts:")
print(summary["n_slices"].describe().to_string())
print()
print("laterality:")
print(summary["laterality"].value_counts(dropna=False).to_string())
print()
print("in-plane sizes:")
print(summary.groupby(["rows", "cols"]).size().sort_values(ascending=False).head(10).to_string())
""")

    nb.md("""
## Next

Publish `/kaggle/working` as a Kaggle dataset (`knee-index`). Notebook 02 attaches
it and builds the image cache without rescanning the DICOM tree.
""")

    nb.write(NOTEBOOKS / "01_scan_index.ipynb")


# ---------------------------------------------------------------------------
# 02 -- cache build
# ---------------------------------------------------------------------------

def build_cache_notebook():
    nb = Notebook()

    nb.md("""
# 02 — Build the Image Cache

Turns each series into a compact uint8 volume: sorted by geometry, intensity
normalised per volume, resampled to 224×224, 24 slices, and mirrored so every
knee faces the canonical side.

**Where:** Kaggle, with the competition data and the `knee-index` dataset from
notebook 01 attached.

**Why it exists:** at ~1.2 MB per series the whole training set is ~30 GB rather
than ~570 GB — small enough to publish as a Kaggle dataset and pull into Colab
once, after which training iterates without touching DICOM again.

### Sharding

`/kaggle/working` caps at 20 GB, so the run is split into shards of ~8,000
series (~9.6 GB each). Set `SHARD` to 0, then 1, then 2, saving a version each
time and publishing each output as a dataset part. Three runs cover the
training set.

**Runtime:** roughly 40–90 min per shard. Only the 24 selected slices of each
series are decoded, not every slice, which is the difference between decoding
~585k images and ~828k.
""")

    nb.md("## Setup")
    nb.code(bootstrap_cell())

    nb.md("""
## Configuration

`SHARD` is the only value to change between runs.
""")

    nb.code("""
import os

SHARD = 0            # 0, 1, 2 -- one per notebook run
N_SHARDS = 3
SPLIT = "train_series"   # switch to "test_series" for a single small run
WORKERS = 4

if SPLIT == "test_series":
    SHARD, N_SHARDS = 0, 1

# The index may sit in either place. /kaggle/working is where notebook 01 wrote
# it, which works when both notebooks are the same one; a separate notebook --
# including each of the parallel shard runs -- sees only the published dataset.
name = "index_test.parquet" if SPLIT == "test_series" else "index_train.parquet"
candidates = [
    f"/kaggle/input/knee-index/{name}",
    f"/kaggle/working/{name}",
]
INDEX_PATH = next((p for p in candidates if os.path.exists(p)), None)
if INDEX_PATH is None:
    raise SystemExit(
        f"No {name} found. Run notebook 01 first, then either stay in that "
        f"notebook or publish its output as a dataset named 'knee-index'.\\n"
        f"Looked in: {candidates}"
    )

OUT_DIR = "/kaggle/working"
print(f"shard {SHARD + 1} of {N_SHARDS}  split={SPLIT}")
print(f"index: {INDEX_PATH}")
""")

    nb.md("""
## Load the index and select this shard

Series are assigned to shards by a hash of the study UID rather than by
position, so every shard holds a representative mix of institutions and
protocols. A positional split would put whole scanners in one shard, and any
per-shard sanity check would then be measuring the site, not the pipeline.
""")

    nb.code("""
import hashlib
import pandas as pd

index = pd.read_parquet(INDEX_PATH)
print(f"index rows: {len(index)}")

groups = list(index.groupby("SeriesInstanceUID", sort=True))
print(f"series: {len(groups)}")


def shard_of(study_uid):
    digest = hashlib.md5(str(study_uid).encode()).hexdigest()
    return int(digest, 16) % N_SHARDS


selected = [
    (uid, group) for uid, group in groups
    if shard_of(group["StudyInstanceUID"].iloc[0]) == SHARD
]
print(f"series in this shard: {len(selected)}")
""")

    nb.md("""
## Build

Each series goes through `cache.build_series_volume`, the same function the
submission notebook uses at inference. Failures are recorded rather than raised:
one unreadable series must not end a 90-minute run, and the manifest records
which ones to investigate.
""")

    nb.code("""
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# cache.build_series_record is a module-level function on purpose. A worker
# defined in a notebook cell cannot be pickled by name, so a process pool
# fails under the spawn start method and survives under fork only by accident.
work = [(uid, group.to_dict("records")) for uid, group in selected]

volumes = np.zeros(
    (len(work), config.CACHE_SLICES, config.CACHE_SIZE, config.CACHE_SIZE),
    dtype=np.uint8,
)
records = []
failures = 0
start = time.time()

with ProcessPoolExecutor(max_workers=WORKERS) as pool:
    for position, (uid, volume, record) in enumerate(
            pool.map(cache.build_series_record, work, chunksize=8)):
        if volume is not None:
            volumes[position] = volume
        else:
            failures += 1
        record["position"] = position
        records.append(record)

        if position % 500 == 0 and position:
            rate = position / (time.time() - start)
            remaining = (len(work) - position) / rate / 60
            print(f"  {position}/{len(work)}  {rate:.1f}/s  "
                  f"~{remaining:.0f} min left  failures={failures}", flush=True)

elapsed = (time.time() - start) / 60
print(f"\\nbuilt {len(work) - failures}/{len(work)} in {elapsed:.1f} min")
""")

    nb.md("""
## Save

One packed array per shard rather than one file per series. Kaggle datasets
handle a few large files far better than 24,000 small ones, and training reads
sequentially from the pack.
""")

    nb.code("""
manifest = pd.DataFrame(records)

# Study-level propagation: one study is one knee, and the side tag is populated
# inconsistently within a study. Shards are assigned by study hash, so every
# series of a study is in this shard and propagation is complete here.
#
# This runs on the manifest rather than during the build because mirroring is
# applied at load time, not baked into the volumes -- so side detection can
# improve later without rebuilding 30 GB.
sides = {
    study: dict(zip(group["SeriesInstanceUID"], group["laterality"]))
    for study, group in manifest.groupby("StudyInstanceUID")
}
before = dicom_io.laterality_coverage(sides)
filled = dicom_io.resolve_study_laterality(sides)
lookup = {s: side for study in filled.values() for s, side in study.items()}
manifest["laterality_resolved"] = manifest["SeriesInstanceUID"].map(lookup)

print("laterality coverage")
print(f"  from tags alone : {before['known_before']}/{before['total_series']} "
      f"({before['known_before'] / before['total_series']:.1%})")
print(f"  after propagation: {before['known_after']}/{before['total_series']} "
      f"({before['known_after'] / before['total_series']:.1%})")
print(f"  recovered        : {before['recovered']} series")
print(f"  conflicting studies (left unresolved): {before['conflicting_studies']}")

volumes_path = f"{OUT_DIR}/cache_{SPLIT}_{SHARD:02d}.npy"
manifest_path = f"{OUT_DIR}/manifest_{SPLIT}_{SHARD:02d}.parquet"

np.save(volumes_path, volumes)
manifest.to_parquet(manifest_path, index=False)

size_gb = os.path.getsize(volumes_path) / 1e9
print(f"{volumes_path}  {volumes.shape}  {size_gb:.1f} GB")
print(f"{manifest_path}  {manifest.shape}")

if "error" in manifest.columns:
    failed = manifest[manifest["error"].notna()]
    if len(failed):
        print(f"\\n{len(failed)} failures:")
        print(failed["error"].value_counts().head().to_string())

if "laterality_resolved" in manifest.columns:
    print("\\nresolved side:")
    print(manifest["laterality_resolved"].value_counts(dropna=False).to_string())
    print("\\nVolumes are stored as acquired. The loader mirrors using this "
          "column, so improving side detection does not require a rebuild.")
""")

    nb.md("""
## Visual check

Slice ordering is the failure that produces no error and no warning, only a
subtly scrambled volume. A correct series reads as a continuous anatomical
sweep; a scrambled one visibly jitters between adjacent frames. Look at a few
before publishing 10 GB.
""")

    nb.code("""
import matplotlib.pyplot as plt

good = manifest[manifest["error"].isna()] if "error" in manifest.columns else manifest
sample = good.sample(min(3, len(good)), random_state=0)

for _, row in sample.iterrows():
    volume = volumes[int(row["position"])]
    figure, axes = plt.subplots(2, 6, figsize=(15, 5.5))
    for axis, slice_index in zip(axes.ravel(), range(0, config.CACHE_SLICES, 2)):
        axis.imshow(volume[slice_index], cmap="gray", vmin=0, vmax=255)
        axis.axis("off")
    figure.suptitle(
        f"{row['SeriesInstanceUID'][-16:]}  "
        f"side={row['laterality']} mirrored={row['mirrored']} "
        f"raw_slices={row['n_slices_raw']}")
    plt.tight_layout()
    plt.show()
""")

    nb.md("""
## Next

Publish `/kaggle/working` as a dataset part (`knee-cache-224-00`, `-01`, `-02`),
then set `SHARD = 1` and run again. Once all three exist, run once more with
`SPLIT = "test_series"` for the public test series.
""")

    nb.write(NOTEBOOKS / "02_build_cache.ipynb")


# ---------------------------------------------------------------------------
# 03 -- report label extraction (Colab)
# ---------------------------------------------------------------------------

def build_extract_notebook():
    nb = Notebook(gpu=True)

    nb.md("""
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

    nb.md("## 1. Check the GPU and pick a model size")

    nb.code("""
import subprocess, torch

# nvidia-smi is absent on a CPU runtime, which is exactly the case the
# message below is for -- so this must not raise first.
try:
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                          "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip())
except FileNotFoundError:
    print("nvidia-smi not found")

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

    nb.md("""
## 2. Install vLLM

Batched generation over ~13,000 prompts (4,349 reports × 3 samples). vLLM's
continuous batching is roughly an order of magnitude faster than a plain
`transformers` loop here.

Colab prompts to restart the runtime after this. **Restart, then re-run
section 1, skip this section, and continue from section 3.** The restart is
required: vLLM replaces torch and the CUDA libraries underneath a process that
has already imported the old ones.

The dependency-conflict wall of text pip prints here is noise. Every conflict is
against RAPIDS and Google SDK packages that this notebook never imports.
""")

    nb.code("""
!pip install -q vllm

# vLLM installs a PyTorch built against a newer CUDA than Colab's preinstalled
# torchaudio, and vLLM's import chain touches torchaudio -- which then raises
# "PyTorch and TorchAudio were compiled with different CUDA versions" at
# `from vllm import LLM`. Nothing here uses audio, so the mismatch is removed
# rather than reconciled.
!pip uninstall -y -q torchaudio

print("installed -- RESTART THE RUNTIME now, then re-run section 1 and skip this cell")
""")

    nb.md("""
## 3. Load `train.csv`

Either mount Drive (if you put the file there) or upload it directly. It is
~5.7 MB.
""")

    nb.code("""
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

    nb.md("""
## 4. Extraction logic

Generated from `src/labels/llm_extract.py` in the repo — **do not edit here**,
edit the module and regenerate, or the notebook and the repo will disagree
about how responses are parsed.
""")

    nb.code(read("src/labels/llm_extract.py").rstrip())

    nb.md("""
## 5. Load the model

`n=3` in the sampling parameters gives three independent samples per report in a
single pass. Majority vote decides each finding and the vote fraction becomes
the confidence weight used later during fusion — findings the model disagrees
with itself about should carry less weight.
""")

    nb.code("""
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

    nb.md("""
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

    nb.code("""
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

    nb.md("""
## 7. Full corpus

Only run this once the table above looks right. This is the long cell.
""")

    nb.code("""
import time

start = time.time()
mined = run_extraction(unlabelled["Report"].tolist(), "(4,349 reports)")
print(f"done in {(time.time() - start) / 60:.1f} min")
""")

    nb.md("""
## 8. Save and download

One row per study: `<finding>` (1/0/NaN), `<finding>_severity`,
`<finding>_confidence`, and the evidence span. The evidence is what makes a
disagreement between the two extractors auditable instead of mysterious, so it
is kept rather than discarded.
""")

    nb.code("""
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

    nb.md("""
## Next

Both parquet files go into `data/meta/`. Fusion then combines these with the
rule-based extractor: agreement becomes a hard target at full weight,
disagreement becomes a soft target at reduced weight, and silence scores zero
with a low sample weight rather than a fabricated prior.
""")

    nb.write(NOTEBOOKS / "03_extract_labels.ipynb")


def main():
    NOTEBOOKS.mkdir(exist_ok=True)
    build_index_notebook()
    build_cache_notebook()
    build_extract_notebook()


if __name__ == "__main__":
    main()
