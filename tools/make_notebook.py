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

# The side classifier notebook needs the model package as well.
SIDE_MODULES = KAGGLE_MODULES + [
    "src/model/__init__.py",
    "src/model/side_classifier.py",
]

# The embedding and training notebooks need the whole model package.
MODEL_MODULES = KAGGLE_MODULES + [
    "src/model/__init__.py",
    "src/model/side_classifier.py",
    "src/model/embed.py",
    "src/model/dataset.py",
    "src/model/net.py",
    "src/model/splits.py",
    "src/labels/__init__.py",
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
import importlib, pathlib, sys

FILES = {
"""

BOOTSTRAP_TAIL = """
}

for path, source in FILES.items():
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source)

sys.path.insert(0, ".")

# Drop any previously imported copy before importing. Writing the files above
# does not affect a module already in sys.modules, so re-running this cell in a
# kernel that has already imported src/ silently keeps the OLD code -- the new
# file sits on disk unused, and the run looks fine until something calls a
# function that only exists in the new version.
for _name in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
    del sys.modules[_name]
importlib.invalidate_caches()

from src import config
from src.data import dicom_io, cache, index as index_mod

# Only some notebooks materialise the model package.
if pathlib.Path("src/model/side_classifier.py").exists():
    from src.model import side_classifier
if pathlib.Path("src/model/net.py").exists():
    from src.model import embed, dataset as knee_dataset, net, splits

print("modules ready:", len(FILES), "files")
print("labels:", len(config.LABELS))
print("cache geometry:", config.CACHE_SLICES, "x", config.CACHE_SIZE)
print("data root:", config.DATA_ROOT)

"""

REQUIRE_RAW_DICOM = """
# Fail here rather than several cells later. Only the scan and cache-build
# notebooks read raw DICOM; the ones that work from the cache do not, and
# demanding it there would block a notebook that has everything it needs.
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


def bootstrap_cell(modules=None, require_raw_dicom=True):
    """A cell that materialises the repo modules on a machine without the repo.

    Sources are embedded via repr() so quoting and escapes survive intact.
    Built by concatenation rather than an f-string: the embedded code contains
    braces and backslash escapes of its own, and interpolating it was how a
    newline escape became a literal line break in the generated notebook.
    """
    entries = ",\n".join(
        "    {0!r}: {1!r}".format(path, read(path))
        for path in (modules or KAGGLE_MODULES)
    )
    cell = BOOTSTRAP_HEAD + entries + BOOTSTRAP_TAIL
    if require_raw_dicom:
        cell += REQUIRE_RAW_DICOM
    return cell


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
# The index may be attached as a dataset, as another notebook's output, or
# sit in the working directory from notebook 01. Search rather than requiring
# a particular name -- attaching a notebook output gives it the notebook's
# name, which is not predictable.
import glob as _glob

name = "index_test.parquet" if SPLIT == "test_series" else "index_train.parquet"
# Attached notebook output nests several levels deep
# (/kaggle/input/notebooks/<user>/<notebook>/...), so search recursively --
# but skip the competition directory, which holds ~820,000 files and would
# take minutes to walk for a file that is never in there.
candidates = _glob.glob(f"/kaggle/working/{name}")
for _root in sorted(_glob.glob("/kaggle/input/*")):
    if os.path.basename(_root) == "competitions":
        continue
    candidates += _glob.glob(f"{_root}/**/{name}", recursive=True)
INDEX_PATH = candidates[0] if candidates else None
if INDEX_PATH is None:
    attached = sorted(os.listdir("/kaggle/input")) if os.path.exists("/kaggle/input") else []
    raise SystemExit(
        f"No {name} found in /kaggle/working or any attached input.\\n"
        f"Attached: {attached}\\n"
        "Run notebook 01, or attach its output via + Add Input."
    )
if len(candidates) > 1:
    print(f"note: {len(candidates)} copies found, using the first")

OUT_DIR = "/kaggle/working"
print(f"shard {SHARD + 1} of {N_SHARDS}  split={SPLIT}")
print(f"index: {INDEX_PATH}")

# Check the disk before the build, not after. /kaggle/working caps at 20 GB and
# a shard is ~9.7 GB, so a previous shard left in place makes the save fail --
# after the fifteen minutes of work, with the array stranded in memory.
import glob
import shutil

free_gb = shutil.disk_usage(OUT_DIR).free / 1e9
stale = sorted(glob.glob(f"{OUT_DIR}/cache_*.npy"))
print(f"free space: {free_gb:.1f} GB")
for path in stale:
    print(f"  existing shard: {path}  {os.path.getsize(path) / 1e9:.1f} GB")

if free_gb < 11.0:
    raise SystemExit(
        f"Only {free_gb:.1f} GB free; a shard needs ~10 GB plus headroom.\\n"
        "Publish the previous shard as a dataset, then remove its .npy from "
        f"{OUT_DIR}. Keep the index parquet -- it is small and needed here."
    )
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
from tqdm.auto import tqdm

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
    stream = pool.map(cache.build_series_record, work, chunksize=8)
    progress = tqdm(stream, total=len(work), unit="series",
                    desc=f"shard {SHARD}", smoothing=0.05)

    for position, (uid, volume, record) in enumerate(progress):
        if volume is not None:
            volumes[position] = volume
        else:
            failures += 1
        record["position"] = position
        records.append(record)

        if failures:
            progress.set_postfix(failures=failures)

elapsed = (time.time() - start) / 60
print(f"built {len(work) - failures}/{len(work)} in {elapsed:.1f} min")
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




# ---------------------------------------------------------------------------
# 04 -- left/right side classifier
# ---------------------------------------------------------------------------

def build_side_notebook():
    nb = Notebook(gpu=True)

    nb.md("""
# 04 — Left/Right Side Classifier

Predicts which knee a series images, for the 41.7% (10,161 of 24,371) whose
DICOM side tag is blank.

Four of the twelve findings are compartment-specific — medial and lateral
meniscus, medial and lateral OA — and the medial compartment sits on opposite
sides of the frame for a left versus a right knee. Without a side those four
labels cannot be placed.

Metadata is exhausted: `ImageLaterality` is never populated, `Laterality` is
present-but-empty on the majority of rows, series descriptions are anonymised
to `DummySeriesDesc!`, and study-level propagation recovered under 1% because
the tag is populated per study rather than per series. So the side is predicted
from the imaging.

**Trains on 14,210 series with a known side, predicts the remaining 10,161.**
Both come from the same cache, so this behaves identically at submission time.

**Where:** Kaggle with the cache datasets attached, GPU on. ~20 minutes.
""")

    nb.md("## Setup")
    # Works from the cache, so the raw DICOM tree is not required.
    nb.code(bootstrap_cell(SIDE_MODULES, require_raw_dicom=False))

    nb.md("""
## Load the cache

Attach all three `knee-cache-224-*` outputs. The manifests carry
`laterality_resolved`, which is the training label where present.
""")

    nb.code("""
import glob
import os

import numpy as np
import pandas as pd


def find_inputs(pattern):
    # Search attached inputs at any depth. A dataset mounts at
    # /kaggle/input/<name>/ but an attached notebook output nests under
    # /kaggle/input/notebooks/<user>/<notebook>/, so a fixed depth misses one
    # of them. The competition directory is skipped: it holds ~820,000 DICOM
    # files and is never where these live.
    found = sorted(glob.glob(f"/kaggle/working/{pattern}"))
    for root in sorted(glob.glob("/kaggle/input/*")):
        if os.path.basename(root) == "competitions":
            continue
        found += sorted(glob.glob(f"{root}/**/{pattern}", recursive=True))
    return found


cache_files = find_inputs("cache_train_series_*.npy")
manifest_files = find_inputs("manifest_train_series_*.parquet")

print("caches:")
for f in cache_files:
    print(" ", f)
if not cache_files:
    raise SystemExit("No cache shards attached. + Add Input -> your cache datasets.")

manifests = []
for path in manifest_files:
    frame = pd.read_parquet(path)
    frame["shard_file"] = path
    manifests.append(frame)
manifest = pd.concat(manifests, ignore_index=True)

print(f"\\nseries: {len(manifest)}")
print(manifest["laterality_resolved"].value_counts(dropna=False).to_string())
""")

    nb.md("""
## Memory-map the volumes

The shards total ~29 GB, well past RAM. `mmap_mode="r"` reads pages on demand,
so batches are pulled from disk without loading a shard whole.
""")

    nb.code("""
shards = {}
for path in cache_files:
    shards[path] = np.load(path, mmap_mode="r")
    print(path.split("/")[-1], shards[path].shape)

# Map each manifest row to (shard array, row index).
def locate(row):
    stem = row["shard_file"].split("/")[-1].replace("manifest_", "cache_").replace(".parquet", ".npy")
    for path in shards:
        if path.endswith(stem):
            return path
    return None

manifest["cache_path"] = manifest.apply(locate, axis=1)
missing = manifest["cache_path"].isna().sum()
if missing:
    print(f"WARNING: {missing} rows have no matching cache shard")
    manifest = manifest[manifest["cache_path"].notna()].reset_index(drop=True)
""")

    nb.md("""
## Splits

Grouped by study, never by series. Series from one study are near-duplicates of
each other — same knee, same session — so a random series split would put
near-copies on both sides and report an accuracy that does not exist.
""")

    nb.code("""
labelled = manifest[manifest["laterality_resolved"].notna()].reset_index(drop=True)
unlabelled = manifest[manifest["laterality_resolved"].isna()].reset_index(drop=True)

rng = np.random.default_rng(0)
studies = labelled["StudyInstanceUID"].unique()
rng.shuffle(studies)
cut = int(0.85 * len(studies))
train_studies = set(studies[:cut])

train_rows = labelled[labelled["StudyInstanceUID"].isin(train_studies)].reset_index(drop=True)
val_rows = labelled[~labelled["StudyInstanceUID"].isin(train_studies)].reset_index(drop=True)

print(f"train {len(train_rows)} series / {train_rows['StudyInstanceUID'].nunique()} studies")
print(f"val   {len(val_rows)} series / {val_rows['StudyInstanceUID'].nunique()} studies")
print(f"to predict: {len(unlabelled)} series")
print("\\nclass balance (train):")
print(train_rows["laterality_resolved"].value_counts(normalize=True).to_string())
""")

    nb.md("""
## Dataset

Slices go in as ordered channels. For sagittal series the medial-lateral axis
*is* the slice axis, so the side is encoded in the direction of travel through
the volume and in nothing else — a single slice cannot express it, and pooling
or sorting slices destroys it.

Mirroring is the natural augmentation here: flipping the input along its
medial-lateral axis flips the label. That doubles the data and stops the model
keying on any left/right imbalance in the source institutions.
""")

    nb.code("""
import torch
from torch.utils.data import Dataset, DataLoader

class SideDataset(Dataset):
    def __init__(self, rows, augment=False):
        self.rows = rows.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows.iloc[i]
        volume = np.asarray(shards[row["cache_path"]][int(row["position"])])
        x = side_classifier.prepare_batch(volume)[0]

        side = row.get("laterality_resolved")
        y = 1.0 if side == "R" else 0.0

        if self.augment and np.random.rand() < 0.5:
            # Mirror along the slice axis; label flips with it. Sagittal is the
            # majority plane and this is the axis that matters there.
            x = x[::-1].copy()
            y = 1.0 - y

        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


train_loader = DataLoader(SideDataset(train_rows, augment=True), batch_size=32,
                          shuffle=True, num_workers=2, drop_last=True)
val_loader = DataLoader(SideDataset(val_rows), batch_size=64, num_workers=2)
print("batches per epoch:", len(train_loader))
""")

    nb.md("## Train")

    nb.code("""
import time
from tqdm.auto import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
model = side_classifier.SideNet().to(device)
optimiser = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
criterion = torch.nn.BCEWithLogitsLoss()

EPOCHS = 6
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimiser, max_lr=1e-3, total_steps=EPOCHS * len(train_loader))

best = 0.0
for epoch in range(EPOCHS):
    model.train()
    for x, y in tqdm(train_loader, desc=f"epoch {epoch + 1}/{EPOCHS}"):
        x, y = x.to(device), y.to(device)
        optimiser.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimiser.step()
        scheduler.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in val_loader:
            probs = torch.sigmoid(model(x.to(device))).cpu().numpy()
            correct += int(((probs >= 0.5) == y.numpy().astype(bool)).sum())
            total += len(y)
    accuracy = correct / total
    print(f"  val accuracy {accuracy:.4f}")
    if accuracy > best:
        best = accuracy
        torch.save(model.state_dict(), "/kaggle/working/side_model.pt")

print(f"\\nbest validation accuracy: {best:.4f}")
""")

    nb.md("""
## Sanity check the failures

Before trusting the model, look at where it is wrong. If errors concentrate in
one acquisition plane — sagittal being the plausible one, since the cue there
is subtler — that is worth knowing rather than averaging away.
""")

    nb.code("""
model.load_state_dict(torch.load("/kaggle/working/side_model.pt"))
model.eval()

probs = []
with torch.no_grad():
    for x, _ in DataLoader(SideDataset(val_rows), batch_size=64, num_workers=2):
        probs.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
probs = np.concatenate(probs)

truth = (val_rows["laterality_resolved"] == "R").to_numpy().astype(int)
val_rows = val_rows.copy()
val_rows["prob_right"] = probs
val_rows["correct"] = (probs >= 0.5).astype(int) == truth

print(f"series accuracy: {val_rows['correct'].mean():.4f}")

votes = side_classifier.study_level_vote(probs, val_rows["StudyInstanceUID"].tolist())
study_truth = val_rows.groupby("StudyInstanceUID")["laterality_resolved"].first()
agree = sum(votes[s] == study_truth[s] for s in study_truth.index)
print(f"study accuracy after voting: {agree / len(study_truth):.4f}")

print("\\naccuracy by slice count:")
val_rows["slices"] = pd.cut(val_rows["n_slices_raw"], [0, 20, 30, 40, 1000])
print(val_rows.groupby("slices", observed=True)["correct"].agg(["mean", "size"]).to_string())
""")

    nb.md("""
## Predict the unknown side and save

Predictions are made per series, then voted per study — a study is one knee, so
its series must agree, and voting across roughly five of them is more reliable
than any single one.
""")

    nb.code("""
unknown_loader = DataLoader(SideDataset(unlabelled), batch_size=64, num_workers=2)
probs = []
with torch.no_grad():
    for x, _ in tqdm(unknown_loader, desc="predicting"):
        probs.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
probs = np.concatenate(probs)

votes = side_classifier.study_level_vote(probs, unlabelled["StudyInstanceUID"].tolist())

predictions = unlabelled[["StudyInstanceUID", "SeriesInstanceUID"]].copy()
predictions["prob_right"] = probs
predictions["side_predicted"] = predictions["StudyInstanceUID"].map(votes)
predictions["source"] = "model"

known = labelled[["StudyInstanceUID", "SeriesInstanceUID"]].copy()
known["prob_right"] = np.nan
known["side_predicted"] = labelled["laterality_resolved"].values
known["source"] = "tag"

out = pd.concat([known, predictions], ignore_index=True)
out.to_parquet("/kaggle/working/side_predictions.parquet", index=False)

print(out["source"].value_counts().to_string())
print(out["side_predicted"].value_counts().to_string())
print(f"\\nwrote side_predictions.parquet  {out.shape}")
""")

    nb.md("""
## Next

`side_predictions.parquet` goes into `data/meta/`. The training loader mirrors
each volume using `side_predicted`, along the axis chosen by acquisition plane —
slice order for sagittal, columns for coronal and axial.
""")

    nb.write(NOTEBOOKS / "04_side_classifier.ipynb")


# ---------------------------------------------------------------------------
# 05 -- frozen DINOv2 embeddings
# ---------------------------------------------------------------------------

def build_embed_notebook():
    nb = Notebook(gpu=True)

    nb.md("""
# 05 — Frozen DINOv2 Embeddings

Runs DINOv2-S once over every cached slice and stores the result, so the
aggregation stack afterwards trains on cached features in minutes instead of
hours.

**Why frozen.** The competition scores a second prize track on
`AUC + RuntimeSeconds / 32400`, so inference cost is scored rather than merely
capped. A frozen trunk is cheap at submission, and it makes the aggregation
design — which is where the modelling actually is — cheap to iterate on.

**Cost:** 24,371 series × 24 slices = ~585,000 forward passes of a 22M
parameter ViT. About 25–40 min on a T4, once.

**Output:** `embeddings_NN.npy` per shard, float16, 768 dims per slice.
~0.9 GB total.

Pretrained weights are permitted — confirmed in the competition's Code
Requirements — but internet is disabled at submission, so the weights are also
saved here for the inference notebook to load from disk.
""")

    nb.md("## Setup")
    nb.code(bootstrap_cell(MODEL_MODULES, require_raw_dicom=False))

    nb.code("""
import glob
import os

import numpy as np
import torch


def find_inputs(pattern):
    # Datasets mount at /kaggle/input/<name>/, attached notebook output at
    # /kaggle/input/notebooks/<user>/<notebook>/. Skip the competition tree:
    # ~820,000 DICOM files and never what we are looking for.
    found = sorted(glob.glob(f"/kaggle/working/{pattern}"))
    for root in sorted(glob.glob("/kaggle/input/*")):
        if os.path.basename(root) == "competitions":
            continue
        found += sorted(glob.glob(f"{root}/**/{pattern}", recursive=True))
    return found


cache_files = find_inputs("cache_*_series_*.npy")
print("caches:")
for f in cache_files:
    print(" ", f)
if not cache_files:
    raise SystemExit("No cache shards attached.")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
""")

    nb.md("""
## Load the encoder

Downloaded here with internet on, then saved to the working directory. The
submission notebook has no internet and loads the checkpoint from disk.
""")

    nb.code("""
encoder = embed.build_encoder(pretrained=True).to(device)
torch.save(encoder.state_dict(), "/kaggle/working/dinov2_s_reg4.pt")

print("model:", embed.MODEL_NAME)
print("params: %.1fM" % (sum(p.numel() for p in encoder.parameters()) / 1e6))
print("embedding dim:", embed.EMBED_DIM)
print("prefix tokens:", encoder.num_prefix_tokens, "(CLS + 4 registers)")
""")

    nb.md("""
## Embed

Each series contributes 24 slices, stacked 2.5D so every slice carries its
neighbours in the three channels a DINOv2 input expects. Stored as float16:
the precision is irrelevant to a downstream linear layer and it halves both
the file and the load time.
""")

    nb.code("""
import time
from tqdm.auto import tqdm

BATCH_SLICES = 256   # slices per forward pass

# 1 = CLS + mean patch (768 dims, 0.9 GB) -- the first model's setting.
# 4 = CLS + 4x4 pooled grid (6,528 dims, 7.6 GB) -- aimed at the focal
#     findings, which averaged 0.695 on gold against 0.865 for regional ones.
# 2 = a middle option at 1,920 dims and 2.3 GB.
embed.POOL_GRID = 4
DIM = embed.embed_dim()
print(f"pool grid {embed.POOL_GRID} -> {DIM} dims/slice "
      f"({24371 * 24 * DIM * 2 / 1e9:.1f} GB over the dataset)")

for path in cache_files:
    stem = os.path.basename(path).replace("cache_", "").replace(".npy", "")
    suffix = "" if embed.POOL_GRID == 1 else f"_g{embed.POOL_GRID}"
    out_path = f"/kaggle/working/embeddings_{stem}{suffix}.npy"
    if os.path.exists(out_path):
        print("skip (exists):", out_path)
        continue

    volumes = np.load(path, mmap_mode="r")
    count, depth = volumes.shape[0], volumes.shape[1]
    output = np.zeros((count, depth, DIM), dtype=np.float16)

    start = time.time()
    buffer, positions = [], []

    def flush():
        if not buffer:
            return
        batch = torch.from_numpy(np.concatenate(buffer)).to(device)
        features = embed.embed_slices(encoder, batch).float().cpu().numpy()
        cursor = 0
        for index, length in positions:
            output[index] = features[cursor:cursor + length].astype(np.float16)
            cursor += length
        buffer.clear()
        positions.clear()

    for i in tqdm(range(count), desc=os.path.basename(path), unit="series"):
        volume = torch.from_numpy(np.asarray(volumes[i]))[None]
        slices = net.stack_2p5d(volume).numpy()
        buffer.append(slices)
        positions.append((i, depth))
        if sum(len(b) for b in buffer) >= BATCH_SLICES:
            flush()
    flush()

    np.save(out_path, output)
    print(f"{out_path}  {output.shape}  "
          f"{os.path.getsize(out_path) / 1e9:.2f} GB  "
          f"{(time.time() - start) / 60:.1f} min")
""")

    nb.md("""
## Sanity check

Embeddings that are constant across series, or that contain non-finite values,
would train a model that appears to work and predicts nothing. Cheaper to catch
here than after a training run.
""")

    nb.code("""
files = sorted(glob.glob("/kaggle/working/embeddings_*.npy"))
for path in files:
    arr = np.load(path, mmap_mode="r")
    sample = np.asarray(arr[:200]).astype(np.float32)
    print(os.path.basename(path), arr.shape)
    print(f"   finite: {np.isfinite(sample).all()}  "
          f"mean {sample.mean():.3f}  std {sample.std():.3f}")
    per_series = sample.reshape(len(sample), -1)
    spread = per_series.std(axis=0).mean()
    print(f"   between-series spread: {spread:.4f}  "
          f"{'OK' if spread > 1e-3 else 'SUSPICIOUS -- embeddings barely vary'}")
""")

    nb.md("""
## Next

Publish `/kaggle/working` as `knee-embeddings`. Notebook 06 trains the
aggregation stack on these features, and the submission notebook loads
`dinov2_s_reg4.pt` to embed the test series with no internet.
""")

    nb.write(NOTEBOOKS / "05_embeddings.ipynb")


# ---------------------------------------------------------------------------
# 06 -- train the aggregation stack on frozen embeddings
# ---------------------------------------------------------------------------

def build_train_notebook():
    nb = Notebook(gpu=True)

    nb.md("""
# 06 — Train the Aggregation Stack

Trains slice pooling, FiLM conditioning, the series transformer, and the head
on the frozen DINOv2 embeddings. The trunk already ran, so this is minutes per
fold rather than hours.

**Inputs to attach**

| input | for |
|---|---|
| `knee-embeddings` | the slice features |
| the 3 cache datasets | manifests (which shard row is which series) |
| `knee-labels` | `train_labels.parquet`, `side_predictions.parquet` |
| competition data | `train.csv` (gold labels), `train_series.csv` (sequence metadata) |

**What to read at the end.** Two numbers, and they are not interchangeable.
The out-of-fold AUC is measured against mined labels, which share their noise
with the training targets — a model can score well there by reproducing the
extractor's mistakes. The gold-58 AUC is measured against radiologist labels
the model has never seen. When they disagree, trust the gold.
""")

    nb.md("## Setup")
    nb.code(bootstrap_cell(MODEL_MODULES, require_raw_dicom=False))

    nb.code("""
import glob
import os

import numpy as np
import pandas as pd
import torch


def find_inputs(pattern):
    # Recursive, because an attached notebook output nests under
    # /kaggle/input/notebooks/<user>/<notebook>/. The competition tree is
    # excluded: ~820,000 DICOM files, and recursing it takes minutes.
    found = sorted(glob.glob(f"/kaggle/working/{pattern}"))
    for root in sorted(glob.glob("/kaggle/input/*")):
        if os.path.basename(root) == "competitions":
            continue
        found += sorted(glob.glob(f"{root}/**/{pattern}", recursive=True))
    return found


def find_competition_file(name):
    # The competition CSVs sit at a known shallow depth, so this is matched
    # explicitly rather than by recursing 820,000 files to find a 6 MB csv.
    for candidate in (f"/kaggle/input/competitions/*/{name}",
                      f"/kaggle/input/*/{name}"):
        hits = sorted(glob.glob(candidate))
        if hits:
            return hits[0]
    raise SystemExit(f"missing competition file: {name}")


def find_one(pattern):
    hits = find_inputs(pattern)
    if not hits:
        raise SystemExit(f"missing input: {pattern}")
    return hits[0]


# Both the plain and the grid-pooled names; the newest wins if both exist.
embedding_files = find_inputs("embeddings_train_series_*.npy")
manifest_files = find_inputs("manifest_train_series_*.parquet")
print("embeddings:", [os.path.basename(f) for f in embedding_files])
print("manifests :", [os.path.basename(f) for f in manifest_files])
if not embedding_files:
    raise SystemExit("No embeddings attached. Run notebook 05 first.")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
""")

    nb.md("""
## Join the sources

The manifest says which row of which shard holds a series; the embeddings are
memory-mapped and read by that row. `cache_path` is rewritten to point at the
embedding file rather than the volume cache.
""")

    nb.code("""
shards = {}
manifests = []
for manifest_path in manifest_files:
    stem = os.path.basename(manifest_path).replace("manifest_", "").replace(".parquet", "")
    matching = [f for f in embedding_files if f.endswith(f"embeddings_{stem}.npy")]
    if not matching:
        print("no embeddings for", stem, "-- skipping")
        continue
    frame = pd.read_parquet(manifest_path)
    frame["cache_path"] = matching[0]
    shards[matching[0]] = np.load(matching[0], mmap_mode="r")
    manifests.append(frame)

manifest = pd.concat(manifests, ignore_index=True)
print(f"series: {len(manifest)}  studies: {manifest['StudyInstanceUID'].nunique()}")

labels = pd.read_parquet(find_one("train_labels.parquet"))
sides = pd.read_parquet(find_one("side_predictions.parquet"))
series_meta = pd.read_csv(find_competition_file("train_series.csv"))
train_csv = pd.read_csv(find_competition_file("train.csv"))

gold = train_csv[train_csv["ACL"].notna()].reset_index(drop=True)
print(f"mined labels: {len(labels)}  gold: {len(gold)}  sides: {len(sides)}")
""")

    nb.md("""
## Folds

Grouped by study and stratified on the number of positive findings. The gold
studies are not in `train_labels.parquet` at all, so they cannot leak — the
assertion is there because that guarantee should be enforced, not assumed.
""")

    nb.code("""
folded = splits.make_folds(labels)
gold_uids = set(gold["StudyInstanceUID"])
splits.assert_no_gold_leakage(folded["StudyInstanceUID"], gold_uids)

print(splits.summarise(folded).round(3).to_string())

index = knee_dataset.StudyIndex(manifest, series_meta, folded, sides)
gold_labels = gold[["StudyInstanceUID"] + config.LABELS].copy()
for name in config.LABELS:
    gold_labels[f"{name}_weight"] = 1.0
gold_index = knee_dataset.StudyIndex(manifest, series_meta, gold_labels, sides)
print(f"\\ntrainable studies: {len(index)}  gold studies: {len(gold_index)}")
""")

    nb.md("## Dataset and loaders")

    nb.code("""
from torch.utils.data import DataLoader, Dataset


class StudyFeatureDataset(Dataset):
    def __init__(self, study_index, studies):
        self.index = study_index
        self.studies = list(studies)

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, i):
        return knee_dataset.load_study_features(self.index, self.studies[i], shards)


def make_loader(study_index, studies, shuffle=False, batch_size=16):
    return DataLoader(
        StudyFeatureDataset(study_index, studies),
        batch_size=batch_size, shuffle=shuffle, num_workers=2,
        collate_fn=knee_dataset.collate_studies)


gold_loader = make_loader(gold_index, gold_index.studies)
print("gold batches:", len(gold_loader))
""")

    nb.md("## Train and evaluate")

    nb.code("""
import time
from tqdm.auto import tqdm

EPOCHS = 12
LR = 3e-4

# Width is read from the embeddings rather than assumed, so the same notebook
# trains on any pooling grid. Anything wider than the stack gets projected --
# running the aggregation at 6,528 would put ~769M parameters in it.
INPUT_DIM = int(next(iter(shards.values())).shape[-1])
FEATURE_DIM = 512 if INPUT_DIM > 1024 else INPUT_DIM
print(f"embedding width {INPUT_DIM} -> stack width {FEATURE_DIM}")


def auc(y_true, scores):
    y_true = np.asarray(y_true); scores = np.asarray(scores, dtype=float)
    pos = y_true == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    unique, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(unique)); np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def predict(model, loader):
    model.eval()
    ids, probs = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model.forward_features(
                batch["volumes"].to(device), batch["meta"].to(device),
                batch["mask"].to(device))
            probs.append(torch.sigmoid(logits).cpu().numpy())
            ids += batch[config.ID_COL]
    return ids, np.concatenate(probs)


def train_fold(train_studies, val_studies, seed=0):
    torch.manual_seed(seed)
    model = net.KneeNet(None, FEATURE_DIM, input_dim=INPUT_DIM).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    train_loader = make_loader(index, train_studies, shuffle=True)
    val_loader = make_loader(index, val_studies)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=LR * 3, total_steps=EPOCHS * len(train_loader))

    for _ in range(EPOCHS):
        model.train()
        for batch in train_loader:
            optimiser.zero_grad()
            logits = model.forward_features(
                batch["volumes"].to(device), batch["meta"].to(device),
                batch["mask"].to(device))
            loss = net.weighted_bce(logits, batch["targets"].to(device),
                                    batch["weights"].to(device))
            loss.backward()
            optimiser.step()
            scheduler.step()

    return model, val_loader


keyed = folded.set_index("StudyInstanceUID")
available = set(index.studies)
oof = {}
gold_probs = []

start = time.time()
for fold in tqdm(range(splits.N_FOLDS), desc="folds"):
    fold_ids = keyed["fold"]
    train_studies = [s for s in available if fold_ids.get(s, -1) != fold]
    val_studies = [s for s in available if fold_ids.get(s, -1) == fold]

    model, val_loader = train_fold(train_studies, val_studies, seed=fold)

    ids, probs = predict(model, val_loader)
    for study, row in zip(ids, probs):
        oof[study] = row

    gold_probs.append(predict(model, gold_loader)[1])
    torch.save(model.state_dict(), f"/kaggle/working/knee_fold{fold}.pt")

print(f"trained {splits.N_FOLDS} folds in {(time.time() - start) / 60:.1f} min")
""")

    nb.md("""
## Results

The gold column is the one that matters. The out-of-fold column is measured
against mined labels, so it shares a noise process with the training targets
and can reward a model for reproducing extraction errors.
""")

    nb.code("""
oof_ids = [s for s in index.studies if s in oof]
oof_matrix = np.stack([oof[s] for s in oof_ids])
oof_truth = folded.set_index("StudyInstanceUID").loc[oof_ids]

gold_matrix = np.mean(gold_probs, axis=0)
gold_ids, _ = predict(model, gold_loader)
gold_truth = gold.set_index("StudyInstanceUID").loc[gold_ids]

print(f"{'finding':18s} {'OOF (mined)':>12s} {'GOLD (real)':>12s}")
print("-" * 46)
oof_scores, gold_scores = [], []
for i, name in enumerate(config.LABELS):
    o = auc((oof_truth[name].to_numpy() > 0.5).astype(int), oof_matrix[:, i])
    g = auc(gold_truth[name].to_numpy().astype(int), gold_matrix[:, i])
    oof_scores.append(o); gold_scores.append(g)
    print(f"{name:18s} {o:12.3f} {g:12.3f}")
print("-" * 46)
print(f"{'macro':18s} {np.nanmean(oof_scores):12.3f} {np.nanmean(gold_scores):12.3f}")
print("\\nreference: fused label quality against gold was 0.854")

np.save("/kaggle/working/oof_predictions.npy", oof_matrix)
pd.DataFrame({"StudyInstanceUID": oof_ids}).to_parquet(
    "/kaggle/working/oof_ids.parquet", index=False)
""")

    nb.md("""
## Next

Publish the fold weights as `knee-model`. The submission notebook loads them
plus `dinov2_s_reg4.pt`, embeds the test series with no internet, and averages
the five folds.

If the gold macro AUC is well below the 0.854 of the labels themselves, the
ceiling is the model rather than the labels, and the next lever is the
backbone: a fine-tuned ConvNeXt-T on the same folds.
""")

    nb.write(NOTEBOOKS / "06_train.ipynb")


# ---------------------------------------------------------------------------
# 07 -- submission
# ---------------------------------------------------------------------------

def build_submit_notebook():
    nb = Notebook(gpu=True)

    nb.md("""
# 07 — Submission

The whole pipeline, offline, on whatever test set Kaggle mounts:

    DICOM -> volume -> side -> embedding -> 5-fold ensemble -> submission.csv

Every step imports the same modules used in training. Reimplementing any of
them here is how a train/test preprocessing skew gets introduced, and that
failure is invisible locally — it shows up only as a collapsed leaderboard
score.

**Inputs to attach**

| input | for |
|---|---|
| competition data | the test DICOMs |
| `knee_embeddings` | `dinov2_s_reg4.pt` — internet is off, timm cannot fetch |
| `knee_side_classifier` | `side_model.pt` |
| `knee-model` | the five fold checkpoints |
| `knee-labels` | training priors, used as the fallback |

**Processing is per study and streaming.** The public test set is 3 studies but
the hidden one is far larger, and holding every volume in memory would not
survive it. Each study is built, embedded, predicted, and discarded.
""")

    nb.md("## Setup")
    nb.code(bootstrap_cell(MODEL_MODULES, require_raw_dicom=True))

    nb.code("""
import glob
import os
import time

import numpy as np
import pandas as pd
import torch

START = time.time()

# Efficiency track: the score includes RuntimeSeconds / 32400, so inference
# cost is scored rather than merely capped. Lowering this trades accuracy for
# runtime and is the first knob to turn if the budget gets tight.
MAX_SERIES_PER_STUDY = None   # e.g. 5
BATCH_SLICES = 256

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)


def find_inputs(pattern):
    found = sorted(glob.glob(f"/kaggle/working/{pattern}"))
    for root in sorted(glob.glob("/kaggle/input/*")):
        if os.path.basename(root) == "competitions":
            continue
        found += sorted(glob.glob(f"{root}/**/{pattern}", recursive=True))
    return found


def find_one(pattern):
    hits = find_inputs(pattern)
    if not hits:
        raise SystemExit(f"missing input: {pattern}")
    return hits[0]


def find_competition_file(name):
    for candidate in (f"/kaggle/input/competitions/*/{name}",
                      f"/kaggle/input/*/{name}"):
        hits = sorted(glob.glob(candidate))
        if hits:
            return hits[0]
    raise SystemExit(f"missing competition file: {name}")
""")

    nb.md("""
## Load the models

All from disk. Nothing here may reach the network.
""")

    nb.code("""
encoder = embed.build_encoder(pretrained=False,
                              checkpoint_path=find_one("dinov2_s_reg4.pt")).to(device)

side_model = side_classifier.SideNet().to(device)
side_model.load_state_dict(torch.load(find_one("side_model.pt"), map_location=device))
side_model.eval()

fold_paths = find_inputs("knee_fold*.pt")
if not fold_paths:
    raise SystemExit("no fold checkpoints found -- publish notebook 06's output")

# The checkpoint carries the widths it was trained at; reading them from the
# state dict avoids a silent mismatch if the pooling grid changed.
state = torch.load(fold_paths[0], map_location="cpu")
INPUT_DIM = (state["project.weight"].shape[1]
             if "project.weight" in state else embed.EMBED_DIM)
FEATURE_DIM = state["head.2.weight"].shape[1]
embed.POOL_GRID = 1 if INPUT_DIM == embed.TOKEN_DIM * 2 else int(
    round(((INPUT_DIM / embed.TOKEN_DIM) - 1) ** 0.5))
print(f"checkpoints: input {INPUT_DIM}, stack {FEATURE_DIM}, "
      f"pool grid {embed.POOL_GRID}")

fold_models = []
for path in fold_paths:
    model = net.KneeNet(None, FEATURE_DIM, input_dim=INPUT_DIM).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    fold_models.append(model)

print(f"folds: {len(fold_models)}")

# Fallback used when a study cannot be processed at all. The training prior
# beats 0.5: it is the best constant prediction available for each finding.
try:
    train_labels = pd.read_parquet(find_one("train_labels.parquet"))
    PRIORS = {n: float(train_labels[n].mean()) for n in config.LABELS}
except SystemExit:
    PRIORS = {n: 0.5 for n in config.LABELS}
print("priors:", {k: round(v, 3) for k, v in list(PRIORS.items())[:4]}, "...")
""")

    nb.md("""
## Test studies

Read from `test.csv` rather than by listing directories, so the submission has
exactly the rows the grader expects even if a study's imaging is unreadable.
""")

    nb.code("""
test_csv = pd.read_csv(find_competition_file("test.csv"))
test_series_csv = pd.read_csv(find_competition_file("test_series.csv"))
test_root = config.DATA_ROOT / "test_series"

studies = test_csv[config.ID_COL].tolist()
series_meta = test_series_csv.set_index("SeriesInstanceUID")

print(f"studies: {len(studies)}  series: {len(test_series_csv)}")
print("root:", test_root)
""")

    nb.md("""
## Inference

One study at a time. A study that fails anywhere falls back to the training
prior rather than aborting the run: a notebook that crashes scores nothing,
while one that degrades on a few studies still scores.
""")

    nb.code("""
from tqdm.auto import tqdm


def study_series(study_uid):
    directory = test_root / study_uid
    if not directory.exists():
        return []
    return sorted(p for p in directory.iterdir() if p.is_dir())


def build_volume(series_dir):
    paths = dicom_io.series_paths(series_dir)
    if not paths:
        return None
    headers = [dicom_io.read_header(p) for p in paths]
    return cache.build_series_volume(headers)


def predict_sides(volumes):
    # Same slice-channel input the side classifier was trained on.
    batch = np.stack([side_classifier.prepare_batch(v)[0] for v in volumes])
    with torch.no_grad():
        probs = torch.sigmoid(side_model(torch.from_numpy(batch).to(device)))
    # One knee per study, so the series vote together.
    return "R" if float(probs.mean()) >= 0.5 else "L"


def embed_volumes(volumes):
    features = []
    for volume in volumes:
        slices = net.stack_2p5d(torch.from_numpy(volume)[None])
        chunks = []
        for start in range(0, len(slices), BATCH_SLICES):
            chunk = slices[start:start + BATCH_SLICES].to(device)
            chunks.append(embed.embed_slices(encoder, chunk).float().cpu())
        features.append(torch.cat(chunks))
    return torch.stack(features)


rows = []
failures = 0

for study in tqdm(studies, desc="studies", unit="study"):
    try:
        series_dirs = study_series(study)
        if MAX_SERIES_PER_STUDY and len(series_dirs) > MAX_SERIES_PER_STUDY:
            keep = np.linspace(0, len(series_dirs) - 1, MAX_SERIES_PER_STUDY)
            series_dirs = [series_dirs[i] for i in np.unique(keep.round().astype(int))]

        built, uids = [], []
        for series_dir in series_dirs:
            result = build_volume(series_dir)
            if result is not None:
                built.append(result["volume"])
                uids.append(series_dir.name)
        if not built:
            raise ValueError("no readable series")

        side = predict_sides(built)

        meta = []
        for uid in uids:
            row = series_meta.loc[uid].to_dict() if uid in series_meta.index else {}
            row["side_predicted"] = side
            meta.append(knee_dataset.series_metadata(pd.Series(row)))

        features = embed_volumes(built).unsqueeze(0).to(device)
        meta_tensor = torch.from_numpy(np.stack(meta)).unsqueeze(0).to(device)
        mask = torch.ones(1, len(built), dtype=torch.bool, device=device)

        with torch.no_grad():
            probs = np.mean([
                torch.sigmoid(m.forward_features(features, meta_tensor, mask))
                .cpu().numpy()[0] for m in fold_models], axis=0)

        rows.append({config.ID_COL: study,
                     **{n: float(p) for n, p in zip(config.LABELS, probs)}})

    except Exception as exc:  # noqa: BLE001
        failures += 1
        print(f"  fallback for {study[-12:]}: {str(exc)[:90]}")
        rows.append({config.ID_COL: study, **PRIORS})

elapsed = time.time() - START
print(f"\\n{len(rows)} studies in {elapsed / 60:.1f} min  ({failures} fallbacks)")
print(f"per study: {elapsed / max(len(rows), 1):.2f}s")
""")

    nb.md("""
## Write and verify

The header is checked against `sample_submission.csv` rather than trusted. The
column names carry an apostrophe and spaces, and a mismatch is rejected by the
grader after the run rather than during it.
""")

    nb.code("""
submission = pd.DataFrame(rows)[config.SUBMISSION_COLUMNS]
submission.to_csv("/kaggle/working/submission.csv", index=False)

sample = pd.read_csv(find_competition_file("sample_submission.csv"))
assert list(submission.columns) == list(sample.columns), (
    f"header mismatch\\n got: {list(submission.columns)}\\n"
    f"want: {list(sample.columns)}")
assert len(submission) == len(test_csv), "row count does not match test.csv"
assert submission[config.ID_COL].is_unique, "duplicate study ids"
assert submission[config.LABELS].notna().all().all(), "missing predictions"
values = submission[config.LABELS]
assert ((values >= 0) & (values <= 1)).all().all(), "predictions out of range"

print(submission.head().to_string())
print(f"\\nwrote submission.csv  {submission.shape}")

# Runtime against the 9-hour cap and the efficiency term.
projected = elapsed / max(len(rows), 1) * 1500
print(f"\\nprojected for 1,500 studies: {projected / 3600:.2f} h "
      f"(cap 9 h, efficiency penalty {projected / 32400:.3f})")
""")

    nb.md("""
## Next

Submit. Then compare the leaderboard score against the gold-58 estimate — a
large gap points at a train/test difference rather than at the model.

If runtime is comfortable, `MAX_SERIES_PER_STUDY` can stay `None`. If it is
tight, capping series is the cheapest lever, and the efficiency term makes it
worth measuring even when the budget is not at risk.
""")

    nb.write(NOTEBOOKS / "07_submit.ipynb")


# ---------------------------------------------------------------------------
# 08 -- end-to-end fine-tuning (Colab)
# ---------------------------------------------------------------------------

def build_finetune_notebook():
    nb = Notebook(gpu=True)

    nb.md("""
# 08 — Fine-Tuned Backbone (Colab)

Trains a ConvNeXt-T end to end instead of reading frozen DINOv2 features. Prior
RSNA competitions were won by fine-tuned ImageNet CNNs, so this is the arm most
likely to top the leaderboard — and the one the frozen path exists to be
measured against.

**This run measures before it commits.** Fine-tuning here is expensive: one
study is ~5.5 series × 24 slices ≈ 130 slice-forwards, so a full epoch over
4,349 studies is ~570,000. The first cell after training reports actual
throughput and projects the cost of the full sweep. Read that before launching
five folds.

**Subsampling is how the cost is controlled**, and both knobs are honest
trade-offs rather than free wins:

    SLICES_PER_SERIES   8 of 24    the tear may fall in a dropped slice
    MAX_SERIES          3 of ~5.5  a finding may live only in a dropped series

**Mirroring is used here**, unlike the frozen path. Raw volumes can be
reflected, so left knees are mapped onto the canonical side along the axis the
acquisition plane dictates. That makes this run an A/B against the frozen
path's covariate approach as well as a backbone comparison — two variables at
once, which is worth remembering when reading the result.
""")

    nb.md("""
## 1. Setup

Colab, A100 recommended. The cache is pulled from your Kaggle notebook outputs,
so an API token is needed.
""")

    nb.code("""
!pip install -q timm kaggle

import os, pathlib, subprocess, torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
if not torch.cuda.is_available():
    raise SystemExit("Runtime -> Change runtime type -> A100 GPU")
""")

    nb.md("""
## 2. Kaggle credentials

Upload `~/.kaggle/access_token` (or `kaggle.json`) from the machine where the
Kaggle CLI already works. Nothing is printed.
""")

    nb.code("""
from google.colab import files

target = pathlib.Path.home() / ".kaggle"
target.mkdir(exist_ok=True)

if not (target / "access_token").exists() and not (target / "kaggle.json").exists():
    print("Select access_token or kaggle.json ...")
    uploaded = files.upload()
    for name, content in uploaded.items():
        destination = target / ("kaggle.json" if name.endswith(".json") else "access_token")
        destination.write_bytes(content)
        destination.chmod(0o600)

print("credentials:", sorted(p.name for p in target.iterdir()))
""")

    nb.md("""
## 3. Pull the data

One cache shard only — roughly 1,460 studies, which is plenty to measure
throughput and get a first fold. Downloading all three would cost 29 GB before
a single step of training.

Replace the notebook slugs with yours if they differ.
""")

    nb.code("""
USER = "pssv27"
# Slugs, not titles. Kaggle lowercases the title and turns non-alphanumerics
# into hyphens, so "knee-cache-224-01_shard-1" is stored as
# "knee-cache-224-01-shard-1". Confirm with the listing printed above.
CACHE_NOTEBOOK = f"{USER}/knee-cache-224-01-shard-1"
SIDE_NOTEBOOK = f"{USER}/knee-side-classifier"
COMPETITION = "rsna-knee-abnormality-detection"

# Your notebook slugs, exactly as Kaggle stores them. Kaggle derives a slug
# from the title by lowercasing and replacing non-alphanumerics with hyphens,
# so an underscore in the title is a hyphen in the slug.
print("your notebooks:")
subprocess.run(["kaggle", "kernels", "list", "--mine"])


def fetch(slug, path):
    path = pathlib.Path(path)
    if path.exists() and any(path.iterdir()):
        print("already present:", path)
        return
    path.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(["kaggle", "kernels", "output", slug, "-p", str(path)],
                            capture_output=True, text=True)
    if result.returncode != 0:
        # Surface the real message. check=True raises CalledProcessError with
        # the output discarded, which says only that something failed.
        raise SystemExit(
            f"failed to fetch {slug}\\n"
            f"stdout: {result.stdout.strip()}\\n"
            f"stderr: {result.stderr.strip()}\\n"
            "Check the slug against the listing above.")
    print(result.stdout.strip()[-300:])

fetch(CACHE_NOTEBOOK, "data/cache")
fetch(SIDE_NOTEBOOK, "data/side")

for name in ("train.csv", "train_series.csv"):
    if not pathlib.Path(f"data/{name}").exists():
        subprocess.run(["kaggle", "competitions", "download", COMPETITION,
                        "-f", name, "-p", "data", "-q"], check=True)

print(sorted(os.listdir("data/cache")))
""")

    nb.md("""
## 4. Repo modules

Generated from `src/` — edit the modules and regenerate rather than editing
here, or this notebook and the repo will disagree about preprocessing.
""")

    nb.code(bootstrap_cell(MODEL_MODULES, require_raw_dicom=False))

    nb.md("## 5. Assemble")

    nb.code("""
import glob
import numpy as np
import pandas as pd

# train_labels.parquet is built locally by src/labels/fuse.py -- upload it if
# this cell cannot find it.
label_paths = glob.glob("data/**/train_labels.parquet", recursive=True)
if not label_paths:
    print("Select train_labels.parquet ...")
    uploaded = files.upload()
    label_paths = [list(uploaded.keys())[0]]

labels = pd.read_parquet(label_paths[0])
sides = pd.read_parquet(glob.glob("data/**/side_predictions.parquet", recursive=True)[0])
series_meta = pd.read_csv("data/train_series.csv")
train_csv = pd.read_csv("data/train.csv")
gold = train_csv[train_csv["ACL"].notna()].reset_index(drop=True)

cache_path = glob.glob("data/cache/cache_train_series_*.npy")[0]
manifest = pd.read_parquet(glob.glob("data/cache/manifest_train_series_*.parquet")[0])
manifest["cache_path"] = cache_path
shards = {cache_path: np.load(cache_path, mmap_mode="r")}
print("volumes:", shards[cache_path].shape)

folded = splits.make_folds(labels)
splits.assert_no_gold_leakage(folded["StudyInstanceUID"], set(gold["StudyInstanceUID"]))

index = knee_dataset.StudyIndex(manifest, series_meta, folded, sides, max_series=3)
gold_labels = gold[["StudyInstanceUID"] + config.LABELS].copy()
for name in config.LABELS:
    gold_labels[f"{name}_weight"] = 1.0
gold_index = knee_dataset.StudyIndex(manifest, series_meta, gold_labels, sides,
                                     max_series=3)
print(f"studies in this shard: {len(index)}  gold present: {len(gold_index)}")
""")

    nb.md("""
## 6. Dataset

Slices are subsampled evenly rather than at random, so a validation score is
reproducible. Horizontal flips are **not** used as augmentation: after
canonicalisation a flip re-mirrors the knee and swaps medial with lateral,
silently corrupting four of the twelve labels.
""")

    nb.code("""
import torch
from torch.utils.data import DataLoader, Dataset

SLICES_PER_SERIES = 8
BATCH_STUDIES = 4
ACCUM_STEPS = 2


class StudyVolumeDataset(Dataset):
    def __init__(self, study_index, studies, augment=False):
        self.index = study_index
        self.studies = list(studies)
        self.augment = augment
        self.keep = np.linspace(0, config.CACHE_SLICES - 1,
                                SLICES_PER_SERIES).round().astype(int)

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, i):
        item = knee_dataset.load_study(self.index, self.studies[i], shards)
        item["volumes"] = item["volumes"][:, self.keep]

        if self.augment:
            # Intensity jitter only. Geometric flips would swap medial and
            # lateral; rotation is safe but left out of this first measurement.
            scale = np.random.uniform(0.9, 1.1)
            item["volumes"] = np.clip(
                item["volumes"].astype(np.float32) * scale, 0, 255).astype(np.uint8)
        return item


def make_loader(study_index, studies, shuffle=False, augment=False):
    return DataLoader(
        StudyVolumeDataset(study_index, studies, augment),
        batch_size=BATCH_STUDIES, shuffle=shuffle, num_workers=2,
        collate_fn=knee_dataset.collate_studies, pin_memory=True)
""")

    nb.md("""
## 7. Model

ConvNeXt-T with ImageNet weights, which the competition permits. The
aggregation above it is the same `KneeNet` the frozen path uses, so the
comparison is between backbones rather than between architectures.
""")

    nb.code("""
import timm

BACKBONE = "convnext_tiny.fb_in22k_ft_in1k"
backbone = timm.create_model(BACKBONE, pretrained=True, num_classes=0, in_chans=3)
FEATURE_DIM = backbone.num_features
print(f"{BACKBONE}: {FEATURE_DIM} features, "
      f"{sum(p.numel() for p in backbone.parameters()) / 1e6:.1f}M params")

device = "cuda"
model = net.KneeNet(backbone, FEATURE_DIM).to(device)
print(f"total: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
""")

    nb.md("""
## 8. Train one fold, and measure

The projection printed at the end is the point of this run. Do not launch five
folds before reading it.
""")

    nb.code("""
import time
from tqdm.auto import tqdm

EPOCHS = 4
LR = 1e-4

keyed = folded.set_index("StudyInstanceUID")
train_studies = [s for s in index.studies if keyed.loc[s, "fold"] != 0]
val_studies = [s for s in index.studies if keyed.loc[s, "fold"] == 0]
train_loader = make_loader(index, train_studies, shuffle=True, augment=True)
val_loader = make_loader(index, val_studies)
gold_loader = make_loader(gold_index, gold_index.studies)
print(f"train {len(train_studies)}  val {len(val_studies)}  gold {len(gold_index)}")

optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scaler = torch.amp.GradScaler("cuda")
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimiser, max_lr=LR * 3,
    total_steps=EPOCHS * (len(train_loader) // ACCUM_STEPS + 1))

epoch_times = []
for epoch in range(EPOCHS):
    model.train()
    start = time.time()
    optimiser.zero_grad()

    for step, batch in enumerate(tqdm(train_loader, desc=f"epoch {epoch + 1}/{EPOCHS}")):
        with torch.amp.autocast("cuda"):
            logits = model(batch["volumes"].to(device, non_blocking=True),
                           batch["meta"].to(device), batch["mask"].to(device))
            loss = net.weighted_bce(logits, batch["targets"].to(device),
                                    batch["weights"].to(device)) / ACCUM_STEPS
        scaler.scale(loss).backward()

        if (step + 1) % ACCUM_STEPS == 0:
            scaler.step(optimiser)
            scaler.update()
            optimiser.zero_grad()
            scheduler.step()

    epoch_times.append(time.time() - start)
    print(f"  epoch {epoch + 1}: {epoch_times[-1] / 60:.1f} min")

torch.save(model.state_dict(), "convnext_fold0.pt")

per_epoch = np.mean(epoch_times)
share = len(index) / 4349
full_epoch = per_epoch / share
print(f"\\nthis shard is {share:.0%} of the training set")
print(f"projected full epoch: {full_epoch / 60:.1f} min")
print(f"projected 5 folds x {EPOCHS} epochs: {full_epoch * EPOCHS * 5 / 3600:.1f} h")
""")

    nb.md("""
## 9. Score

Against the gold studies present in this shard. There are only about a third of
the 58 here, so the estimate is noisier than the frozen path's — treat it as a
direction rather than a number.
""")

    nb.code("""
def auc(y_true, scores):
    y_true = np.asarray(y_true); scores = np.asarray(scores, dtype=float)
    pos = y_true == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float); ranks[order] = np.arange(1, len(scores) + 1)
    unique, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(unique)); np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def predict(loader):
    model.eval(); ids, probs = [], []
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for batch in loader:
            logits = model(batch["volumes"].to(device), batch["meta"].to(device),
                           batch["mask"].to(device))
            probs.append(torch.sigmoid(logits.float()).cpu().numpy())
            ids += batch[config.ID_COL]
    return ids, np.concatenate(probs)


gold_ids, gold_probs = predict(gold_loader)
truth = gold.set_index("StudyInstanceUID").loc[gold_ids]

print(f"{'finding':18s} {'gold AUC':>9s}  {'frozen DINOv2':>13s}")
print("-" * 44)
FROZEN = dict(zip(config.LABELS,
                  [0.728, 0.671, 0.698, 0.680, 0.895, 0.899,
                   0.772, 0.897, 0.746, 0.862, 0.711, 0.683]))
scores = []
for i, name in enumerate(config.LABELS):
    value = auc(truth[name].to_numpy().astype(int), gold_probs[:, i])
    scores.append(value)
    print(f"{name:18s} {value:9.3f}  {FROZEN[name]:13.3f}")
print("-" * 44)
print(f"{'macro':18s} {np.nanmean(scores):9.3f}  {np.mean(list(FROZEN.values())):13.3f}")
""")

    nb.md("""
## Next

Two things decide what follows.

**The runtime projection.** If five folds fit comfortably, run them and compare
against the frozen path properly. If not, the levers are `SLICES_PER_SERIES`,
`MAX_SERIES`, fewer folds, or freezing the backbone's early stages.

**The focal findings.** Fine-tuning is expected to help meniscus, ligaments,
and fracture most — those are the ones frozen natural-image features handle
worst. If they do not move, the limit is elsewhere: the mined labels, or the
224px resolution relative to the size of the structures.
""")

    nb.write(NOTEBOOKS / "08_finetune.ipynb")


def main():
    NOTEBOOKS.mkdir(exist_ok=True)
    build_index_notebook()
    build_cache_notebook()
    build_extract_notebook()
    build_side_notebook()
    build_embed_notebook()
    build_train_notebook()
    build_submit_notebook()
    build_finetune_notebook()


if __name__ == "__main__":
    main()
