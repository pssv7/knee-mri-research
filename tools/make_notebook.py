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

# Only notebook 08 trains, and only it needs Lightning. The module imports
# lightning at import time, which the submission notebook has no reason to
# require.
FINETUNE_MODULES = MODEL_MODULES + ["src/model/lightning.py"]


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
# Only the fine-tuning notebook materialises this one; it imports lightning at
# module scope, which nothing else needs.
if pathlib.Path("src/model/lightning.py").exists():
    from src.model import lightning as knee_lightning

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

# AWQ 4-bit puts the 32B at ~19 GB of weights, so it fits an A100 40GB or an
# L4 24GB and replaces the 14B-bf16 that used to top this ladder. The gold
# checkpoint is the arbiter: compare its per-finding AUCs against the 14B run
# before spending an hour of corpus extraction on a bigger model.
if VRAM_GB >= 22:
    MODEL = "Qwen/Qwen2.5-32B-Instruct-AWQ"
elif VRAM_GB >= 15:
    MODEL = "Qwen/Qwen2.5-14B-Instruct-AWQ"
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

**The `ERROR: pip's dependency resolver ...` wall of text here is noise, and
the install it follows succeeded** — pip prints that report at ERROR level
after resolving, not instead of installing. Every conflict is against
packages this notebook never imports: RAPIDS (`cudf`, `cuml`, `raft`, `rmm`,
`cuvs`, `dask-cuda`, `numba-cuda`, and their `cuda-toolkit` / `cuda-python`
pins, which vLLM's newer CUDA wheels invalidate), the Google SDKs
(`google-adk`, `google-ai-generativelanguage`, `grpcio-status` over
protobuf), and `ipython`'s optional `jedi` — that last one costs tab
completion and `obj?` introspection, nothing this notebook runs.

The cell prints a banner with the installed vLLM version underneath the
wall. **If that banner is missing, the install genuinely failed** — that is
the one case to stop and report; a conflict list above a banner is not.
""")

    nb.code("""
!pip install -q vllm

# vLLM installs a PyTorch built against a newer CUDA than Colab's preinstalled
# torchaudio, and vLLM's import chain touches torchaudio -- which then raises
# "PyTorch and TorchAudio were compiled with different CUDA versions" at
# `from vllm import LLM`. Nothing here uses audio, so the mismatch is removed
# rather than reconciled.
!pip uninstall -y -q torchaudio

# pip's conflict report is long enough to push the restart instruction off
# screen, and it reads like a failure while being none -- so confirm the
# install positively and say so louder than the noise above. Reading the
# version from package metadata does not import vllm, which must not happen
# before the restart.
from importlib.metadata import PackageNotFoundError, version

try:
    installed = version("vllm")
except PackageNotFoundError:
    installed = None

print()
print("=" * 68)
if installed:
    print(f"  vLLM {installed} installed. Conflicts printed above are noise.")
    print("  RESTART THE RUNTIME NOW, then re-run section 1, skip this")
    print("  cell, and continue from section 3.")
else:
    print("  vLLM did NOT install. Stop here and report the pip output --")
    print("  the conflict list above is normally harmless, so a missing")
    print("  version means something else failed.")
print("=" * 68)
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
from transformers import AutoTokenizer

MAX_REPORT_CHARS = 6000   # longest report in the corpus is ~4,750
N_SAMPLES = 3
MAX_NEW_TOKENS = 512      # a probability JSON is short; the reason_first
                          # variant spends ~100 extra tokens of prose first

tokenizer = AutoTokenizer.from_pretrained(MODEL)

# vLLM is worth it for 4,349 reports and overkill for the 58-report gold
# checkpoint -- which is the loop we want to run repeatedly while iterating the
# prompt. Its compiled extension is also tied to a specific CUDA runtime and
# breaks whenever Colab's torch shifts under it, so a transformers fallback
# keeps the fast loop available even when vLLM will not import.
USE_VLLM = True
try:
    from vllm import LLM, SamplingParams
except Exception as exc:  # noqa: BLE001
    print("vLLM unavailable, falling back to transformers:", str(exc)[:120])
    USE_VLLM = False

if USE_VLLM:
    # dtype="auto" reads the checkpoint config: AWQ models declare float16,
    # which their kernels require; full-precision Qwen declares bfloat16.
    llm = LLM(model=MODEL, dtype="auto", gpu_memory_utilization=0.90,
              max_model_len=6144, trust_remote_code=True)
else:
    import torch
    from transformers import AutoModelForCausalLM
    if "AWQ" in MODEL:
        # transformers needs the autoawq kernels for AWQ checkpoints; vLLM
        # ships its own, so this only matters on the fallback path.
        import importlib.util, subprocess, sys
        if importlib.util.find_spec("awq") is None:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                            "autoawq"], check=False)
    dtype = torch.float16 if "AWQ" in MODEL else torch.bfloat16
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=dtype, device_map="auto")
    hf_model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"


def to_chat(report, variant="baseline"):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",
         "content": build_probability_prompt(str(report)[:MAX_REPORT_CHARS],
                                             variant=variant)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)


def _generate_hf(prompts, n_samples, batch_size=8, max_new_tokens=None):
    import torch
    from tqdm.auto import tqdm

    texts = []
    for start in tqdm(range(0, len(prompts), batch_size), desc="generating"):
        chunk = prompts[start:start + batch_size]
        encoded = tokenizer(chunk, return_tensors="pt", padding=True,
                            truncation=True, max_length=6144).to(hf_model.device)
        with torch.no_grad():
            out = hf_model.generate(
                **encoded, max_new_tokens=max_new_tokens or MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7, top_p=0.9, num_return_sequences=n_samples,
                pad_token_id=tokenizer.pad_token_id)
        # Strip the prompt so only the completion is parsed.
        generated = out[:, encoded["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for i in range(len(chunk)):
            texts.append(decoded[i * n_samples:(i + 1) * n_samples])
    return texts


def run_extraction(reports, desc="", variant="baseline", n_samples=N_SAMPLES):
    prompts = [to_chat(r, variant) for r in reports]
    print(f"generating {len(prompts)} x {n_samples} samples {desc} "
          f"variant={variant} ({'vLLM' if USE_VLLM else 'transformers'})...")

    if USE_VLLM:
        sampling = SamplingParams(temperature=0.7, top_p=0.9,
                                  max_tokens=MAX_NEW_TOKENS, n=n_samples)
        outputs = llm.generate(prompts, sampling)
        per_report = [[o.text for o in output.outputs] for output in outputs]
    else:
        per_report = _generate_hf(prompts, n_samples)

    return [aggregate_probabilities([parse_probabilities(t) for t in texts])
            for texts in per_report]


# Categorical mode (Phase C lever 3). The shipped categorical corpus is
# still from the 14B; these reuse the module's original categorical prompt
# and parser so the 32B can be measured on the same contract. The JSON here
# carries severity/certainty/evidence per finding, so it needs more room
# than a bare number map.
CAT_MAX_NEW_TOKENS = 900


def to_chat_categorical(report):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(str(report)[:MAX_REPORT_CHARS])},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)


def run_categorical(reports, desc="", n_samples=N_SAMPLES):
    prompts = [to_chat_categorical(r) for r in reports]
    print(f"generating {len(prompts)} x {n_samples} categorical samples {desc} "
          f"({'vLLM' if USE_VLLM else 'transformers'})...")

    if USE_VLLM:
        sampling = SamplingParams(temperature=0.7, top_p=0.9,
                                  max_tokens=CAT_MAX_NEW_TOKENS, n=n_samples)
        outputs = llm.generate(prompts, sampling)
        per_report = [[o.text for o in output.outputs] for output in outputs]
    else:
        per_report = _generate_hf(prompts, n_samples,
                                  max_new_tokens=CAT_MAX_NEW_TOKENS)

    return [aggregate_samples([parse_response(t) for t in texts])
            for texts in per_report]
""")

    nb.md("""
## 6. Checkpoint — the gold sweep (Phase C, extraction v3)

**Stop here and read the tables.** This is the only honest measurement
available, and it is deliberately a sweep: five configurations over the 58
gold reports — the shipped baseline at 3 and at 5 self-consistency samples,
then three one-idea prompt variants at 5 samples (`correlates`: named
co-occurrence priors for silent findings, aimed at Synovitis/Lateral/PF OA;
`granular`: pressure against tie blocks; `reason_first`: a short sorting step
before the numbers). ~290 prompts × 5 samples — minutes on an A100.

The raw AUC printed here is a screen, not the gate. **Download the
`gold_sweep.zip` this cell produces**, unzip it into `data/meta/gold_sweep/`,
and run `python3 tools/score_prompt_variants.py` locally — that scores the
full fusion (the number comparable to the shipped labels' 0.891) and applies
the gate: beat 0.891 by more than 0.005 with no finding below 0.75. Only a
gate-passing variant earns the corpus run below.

**Read the raw table with the leverage column in mind.** It weights all
twelve findings equally; fusion does not, because the categorical extractor
and the lexicon already carry several of them. Measured 2026-08-26 — how
much fused gold AUC the probability mode adds over categorical-only:

| leverage | findings |
|---|---|
| high (+0.04 to +0.11) | Fracture, Effusion, Synovitis, Contusion, Medial OA, Lateral OA |
| none (+0.00 to +0.01) | Medial Meniscus, ACL, PF OA, MCL, Lateral Meniscus, Baker's |

A variant that wins raw macro on the bottom row has won nothing — that
ground is already held. Judge a candidate by the top row, then confirm with
the local fused score.
""")

    nb.code("""
import numpy as np

SWEEP = [("baseline", 3), ("baseline", 5), ("correlates", 5),
         ("granular", 5), ("reason_first", 5)]

sweep_results = {}
for variant, n in SWEEP:
    sweep_results[(variant, n)] = run_extraction(
        gold["Report"].tolist(), "(gold 58)", variant=variant, n_samples=n)

gold_results = sweep_results[("baseline", 3)]


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


# Raw per-finding AUC, one column per sweep configuration. This is a screen:
# the decision number is the fused score, computed locally from the dumps by
# tools/score_prompt_variants.py. The shipped baseline's raw macro was 0.860.
def raw_scores(results, name):
    return np.array([r[name]["probability"] if r[name]["probability"] is not None
                     else 0.0 for r in results])


header = "  ".join(f"{v[:9]}@{n}" for v, n in SWEEP)
print(f"\\n{'finding':18s} {header}")
print("-" * (20 + 13 * len(SWEEP)))
macros = {key: [] for key in sweep_results}
for name in LABELS:
    truth = gold[name].astype(int).to_numpy()
    row = []
    for key in ((v, n) for v, n in SWEEP):
        a = auc(truth, raw_scores(sweep_results[key], name))
        macros[key].append(a)
        row.append(f"{a:11.3f}")
    print(f"{name:18s} {'  '.join(row)}")
print("-" * (20 + 13 * len(SWEEP)))
print(f"{'macro':18s} " + "  ".join(
    f"{np.nanmean(macros[(v, n)]):11.3f}" for v, n in SWEEP))
print("  shipped baseline raw macro: 0.860 -- fused labels: 0.891 (the gate)")

# Dump every configuration so fusion is scored locally in seconds rather than
# by re-running the model. 58 rows each.
#
# One zip, one download. Calling files.download() once per file loses all but
# the first: the browser blocks a burst of downloads from a single cell, and
# the 2026-08-26 sweep lost four of its five dumps that way -- silently, since
# the failure is browser-side and nothing in the notebook raises.
import zipfile

written = []
for (variant, n), results in sweep_results.items():
    dump = pd.DataFrame(
        [{name: r[name]["probability"] for name in LABELS} for r in results])
    dump.insert(0, "StudyInstanceUID", gold["StudyInstanceUID"].values)
    for name in LABELS:
        dump[f"{name}_confidence"] = [r[name]["confidence"] for r in results]
    path = f"llm_probs_gold__{variant}_n{n}.parquet"
    dump.to_parquet(path, index=False)
    written.append(path)
    print("wrote", path)

BUNDLE = "gold_sweep.zip"
with zipfile.ZipFile(BUNDLE, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in written:
        archive.write(path)
print(f"\\nbundled {len(written)} dumps into {BUNDLE}")
print("unzip into data/meta/gold_sweep/ and run tools/score_prompt_variants.py")

try:
    from google.colab import files
    files.download(BUNDLE)
except Exception:
    pass
""")

    nb.md("""
## 6b. Categorical mode at the gold 58 (Phase C lever 3)

The shipped categorical corpus — the half of fusion that carries Medial
Meniscus, ACL, PF OA, MCL, Lateral Meniscus and Baker's — is still from the
**14B**. This cell runs the original categorical prompt on the gold 58 with
the model section 1 picked (the 32B on an A100/L4), so the upgrade is
measured for minutes before the ~1 h corpus re-extraction is even
considered.

Download the parquet, drop it into `data/meta/gold_sweep/`, and run
`python3 tools/score_categorical_upgrade.py` locally — it swaps the 32B
frame into the full fusion next to the shipped one and applies the same
gate (fused macro > shipped + 0.005, no finding below 0.75). The corpus
re-run is earned only by a PASS.
""")

    nb.code("""
cat_gold_results = run_categorical(gold["Report"].tolist(), "(gold 58)")

# Same schema as llm_labels_gold.parquet, so the local scorer can hand it
# to the fusion unchanged: <name> (1/0/NaN), _severity, _confidence,
# _evidence per finding.
cat_rows = []
for result in cat_gold_results:
    row = {}
    for name in LABELS:
        entry = result[name]
        row[name] = entry["present"]
        row[f"{name}_severity"] = entry["severity"]
        row[f"{name}_confidence"] = entry["confidence"]
        row[f"{name}_evidence"] = entry["evidence"]
    cat_rows.append(row)
cat_dump = pd.DataFrame(cat_rows)
cat_dump.insert(0, "StudyInstanceUID", gold["StudyInstanceUID"].values)

CAT_PATH = "llm_labels_gold__32b.parquet"
cat_dump.to_parquet(CAT_PATH, index=False)

# A quick parse-health screen; the verdict itself is the local fused score.
unparsed = sum(1 for r in cat_gold_results
               if all(r[n]["present"] is None for n in LABELS))
print(f"wrote {CAT_PATH}: {len(cat_dump)} studies, "
      f"{unparsed} with nothing parsed")
for name in LABELS:
    calls = [r[name]["present"] for r in cat_gold_results]
    print(f"  {name:18s} yes {calls.count(1):2d}  no {calls.count(0):2d}  "
          f"silent {calls.count(None):2d}")

try:
    from google.colab import files
    files.download(CAT_PATH)
except Exception:
    pass
""")

    nb.md("""
## 7. Full corpus

Only run this after `tools/score_prompt_variants.py` has picked a winner
**locally, on the fused score** — set the two constants to the winning
configuration first. If nothing passes the gate (fused macro > 0.891 + 0.005,
no finding below 0.75), stop here: the sweep cost minutes, and a corpus run
that ships nothing costs an hour. This is the long cell (~45–75 min at 3
samples, scale accordingly).

The gold 58 are re-run at the chosen configuration so the pair of files
section 8 writes is internally consistent.
""")

    nb.code("""
import time

# The sweep's verdict goes here, and stays at the shipped default until a
# variant passes the local fused gate.
PROMPT_VARIANT = "baseline"
N_SAMPLES_CORPUS = 3

start = time.time()
gold_results = run_extraction(gold["Report"].tolist(), "(gold 58)",
                              variant=PROMPT_VARIANT,
                              n_samples=N_SAMPLES_CORPUS)
mined = run_extraction(unlabelled["Report"].tolist(), "(4,349 reports)",
                       variant=PROMPT_VARIANT, n_samples=N_SAMPLES_CORPUS)
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
            row[name] = entry["probability"]
            row[f"{name}_confidence"] = entry.get("confidence")
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.insert(0, "StudyInstanceUID", list(ids))
    return frame


mined_frame = to_frame(mined, unlabelled["StudyInstanceUID"])
gold_frame = to_frame(gold_results, gold["StudyInstanceUID"])

mined_frame.to_parquet("llm_probs.parquet", index=False)
gold_frame.to_parquet("llm_probs_gold.parquet", index=False)

print(mined_frame.shape)
print("\\nprobability distribution per finding (mined):")
for name in LABELS:
    values = mined_frame[name].dropna()
    print(f"  {name:18s} mean {values.mean():5.2f}  "
          f"distinct {values.nunique():5d}  missing {mined_frame[name].isna().mean():5.1%}")

from google.colab import files
files.download("llm_probs.parquet")
files.download("llm_probs_gold.parquet")
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
    # Prefix match, not exact: a pooling grid appends a suffix
    # (embeddings_train_series_00_g4.npy), and an exact match would skip every
    # shard and train on nothing.
    matching = [f for f in embedding_files
                if os.path.basename(f).startswith(f"embeddings_{stem}")]
    if not matching:
        print("no embeddings for", stem, "-- skipping")
        continue
    if len(matching) > 1:
        raise SystemExit(
            f"{len(matching)} embedding files match {stem}:\\n  "
            + "\\n  ".join(matching)
            + "\\nAttach one variant at a time so the run is unambiguous.")
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

    # Per-epoch rather than per-fold: a bar that only ticks once a fold
    # completes is indistinguishable from a hang for ten minutes at a time.
    for epoch in tqdm(range(EPOCHS), desc=f"fold {seed}", leave=False):
        model.train()
        running = 0.0
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
            running += float(loss)
        if epoch == 0 or (epoch + 1) % 4 == 0:
            print(f"    fold {seed} epoch {epoch + 1}/{EPOCHS} "
                  f"loss {running / len(train_loader):.4f}", flush=True)

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
# 08 — Fine-Tuned Backbone

Trains a ConvNeXt-T end to end instead of reading frozen DINOv2 features. Prior
RSNA competitions were won by fine-tuned ImageNet CNNs, so this is the arm most
likely to top the leaderboard, and the one the frozen path exists to be
measured against.

**Runs on Kaggle, not Colab.** Fine-tuning needs raw volumes, and one cache
shard is 9.7 GB. Pulling that to Colab failed twice at ~5.5 GB with
IncompleteRead -- a consistent ceiling rather than a flaky connection. On
Kaggle the cache is already mounted and nothing transfers. A T4 is slower than
an A100, but not slower than a download that never finishes.

**Attach:** one cache dataset, `knee-labels`, `knee_side_classifier`, and the
competition data. GPU on. The auxiliary report loss additionally wants
`report_embeddings.parquet` (built locally by `src/labels/report_embed.py`)
in the `knee-labels` dataset; without it that loss silently disables, which
the assemble cell prints.

**This run measures before it commits.** One study is ~5.5 series x 24 slices
= ~130 slice-forwards, so a full epoch over 4,349 studies is ~570,000. The
training cell reports actual throughput and projects the full sweep. Read that
before launching five folds.

Subsampling controls the cost, and both knobs are real trade-offs:

    SLICES_PER_SERIES  16 of 24     the reference geometry; 24 overfit for +0.002
    MAX_SERIES          5 of ~5.5   a finding may live only in a dropped series

The capacity ladder is finished (0.845 -> 0.869 -> 0.889 -> 0.891) and 16
slices x 5 series is where experiments iterate. Expect roughly **9 min/epoch,
~1.8 h for twelve epochs**. If it OOMs, drop `MAX_SERIES` to 4.

**Mirroring is used here**, unlike the frozen path: raw volumes can be
reflected, so left knees are mapped onto the canonical side along the axis the
acquisition plane dictates. That makes this a comparison of two things at once,
backbone and laterality handling, which is worth remembering when reading the
result.
""")

    nb.md("## 1. Setup")

    nb.code("""
!pip install -q timm

import os
import os
import subprocess

# CUDA MUST NOT BE INITIALISED IN THIS PROCESS. Lightning's ddp_notebook
# strategy forks, and a fork inheriting a live CUDA context dies in the child
# with "Cannot re-initialize CUDA in forked subprocess".
#
# The trap is that the obvious check is the unsafe one. torch's own docs are
# explicit: is_available() "will NOT poison fork if
# PYTORCH_NVML_BASED_CUDA_CHECK=1" -- meaning by default it does, because it
# falls through to _cuda_getDeviceCount(), which calls cuInit. device_count()
# asks NVML first and stays clean. get_device_name and get_device_properties
# reach _lazy_init and are also out. Device names come from nvidia-smi, which
# is a separate process entirely.
#
# The env var is set as well, before torch is touched, so that any library
# calling is_available() downstream takes the NVML path too.
os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"

import torch

DEVICES = torch.cuda.device_count()
if DEVICES == 0:
    raise SystemExit("Settings -> Accelerator -> GPU T4 x2")
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip())
print("devices:", DEVICES, "| cuda initialised:", torch.cuda.is_initialized())
if DEVICES < 2:
    print("NOTE: one GPU visible. Settings -> Accelerator -> GPU T4 x2 "
          "roughly halves wall-clock. The run is correct either way -- "
          "ACCUM_STEPS below adapts to hold the effective batch constant.")

# Lightning ships in the Kaggle image, but the package name moved between
# major versions and a missing install should fail here rather than thirty
# minutes into the cell that matters.
try:
    import lightning.pytorch as pl
except ImportError:
    try:
        import pytorch_lightning as pl
    except ImportError:
        raise SystemExit("pip install lightning, then re-run this cell")
print("lightning:", pl.__version__)
""")

    nb.md("## 2. Repo modules")
    nb.code(bootstrap_cell(FINETUNE_MODULES, require_raw_dicom=False))

    nb.md("""
## 3. Assemble

One shard only. ~1,460 studies is enough to measure throughput and train a
first fold, and loading all three would cost memory without changing what this
run establishes.
""")

    nb.code("""
import glob

import numpy as np
import pandas as pd


def find_inputs(pattern):
    # ./data first, so the same notebook runs on Kaggle with the cache
    # attached and on Colab with it downloaded by notebook 09.
    found = sorted(glob.glob(f"data/**/{pattern}", recursive=True))
    found += sorted(glob.glob(f"/kaggle/working/{pattern}"))
    for root in sorted(glob.glob("/kaggle/input/*")):
        if os.path.basename(root) == "competitions":
            continue
        found += sorted(glob.glob(f"{root}/**/{pattern}", recursive=True))
    return found


def find_competition_file(name):
    for candidate in (f"data/{name}",
                      f"/kaggle/input/competitions/*/{name}",
                      f"/kaggle/input/*/{name}"):
        hits = sorted(glob.glob(candidate))
        if hits:
            return hits[0]
    raise SystemExit(f"missing competition file: {name}")


labels = pd.read_parquet(find_inputs("train_labels.parquet")[0])
sides = pd.read_parquet(find_inputs("side_predictions.parquet")[0])

# Report sentence embeddings, built locally by src/labels/report_embed.py and
# uploaded alongside the labels. Optional: without the file the auxiliary
# report loss is disabled and training proceeds as before.
report_hits = find_inputs("report_embeddings.parquet")
reports = pd.read_parquet(report_hits[0]) if report_hits else None
if reports is None:
    print("report_embeddings.parquet not attached -- report loss disabled")
series_meta = pd.read_csv(find_competition_file("train_series.csv"))
train_csv = pd.read_csv(find_competition_file("train.csv"))
gold = train_csv[train_csv["ACL"].notna()].reset_index(drop=True)

# The measurement run is done: 2.5 min/epoch on one shard, ~7.6 min on all
# three, and the gold score was still climbing at epoch 4 rather than
# plateauing. So this is the real run.
ALL_SHARDS = True

cache_paths = find_inputs("cache_train_series_*.npy")
if not ALL_SHARDS:
    cache_paths = cache_paths[:1]

shards, frames = {}, []
for cache_path in cache_paths:
    manifest_path = cache_path.replace("cache_", "manifest_").replace(".npy", ".parquet")
    if not os.path.exists(manifest_path):
        stem = os.path.basename(cache_path).replace("cache_", "").replace(".npy", "")
        hits = [m for m in find_inputs("manifest_train_series_*.parquet")
                if os.path.basename(m).startswith(f"manifest_{stem}")]
        if not hits:
            continue
        manifest_path = hits[0]
    frame = pd.read_parquet(manifest_path)
    frame["cache_path"] = cache_path
    shards[cache_path] = np.load(cache_path, mmap_mode="r")
    frames.append(frame)
    print("shard:", os.path.basename(cache_path), shards[cache_path].shape)

manifest = pd.concat(frames, ignore_index=True)

folded = splits.make_folds(labels)
splits.assert_no_gold_leakage(folded["StudyInstanceUID"], set(gold["StudyInstanceUID"]))

# Capacity ladder, step 2. Doubling slices was worth +0.024 on the
# leaderboard, so the other axis the study is subsampled on is the next thing
# to test. Most studies hold ~5.5 series and three were being kept; PF OA is
# one of only two findings where this model still loses to frozen DINOv2
# features, and the patella may simply not be in the three that survive.
MAX_SERIES = 5

index = knee_dataset.StudyIndex(manifest, series_meta, folded, sides,
                                max_series=MAX_SERIES, reports=reports)
gold_labels = gold[["StudyInstanceUID"] + config.LABELS].copy()
for name in config.LABELS:
    gold_labels[f"{name}_weight"] = 1.0
gold_index = knee_dataset.StudyIndex(manifest, series_meta, gold_labels, sides,
                                     max_series=MAX_SERIES)
print(f"studies in this shard: {len(index)}  gold present: {len(gold_index)}")
""")

    nb.md("""
## 4. Dataset

`StudyVolumeDataset` lives in `src/model/lightning.py` rather than in this
cell. Lightning's notebook DDP strategy spawns a process per GPU and pickles
the dataset into each one; a class defined in a notebook cell belongs to
`__main__` and does not survive that. The same rule already sent the cache's
process-pool worker into `src/data/cache.py`.

The memory-mapped shards are deliberately **not** handed to the dataset. It
takes paths and opens them lazily per process, and drops the handles in
`__getstate__`, so ~29 GB of memmap never gets serialised into a worker.
""")

    nb.code("""
from torch.utils.data import DataLoader

# The capacity ladder is finished: 8 -> 16 slices paid +0.024 on the
# leaderboard, 3 -> 5 series +0.020, and 16 -> 24 slices +0.002 while the
# gold curve began falling after epoch 7 -- overfitting, not headroom.
#
# Experiments therefore iterate at 16 slices, the reference geometry behind
# LB 0.889 / gold 0.861 on fold 1. The current experiment is the pair of
# auxiliary losses above; 24 remains a one-line change reserved for a final
# AUC-slot run, where its +0.002 may still be worth the runtime.
SLICES_PER_SERIES = 16

# One study per rank, held over from the five-series rung (1 x 5 x 16 = 80
# images per step). If it OOMs the fallback is MAX_SERIES 5 -> 4, not a
# lower slice count, because slices are the axis that has paid twice.
BATCH_STUDIES = 1

# Effective batch is BATCH_STUDIES x DEVICES x ACCUM_STEPS, and it must stay
# at eight studies whether this runs on one GPU or two -- otherwise the two
# configurations are not comparable and the OneCycleLR step count shifts
# underneath the schedule.
EFFECTIVE_STUDIES = 8
ACCUM_STEPS = max(1, EFFECTIVE_STUDIES // (BATCH_STUDIES * DEVICES))
print(f"{BATCH_STUDIES} studies x {DEVICES} devices x {ACCUM_STEPS} accum "
      f"= {BATCH_STUDIES * DEVICES * ACCUM_STEPS} effective")

CACHE_PATHS = list(shards)


def make_loader(study_index, studies, shuffle=False, augment=False):
    return DataLoader(
        knee_lightning.StudyVolumeDataset(
            study_index, studies, CACHE_PATHS, SLICES_PER_SERIES, augment),
        batch_size=BATCH_STUDIES, shuffle=shuffle, num_workers=2,
        collate_fn=knee_dataset.collate_studies, pin_memory=True,
        persistent_workers=True)
""")

    nb.md("""
## 5. Model

ConvNeXt-T with ImageNet weights, which the competition permits. The
aggregation above it is the same `KneeNet` the frozen path uses, so the
comparison is between backbones rather than between architectures.
""")

    nb.code("""
import timm

BACKBONE = "convnext_tiny.fb_in22k_ft_in1k"

# Built once here before the trainer spawns its workers. Each rank constructs
# its own backbone, and two processes racing to download the same weights into
# the same cache is a failure that only appears on multi-GPU runs -- warming
# the cache first turns that race into two cache hits.
_warm = timm.create_model(BACKBONE, pretrained=True, num_classes=0, in_chans=3)
FEATURE_DIM = _warm.num_features
print(f"{BACKBONE}: {FEATURE_DIM} features, "
      f"{sum(p.numel() for p in _warm.parameters()) / 1e6:.1f}M params")
del _warm

auc = knee_lightning.auc
""")

    nb.md("""
## 6. Train one fold

`FOLD` below decides which of the five folds this session trains and validates
on; five sessions with `FOLD = 0` through `4` produce the checkpoints notebook
10 ensembles. The best epoch is chosen on the **validation fold** against the
mined labels. Gold is printed every epoch but never selects anything -- an
earlier version checkpointed on gold macro, which quietly turned the only
honest measuring stick into a model-selection set and inflated its reading by
an estimated 0.01-0.03.
""")

    nb.code("""
import time

import timm

# The measurement run reached 0.729 at epoch 4 and was still rising
# (0.584 -> 0.679 -> 0.708 -> 0.729), so four was undertrained.
EPOCHS = 12
LR = 1e-4

# Two auxiliary objectives, both training-time only and both dropped at
# export, so notebook 10 is untouched. Either is disabled by setting its
# weight to 0.0, which is how a single-variable ablation is run.
#
# LAMBDA_REPORT: the pooled study embedding regresses the report's sentence
# embedding (cosine distance). The report describes more than twelve numbers,
# and this hands that richer summary to the network as extra supervision --
# also a lever against the overfitting the 24-slice run showed, since the
# regression target is far harder to memorise than twelve soft labels.
#
# LAMBDA_RANK: the metric is macro AUC, a ranking statistic, but BCE
# optimises calibration. This RankNet term orders the current study against
# a queue of recent (detached) predictions, pairing only studies whose soft
# targets differ by >= 0.25 so extraction noise is not what gets ranked.
# Screen result (fold 1, both at 0.2): gold 0.854 vs the 0.861 reference.
# Both losses trained (report 0.17->0.044, rank 0.65->0.026), but rank
# collapsed to near-zero -- memorised its own queue.
#
# LB verdict (2026-08-26): report 0.5 alone scored val 0.881 / gold 0.867
# on fold 1 at 16 x 5 -- the best fold-1 numbers measured -- but the
# submission came in at LB 0.880 against the aux-free twin's 0.889 on the
# identical geometry. Val labels are mined from the same reports the aux
# target embeds, so the val gain was partly circular, and gold's +0.006 sat
# inside its noise floor. Both lambdas therefore ship at 0.0; the code
# stays for any later phase that reaches for more capacity.
LAMBDA_REPORT = 0.0
LAMBDA_RANK = 0.0
REPORT_DIM = (len([c for c in reports.columns if c != "StudyInstanceUID"])
              if reports is not None else None)

# Which fold this session trains.
#
# The fold-ensemble phase is on (2026-08-26): sweep 0 through 4, one per
# session, editing only this line, producing convnext_fold0.pt through
# convnext_fold4.pt. Fold 1 already exists (the 16 x 5 reference,
# LB 0.889). For single-config experiments, return to 1 -- the only split
# with a measured baseline; changing the fold and the configuration
# together has already cost this project one ambiguous run.
FOLD = 3
CHECKPOINT = f"/kaggle/working/convnext_fold{FOLD}.pt"

keyed = folded.set_index("StudyInstanceUID")
train_studies = [s for s in index.studies if keyed.loc[s, "fold"] != FOLD]
val_studies = [s for s in index.studies if keyed.loc[s, "fold"] == FOLD]

train_loader = make_loader(index, train_studies, shuffle=True, augment=True)
# Dataloader 0 is the validation fold, carrying mined soft targets -- the only
# thing allowed to select. Dataloader 1 is gold, carrying real 0/1 labels,
# logged for convergence and never monitored.
val_loader = make_loader(index, val_studies)
gold_loader = make_loader(gold_index, gold_index.studies)
print(f"fold {FOLD}: train {len(train_studies)}  val {len(val_studies)}  "
      f"gold {len(gold_index)}")

TOTAL_STEPS = knee_lightning.optimiser_steps(
    len(train_studies), BATCH_STUDIES, DEVICES, ACCUM_STEPS, EPOCHS)
print("optimiser steps:", TOTAL_STEPS)

module = knee_lightning.KneeLitModule(
    backbone=timm.create_model(BACKBONE, pretrained=True, num_classes=0,
                               in_chans=3),
    feature_dim=FEATURE_DIM, lr=LR, total_steps=TOTAL_STEPS,
    report_dim=REPORT_DIM, lambda_report=LAMBDA_REPORT,
    lambda_rank=LAMBDA_RANK)
print(f"aux losses: report {LAMBDA_REPORT} (dim {REPORT_DIM})  "
      f"rank {LAMBDA_RANK}")

checkpointer = pl.callbacks.ModelCheckpoint(
    dirpath="/kaggle/working", filename=f"lit_fold{FOLD}",
    monitor="val_macro", mode="max", save_top_k=1)
report = knee_lightning.EpochReport()

# Metrics go to a CSV as well as to stdout. Under ddp_notebook every rank is
# a forked child, and a child's prints do not always find their way back to
# the cell output -- so the per-epoch numbers are read back from disk below,
# in this process, where they are guaranteed to be visible.
csv_logger = pl.loggers.CSVLogger("/kaggle/working", name=f"lit_fold{FOLD}",
                                  version="metrics")

# ddp_notebook forks rather than re-executing the script, which is what makes
# DDP usable from a notebook at all -- plain "ddp" deadlocks here. If the fork
# route ever proves unworkable, "ddp_spawn" is a drop-in: everything the child
# needs is picklable by design, since the dataset and module live in
# src/model/lightning.py and the memmaps are dropped in __getstate__.
STRATEGY = "ddp_notebook" if DEVICES > 1 else "auto"

# The fork cannot survive a CUDA context that already exists here. Catch it
# now, with an instruction, rather than as a sixty-line traceback out of a
# child process after the model and loaders are built.
if STRATEGY == "ddp_notebook" and torch.cuda.is_initialized():
    raise SystemExit(
        "CUDA is already initialised in this process, so the forked workers "
        "cannot start.\\n"
        "Run -> Restart Session, then run the cells in order. If it recurs, "
        "set STRATEGY = 'ddp_spawn' in this cell."
    )

trainer = pl.Trainer(
    max_epochs=EPOCHS,
    accelerator="gpu",
    devices=DEVICES,
    strategy=STRATEGY,
    precision="16-mixed",
    accumulate_grad_batches=ACCUM_STEPS,
    callbacks=[checkpointer, report, knee_lightning.BatchProgress()],
    logger=csv_logger,
    log_every_n_steps=50,
    enable_model_summary=False,
    enable_progress_bar=False,
)

start = time.time()
trainer.fit(module, train_dataloaders=train_loader,
            val_dataloaders=[val_loader, gold_loader])
elapsed = time.time() - start

# EXPORT FIRST. Ninety minutes of GPU time exist only as a checkpoint file at
# this point, and anything that runs before the export is a chance to lose it
# to an unrelated error -- which is exactly what a stray reporting bug did on
# the first two-GPU run. Reporting is cosmetic and comes after, wrapped.
# Under ddp_notebook the fit happens in forked children, and Lightning
# transfers only `best_model_path` back to this process -- `best_model_score`
# is left None, and their launcher carries a "TODO: pass also best score" to
# that effect. Formatting it here crashed a finished twelve-epoch run. So the
# path is used, the score comes from the CSV below, and the path itself is
# checked against the filesystem rather than trusted.
checkpoint_path = checkpointer.best_model_path or ""
if not checkpoint_path or not os.path.exists(checkpoint_path):
    found = sorted(glob.glob(f"/kaggle/working/lit_fold{FOLD}*.ckpt"))
    if not found:
        raise SystemExit(
            "training finished but no checkpoint was written -- check the "
            "epoch log for whether val_macro was ever computed")
    checkpoint_path = found[0]
print("\\nbest checkpoint:", checkpoint_path)

# Convert Lightning's checkpoint into the bare KneeNet state dict notebook 10
# loads. Read from the file, not from `module`: the process that called fit is
# not the process that trained.
n_tensors = knee_lightning.export_from_checkpoint(checkpoint_path, CHECKPOINT)
print(f"wrote {CHECKPOINT} ({n_tensors} tensors)")

# Re-read the epoch table rather than trusting the forked ranks' stdout. The
# logger's version is pinned, because the child resolves it during fit and a
# parent asking afterwards would count the child's directory as taken and
# point one version too far along an empty path.
try:
    history = pd.read_csv(os.path.join(csv_logger.log_dir, "metrics.csv"))
    per_epoch_rows = (history.dropna(subset=["val_macro"])
                      .groupby("epoch")[["val_macro", "gold_macro"]].last())
    best_epoch = per_epoch_rows["val_macro"].idxmax()
    print(f"\\n{'epoch':>5s} {'val macro':>9s} {'gold':>7s}")
    for epoch_no, row in per_epoch_rows.iterrows():
        marker = "  <- selected" if epoch_no == best_epoch else ""
        print(f"{epoch_no + 1:5d} {row['val_macro']:9.3f} "
              f"{row['gold_macro']:7.3f}{marker}")
    best = per_epoch_rows.loc[best_epoch]
    print(f"\\nbest val macro {best['val_macro']:.3f} at epoch "
          f"{best_epoch + 1}; gold there {best['gold_macro']:.3f} "
          "(monitor only, never selected on)")
except Exception as exc:  # noqa: BLE001 - the checkpoint is already safe
    print(f"\\ncould not read the epoch table ({exc}); "
          f"look under {csv_logger.save_dir} for metrics.csv")

per_epoch = elapsed / EPOCHS
share = len(index) / 4349
print(f"\\nthis shard is {share:.0%} of the training set")
print(f"per epoch: {per_epoch / 60:.1f} min on {DEVICES} GPU(s)")
print(f"projected 5 folds x {EPOCHS} epochs: "
      f"{per_epoch / share * EPOCHS * 5 / 3600:.1f} h")
""")

    nb.md("""
## 7. Score

Against the gold studies present in this shard -- roughly a third of the 58, so
this is noisier than the frozen path's estimate. Read it as a direction, not a
number.
""")

    nb.code("""
# Scored in this process from the exported checkpoint, which is the artefact
# the submission will actually load -- not from a module left over in memory.
# If these numbers disagree with the last epoch's logged gold, the export is
# wrong, and that is worth knowing here rather than after a submission.
device = "cuda"

# Prefer the exported state dict, but fall back to the Lightning checkpoint
# it came from. A recovered run -- one where the training cell died after fit
# and the export was re-run by hand -- can easily have the .ckpt and not the
# .pt, and this cell is the verification step, so it must not be the thing
# that refuses to run.
scoring_path = CHECKPOINT
if not os.path.exists(scoring_path):
    found = sorted(glob.glob(f"/kaggle/working/lit_fold{FOLD}*.ckpt"))
    if not found:
        raise SystemExit(f"neither {CHECKPOINT} nor a lit_fold{FOLD} "
                         "checkpoint is present")
    scoring_path = found[0]
    print(f"{CHECKPOINT} missing; scoring from {scoring_path}")

scoring_model = net.KneeNet(
    timm.create_model(BACKBONE, pretrained=False, num_classes=0, in_chans=3),
    FEATURE_DIM).to(device)
scoring_model.load_state_dict(net.load_knee_state(scoring_path,
                                                  map_location=device))
scoring_model.eval()

gold_ids, rows = [], []
with torch.no_grad(), torch.amp.autocast("cuda"):
    for batch in gold_loader:
        logits = scoring_model(batch["volumes"].to(device),
                               batch["meta"].to(device),
                               batch["mask"].to(device))
        rows.append(torch.sigmoid(logits.float()).cpu().numpy())
        gold_ids += batch[config.ID_COL]
gold_probs = np.concatenate(rows)
truth = gold.set_index("StudyInstanceUID").loc[gold_ids]

FROZEN = dict(zip(config.LABELS,
                  [0.728, 0.671, 0.698, 0.680, 0.895, 0.899,
                   0.772, 0.897, 0.746, 0.862, 0.711, 0.683]))

print(f"{'finding':18s} {'this run':>9s}  {'frozen DINOv2':>13s}")
print("-" * 44)
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

**The runtime projection.** If five folds fit in the weekly GPU quota, run them
and compare against the frozen path properly. If not, the levers are
`SLICES_PER_SERIES`, `MAX_SERIES`, fewer folds, or freezing the backbone's
early stages.

**The focal findings.** Fine-tuning should help meniscus, ligaments, and
fracture most, since those are what frozen natural-image features handle worst.
If they do not move, the limit is elsewhere: the mined labels, or 224px
resolution against the size of the structures.
""")

    nb.write(NOTEBOOKS / "08_finetune.ipynb")


# ---------------------------------------------------------------------------
# 09 -- export a Colab-sized cache
# ---------------------------------------------------------------------------

def build_export_notebook():
    nb = Notebook(gpu=False)

    nb.md("""
# 09 — Export a Colab-Sized Cache

Writes a thinned copy of one cache shard: the same 224px volumes, but only the
8 slices per series that fine-tuning uses anyway.

    full shard      1.2 MB/series     9.7 GB     will not transfer
    thinned shard   401 KB/series     3.2 GB     transfers fine

Pulling a full shard to Colab failed twice at ~5.5 GB with IncompleteRead, so
3.2 GB is the point of this. Nothing is lost that notebook 08 would have read:
it subsamples to these same 8 slices on load.

**No GPU.** This is a memmap read and a write, so it does not touch the weekly
GPU quota -- which matters when that quota is 25 hours and only one session
runs at a time.

**Attach:** one cache dataset, plus `knee-labels` and `knee_side_classifier` so
the whole Colab bundle comes out of one publish.
""")

    nb.md("## Setup")
    nb.code(bootstrap_cell(MODEL_MODULES, require_raw_dicom=False))

    nb.code("""
import glob
import os
import shutil

import numpy as np

# Deliberately decoupled from notebook 08, which now trains at 16. This
# export exists to fit Colab's transfer budget: 8 slices is 3.2 GB, and a
# full-size pull fails at ~5.5 GB with IncompleteRead. A 16-slice thin would
# be ~6.4 GB and would not arrive. A cache thinned here is therefore only
# usable by a training config set to this same number -- notebook 08 raises
# rather than silently training on short volumes.
SLICES_PER_SERIES = 8


def find_inputs(pattern):
    found = sorted(glob.glob(f"/kaggle/working/{pattern}"))
    for root in sorted(glob.glob("/kaggle/input/*")):
        if os.path.basename(root) == "competitions":
            continue
        found += sorted(glob.glob(f"{root}/**/{pattern}", recursive=True))
    return found


cache_path = find_inputs("cache_train_series_*.npy")[0]
manifest_path = find_inputs("manifest_train_series_*.parquet")[0]
print("source:", cache_path, f"{os.path.getsize(cache_path) / 1e9:.1f} GB")
""")

    nb.md("""
## Thin and write

Slice indices are the same evenly spaced selection notebook 08 makes, so the
thinned cache is exactly what it would have read from the full one.
""")

    nb.code("""
import time

volumes = np.load(cache_path, mmap_mode="r")
keep = np.linspace(0, config.CACHE_SLICES - 1,
                   SLICES_PER_SERIES).round().astype(int)
print("volumes:", volumes.shape, "-> keeping slices", keep.tolist())

out = np.zeros((volumes.shape[0], SLICES_PER_SERIES, volumes.shape[2],
                volumes.shape[3]), dtype=np.uint8)

start = time.time()
CHUNK = 256
for begin in range(0, len(volumes), CHUNK):
    end = min(begin + CHUNK, len(volumes))
    out[begin:end] = np.asarray(volumes[begin:end])[:, keep]
    if begin % (CHUNK * 8) == 0:
        print(f"  {begin}/{len(volumes)}", flush=True)

name = os.path.basename(cache_path).replace(".npy", f"_s{SLICES_PER_SERIES}.npy")
np.save(f"/kaggle/working/{name}", out)
shutil.copy(manifest_path, "/kaggle/working/")

size = os.path.getsize(f"/kaggle/working/{name}") / 1e9
print(f"\\n{name}  {out.shape}  {size:.2f} GB  in {(time.time() - start) / 60:.1f} min")
""")

    nb.md("""
## Bundle the small files

So the Colab side needs one download rather than four.
""")

    nb.code("""
for pattern in ("train_labels.parquet", "side_predictions.parquet"):
    for path in find_inputs(pattern):
        shutil.copy(path, "/kaggle/working/")
        break

for name in ("train.csv", "train_series.csv"):
    hits = sorted(glob.glob(f"/kaggle/input/competitions/*/{name}")) or \\
        sorted(glob.glob(f"/kaggle/input/*/{name}"))
    if hits:
        shutil.copy(hits[0], "/kaggle/working/")

total = sum(os.path.getsize(f"/kaggle/working/{f}")
            for f in os.listdir("/kaggle/working")
            if os.path.isfile(f"/kaggle/working/{f}"))
print(f"bundle: {total / 1e9:.2f} GB")
for f in sorted(os.listdir("/kaggle/working")):
    path = f"/kaggle/working/{f}"
    if os.path.isfile(path):
        print(f"  {f}  {os.path.getsize(path) / 1e6:.1f} MB")
""")

    nb.md("""
## Next

Save & Run All, then publish the output. On Colab:

```python
!pip install -q kaggle
# upload ~/.kaggle/access_token, then:
!kaggle kernels output <user>/<this-notebook-slug> -p data
```

Notebook 08 looks in `./data` before `/kaggle/input`, so it runs unchanged in
either place.
""")

    nb.write(NOTEBOOKS / "09_export_for_colab.ipynb")


# ---------------------------------------------------------------------------
# 10 -- ensemble submission
# ---------------------------------------------------------------------------

def build_ensemble_notebook():
    nb = Notebook(gpu=True)

    nb.md("""
# 10 — Ensemble Submission

Runs both models over the test set and blends them.

    frozen DINOv2-S + KneeNet   gold 0.770   strong on PF OA, Lateral OA, Effusion, Synovitis
    fine-tuned ConvNeXt-T       gold 0.825   strong on MCL, ACL, meniscus, Baker's, Fracture

They are complementary: a per-finding oracle pick of the better one scores
~0.848 against 0.825 for the better model alone.

**Blending is by rank, not by probability.** The two models are calibrated
differently -- one reads frozen features, the other was trained end to end --
so averaging raw probabilities lets the more confident model dominate
regardless of which is more accurate. The metric is AUC, which depends only on
ordering, so ranks are the natural currency and calibration stops mattering.

**A fixed blend weight, not per-finding weights.** Per-finding weights fitted on
58 gold studies is exactly what was tried during label fusion: it scored 0.860
in-sample and 0.812 under leave-one-out, worse than the better single source.
The same trap applies here with the same sample size, so the blend is one
number rather than twelve.

**Inputs:** competition data, `knee_embeddings` (DINOv2 weights),
`knee_side_classifier`, `knee-train` (frozen folds), `knee-convnext` (tuned
folds), `knee-labels` (priors for the fallback).
""")

    nb.md("## Setup")
    nb.code(bootstrap_cell(MODEL_MODULES, require_raw_dicom=True))

    nb.code("""
!pip install -q timm

import glob
import os
import time

import numpy as np
import pandas as pd
import torch

START = time.time()

# 0.5 blends evenly. 1.0 is the fine-tuned model alone, 0.0 the frozen one.
#
# At 0.0 or 1.0 the other path is skipped entirely rather than computed and
# discarded. That matters because the Efficiency Prize scores runtime: a
# single-path run must be timed as a single-path run, or the comparison
# measures nothing useful.
#
# The leaderboard settled this at 1.0: frozen 0.787, 50/50 blend 0.839, tuned
# alone 0.845. Averaging a weak ranker into a strong one landed between them.
# The tuned path averages probabilities across every convnext_fold*.pt it
# finds, so a five-fold upload becomes a five-model ensemble with no edit here.
BLEND = 1.0
USE_FROZEN = BLEND < 1.0
USE_TUNED = BLEND > 0.0

# Must match how the fine-tuned model was trained, on BOTH axes. Nothing
# enforces it: the aggregation masks over the slice axis and pads the series
# axis, so a mismatch runs clean and simply scores worse than it should. These
# track notebook 08's SLICES_PER_SERIES and MAX_SERIES -- a checkpoint trained
# at 8 slices and 3 series must be scored with 8 and 3 here. Notebook 08 is
# back at 16 for the auxiliary-loss experiments; a submission of the 24-slice
# 0.891 checkpoint needs 24 here instead.
TUNED_SLICES = 16
TUNED_MAX_SERIES = 5

BATCH_SLICES = 256
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, "| blend:", BLEND)


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
## Load both models

Everything from disk. The ConvNeXt checkpoint carries its own backbone weights,
so `pretrained=False` is correct here -- timm cannot reach the network anyway.
""")

    nb.code("""
import timm

# --- frozen path ---------------------------------------------------------
# Loaded only when the blend actually uses it. At BLEND = 1.0 the frozen
# predictions are discarded further down, so demanding dinov2_s_reg4.pt and
# knee_fold*.pt here would fail a tuned-only submission on inputs it never
# reads -- which is exactly what it did.
encoder = None
frozen_models = []

if USE_FROZEN:
    encoder = embed.build_encoder(
        pretrained=False,
        checkpoint_path=find_one("dinov2_s_reg4.pt")).to(device)

    widths = set()
    for path in find_inputs("knee_fold*.pt"):
        state = torch.load(path, map_location="cpu")
        input_dim = (state["project.weight"].shape[1]
                     if "project.weight" in state else embed.EMBED_DIM)
        feature_dim = state["head.2.weight"].shape[1]
        widths.add(input_dim)
        model = net.KneeNet(None, feature_dim, input_dim=input_dim).to(device)
        model.load_state_dict(state)
        model.eval()
        frozen_models.append(model)

    if len(widths) > 1:
        # Checkpoints from two pooling grids are attached. Test slices are
        # embedded once, at one grid, so half these models would silently
        # receive features they were never trained on -- and produce
        # plausible-looking noise.
        raise SystemExit(
            f"frozen checkpoints expect different input widths: "
            f"{sorted(widths)}.\\n"
            "Attach one training run's output, not both.")

    if widths:
        input_dim = widths.pop()
        embed.POOL_GRID = 1 if input_dim == embed.TOKEN_DIM * 2 else int(
            round(((input_dim / embed.TOKEN_DIM) - 1) ** 0.5))

# --- fine-tuned path -----------------------------------------------------
# Both shapes the training path produces are accepted: the bare state dict
# notebook 08 exports, and Lightning's own checkpoint, which nests the same
# tensors under `state_dict` behind a `model.` prefix. Requiring the export
# first would be requiring a step that can be skipped -- one already was, when
# a crash in the reporting cell landed between training and the conversion.
TUNED_PATTERNS = ("convnext_fold*.pt", "lit_fold*.ckpt")

tuned_paths = []
for pattern in TUNED_PATTERNS:
    tuned_paths += find_inputs(pattern)

tuned_models = []
for path in tuned_paths:
    backbone = timm.create_model("convnext_tiny.fb_in22k_ft_in1k",
                                 pretrained=False, num_classes=0, in_chans=3)
    model = net.KneeNet(backbone, backbone.num_features).to(device)
    model.load_state_dict(net.load_knee_state(path, map_location="cpu"))
    model.eval()
    tuned_models.append(model)
    print("  tuned:", os.path.basename(path))

side_model = side_classifier.SideNet().to(device)
side_model.load_state_dict(torch.load(find_one("side_model.pt"), map_location=device))
side_model.eval()

print(f"frozen folds: {len(frozen_models)} (pool grid {embed.POOL_GRID})"
      + ("" if USE_FROZEN else "  [skipped, BLEND=1.0]"))
print(f"tuned folds : {len(tuned_models)}"
      + ("" if USE_TUNED else "  [skipped, BLEND=0.0]"))
# Check the paths the blend actually uses, not whether any checkpoint at all
# turned up. A BLEND of 1.0 with the frozen weights attached and the tuned
# ones missing passes an "is anything here?" test, then skips the frozen path
# by design and predicts None for every study -- which becomes an empty frame
# and a submission of NaN, several cells later and far from the cause.
if USE_TUNED and not tuned_models:
    raise SystemExit(
        f"BLEND={BLEND} needs the tuned path, but nothing matching "
        f"{' or '.join(TUNED_PATTERNS)} was found. Attach the dataset "
        "holding the fine-tuned checkpoints.")
if USE_FROZEN and not frozen_models:
    raise SystemExit(
        f"BLEND={BLEND} needs the frozen path, but no knee_fold*.pt was "
        "found. Attach the frozen training output.")

try:
    train_labels = pd.read_parquet(find_one("train_labels.parquet"))
    PRIORS = {n: float(train_labels[n].mean()) for n in config.LABELS}
except SystemExit:
    PRIORS = {n: 0.5 for n in config.LABELS}
""")

    nb.md("""
## Inference

Volumes are built once per study and fed to both paths: the frozen path embeds
them with DINOv2, the tuned path mirrors and subsamples them exactly as it was
trained. A study that fails anywhere falls back to the training prior.
""")

    nb.code("""
from tqdm.auto import tqdm

test_csv = pd.read_csv(find_competition_file("test.csv"))
test_series_csv = pd.read_csv(find_competition_file("test_series.csv"))
series_meta = test_series_csv.set_index("SeriesInstanceUID")
test_root = config.DATA_ROOT / "test_series"
studies = test_csv[config.ID_COL].tolist()
print(f"studies: {len(studies)}")

KEEP = np.linspace(0, config.CACHE_SLICES - 1, TUNED_SLICES).round().astype(int)


def build_volumes(study_uid):
    directory = test_root / study_uid
    if not directory.exists():
        return [], []
    volumes, uids = [], []
    for series_dir in sorted(p for p in directory.iterdir() if p.is_dir()):
        paths = dicom_io.series_paths(series_dir)
        if not paths:
            continue
        headers = [dicom_io.read_header(p) for p in paths]
        volumes.append(cache.build_series_volume(headers)["volume"])
        uids.append(series_dir.name)
    return volumes, uids


def predict_side(volumes):
    batch = np.stack([side_classifier.prepare_batch(v)[0] for v in volumes])
    with torch.no_grad():
        probs = torch.sigmoid(side_model(torch.from_numpy(batch).to(device)))
    return "R" if float(probs.mean()) >= 0.5 else "L"


def meta_tensor(uids, side):
    rows = []
    for uid in uids:
        row = series_meta.loc[uid].to_dict() if uid in series_meta.index else {}
        row["side_predicted"] = side
        rows.append(knee_dataset.series_metadata(pd.Series(row)))
    return torch.from_numpy(np.stack(rows)).unsqueeze(0).to(device)


def frozen_predict(volumes, meta, mask):
    features = []
    for volume in volumes:
        slices = net.stack_2p5d(torch.from_numpy(volume)[None])
        chunks = [embed.embed_slices(encoder, slices[i:i + BATCH_SLICES].to(device)).float().cpu()
                  for i in range(0, len(slices), BATCH_SLICES)]
        features.append(torch.cat(chunks))
    stacked = torch.stack(features).unsqueeze(0).to(device)
    with torch.no_grad():
        return np.mean([torch.sigmoid(m.forward_features(stacked, meta, mask))
                        .cpu().numpy()[0] for m in frozen_models], axis=0)


def tuned_predict(volumes, uids, side):
    # Mirrored and subsampled exactly as during training.
    picked = volumes[:TUNED_MAX_SERIES]
    picked_uids = uids[:TUNED_MAX_SERIES]
    mirrored = []
    for volume, uid in zip(picked, picked_uids):
        plane = series_meta.loc[uid]["Anatomical_Plane"] if uid in series_meta.index else None
        mirrored.append(knee_dataset.mirror_volume(volume, side, plane)[KEEP])

    batch = torch.from_numpy(np.stack(mirrored)).unsqueeze(0).to(device)
    meta = meta_tensor(picked_uids, side)
    mask = torch.ones(1, len(mirrored), dtype=torch.bool, device=device)
    with torch.no_grad(), torch.amp.autocast("cuda"):
        return np.mean([torch.sigmoid(m(batch, meta, mask).float()).cpu().numpy()[0]
                        for m in tuned_models], axis=0)


ids, frozen_rows, tuned_rows, failures = [], [], [], 0

# Timed separately from setup. Model loading is a fixed cost, and amortising
# it over three public studies inflates the per-study figure enormously --
# it is noise against a real test set.
loop_start = time.time()

for study in tqdm(studies, desc="studies", unit="study"):
    try:
        volumes, uids = build_volumes(study)
        if not volumes:
            raise ValueError("no readable series")
        side = predict_side(volumes)

        meta = meta_tensor(uids, side)
        mask = torch.ones(1, len(volumes), dtype=torch.bool, device=device)

        frozen_rows.append(frozen_predict(volumes, meta, mask)
                           if (frozen_models and USE_FROZEN) else None)
        tuned_rows.append(tuned_predict(volumes, uids, side)
                          if (tuned_models and USE_TUNED) else None)
    except Exception as exc:  # noqa: BLE001
        failures += 1
        print(f"  fallback for {study[-12:]}: {str(exc)[:80]}")
        prior = np.array([PRIORS[n] for n in config.LABELS])
        frozen_rows.append(prior)
        tuned_rows.append(prior)
    ids.append(study)

loop_elapsed = time.time() - loop_start
elapsed = time.time() - START
setup = elapsed - loop_elapsed
print(f"\\n{len(ids)} studies in {loop_elapsed / 60:.1f} min "
      f"({failures} fallbacks); setup took {setup / 60:.1f} min")
print(f"per study: {loop_elapsed / max(len(ids), 1):.2f}s")
""")

    nb.md("""
## Blend by rank and write

Ranks are taken per finding across the test set, so each model contributes its
ordering rather than its confidence. With a single test study ranking is
degenerate, so the raw probabilities are used instead.
""")

    nb.code("""
def to_ranks(matrix):
    ranks = np.empty_like(matrix, dtype=float)
    for column in range(matrix.shape[1]):
        order = np.argsort(matrix[:, column], kind="mergesort")
        ranks[order, column] = np.arange(len(matrix))
    return ranks / max(len(matrix) - 1, 1)


have_frozen = frozen_models and USE_FROZEN and all(r is not None for r in frozen_rows)
have_tuned = tuned_models and USE_TUNED and all(r is not None for r in tuned_rows)
frozen_matrix = np.stack(frozen_rows) if have_frozen else None
tuned_matrix = np.stack(tuned_rows) if have_tuned else None
print(f"blending: frozen={have_frozen} tuned={have_tuned} weight={BLEND}")

if frozen_matrix is not None and tuned_matrix is not None and len(ids) > 1:
    blended = BLEND * to_ranks(tuned_matrix) + (1 - BLEND) * to_ranks(frozen_matrix)
elif frozen_matrix is not None and tuned_matrix is not None:
    blended = BLEND * tuned_matrix + (1 - BLEND) * frozen_matrix
else:
    blended = tuned_matrix if tuned_matrix is not None else frozen_matrix

# pd.DataFrame(None, columns=...) is an empty frame, not an error, and the
# insert below would then pad it into rows of NaN. Say so here instead.
if blended is None:
    raise SystemExit(
        "no predictions to write: every row of the active path was None. "
        "Check the fold counts printed above.")

submission = pd.DataFrame(blended, columns=config.LABELS)
submission.insert(0, config.ID_COL, ids)
submission = submission[config.SUBMISSION_COLUMNS]
submission.to_csv("/kaggle/working/submission.csv", index=False)

sample = pd.read_csv(find_competition_file("sample_submission.csv"))
assert list(submission.columns) == list(sample.columns), "header mismatch"
assert len(submission) == len(test_csv), "row count does not match test.csv"
assert submission[config.ID_COL].is_unique, "duplicate study ids"
values = submission[config.LABELS]
assert values.notna().all().all(), "missing predictions"
assert ((values >= 0) & (values <= 1)).all().all(), "predictions out of range"

print(submission.head().to_string())
# Setup counted once, inference scaled -- rather than scaling both.
projected = setup + loop_elapsed / max(len(ids), 1) * 1500
print(f"\\nprojected for 1,500 studies: {projected / 3600:.2f} h "
      f"(cap 9 h, efficiency penalty {projected / 32400:.3f})")
""")

    nb.md("""
## Next

Submit, and compare against the two single-model scores. If the blend does not
beat the better of them, `BLEND` is the knob -- and the honest way to set it is
the leaderboard, not the 58 gold studies, which have already been used once for
checkpoint selection.

Running both paths roughly doubles inference cost, which the Efficiency Prize
charges for. Worth checking whether the blend earns its runtime on that metric
as well as on AUC.
""")

    nb.write(NOTEBOOKS / "10_submit_ensemble.ipynb")


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
    build_export_notebook()
    build_ensemble_notebook()


if __name__ == "__main__":
    main()
