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
    # Torch-free, so it costs nothing to carry: the conditional-AUC metric is
    # needed by the training loop's validation hook (notebook 08) and by the
    # error-analysis notebook, and a second copy of it would be a second
    # definition of the number synovitis is judged on.
    "src/model/conditional.py",
    "src/model/augment.py",
    "src/model/averaging.py",
    "src/model/splits.py",
    "src/labels/__init__.py",
]

# Only notebook 08 trains, and only it needs Lightning. The module imports
# lightning at import time, which the submission notebook has no reason to
# require.
# Only notebook 11 renders anything, so the inspection module travels only
# there rather than adding 9 KB of source to every notebook's bootstrap.
INSPECT_MODULES = MODEL_MODULES + ["src/model/inspect.py"]

# Notebook 08 additionally needs the label package: the silence reweighting
# lives in `fuse` beside WEIGHT_SILENT and the rationale for it, rather than
# being reimplemented in the model package where the two could drift. `fuse`
# imports `calibrate` and `lexicon` at module scope, so they come too -- 1,000
# lines of pure Python with no dependency beyond numpy and pandas.
FINETUNE_MODULES = MODEL_MODULES + [
    "src/labels/calibrate.py",
    "src/labels/lexicon.py",
    "src/labels/fuse.py",
    "src/model/lightning.py",
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
    from src.model import (augment as knee_augment, averaging, conditional,
                           embed, dataset as knee_dataset, net, splits)
# Only the fine-tuning notebook materialises these; lightning imports its
# framework at module scope and fuse pulls in the two extractors.
if pathlib.Path("src/model/inspect.py").exists():
    from src.model import inspect as knee_inspect
if pathlib.Path("src/labels/fuse.py").exists():
    from src.labels import fuse
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

# The gold predictions themselves, keyed by UID, not just their AUCs. Stage 0a
# of the synovitis plan needs this file and nothing else, and until it existed
# the frozen path's numbers could not be compared to the fine-tuned path's on
# anything but a printed macro. Small -- 58 rows.
GOLD_PREDICTIONS = "/kaggle/working/gold_predictions_frozen.parquet"
pd.DataFrame(gold_matrix, columns=config.LABELS).assign(
    **{config.ID_COL: gold_ids})[config.SUBMISSION_COLUMNS].to_parquet(
        GOLD_PREDICTIONS, index=False)
print("wrote", GOLD_PREDICTIONS)

pd.DataFrame(oof_matrix, columns=config.LABELS).assign(
    **{config.ID_COL: oof_ids})[config.SUBMISSION_COLUMNS].to_parquet(
        "/kaggle/working/oof_predictions_frozen.parquet", index=False)
""")

    nb.md("""
## Stage 0a — the gate

The frozen path is the decisive read on whether synovitis is in the pixels at
all. It has only a small aggregation head, so it cannot have fit the mined
label noise the way a fine-tuned trunk can: if its general-purpose features
discriminate synovitis *within* an effusion stratum, the signal is there and
the fine-tuned model is failing to use it. If they do not, the frozen path's
higher pooled number (0.746 against 0.701) is a better effusion proxy and
nothing more.

**Pooled AUC cannot answer this.** On these 58 studies the gold effusion label
alone scores 0.698 as a synovitis classifier. The number below is conditional
AUC — concordance restricted to pairs from the same effusion stratum — with
the cutoffs pre-registered in `conditional.gate_reading`, not chosen here.

Attach the fine-tuned run's `gold_predictions.parquet` as well and the
comparison is made paired, on the same studies, which is the only way a
difference of this size is readable at n=58.
""")

    nb.code("""
FINDING, STRATIFY_BY = "Synovitis", "Effusion"

y = gold_truth[FINDING].to_numpy().astype(int)
scores_syn = gold_matrix[:, config.LABELS.index(FINDING)]
strata = gold_truth[STRATIFY_BY].to_numpy().astype(int)

stats = conditional.bootstrap_conditional(y, scores_syn, strata, draws=4000)
counts = conditional.pair_counts(y, strata)
reading = conditional.gate_reading(stats)
why, action = conditional.GATE_ACTIONS[reading]

print(f"conditional  {stats['auc']:.3f}  "
      f"95% CI [{stats['lo']:.3f}, {stats['hi']:.3f}]  "
      f"P(<= 0.5) = {stats['p_at_or_below_chance']:.3f}  "
      f"({counts['within']} within-stratum pairs)")
print(f"pooled       {conditional.auc(y, scores_syn):.3f}")
print(f"the effusion head, used directly as a {FINDING} classifier: "
      f"{conditional.auc(y, gold_matrix[:, config.LABELS.index(STRATIFY_BY)]):.3f}")
print()
print(f"GATE: {reading.upper()} -- {why}")
print(f"  -> {action}")
print()

# Paired against the fine-tuned path, if its predictions are attached. An
# unpaired comparison of 0.746 and 0.701 is 0.66 of a standard deviation on
# this set and settles nothing; the same 58 studies resampled together do.
tuned_paths = find_inputs("gold_predictions.parquet")
if not tuned_paths:
    print("no fine-tuned gold_predictions.parquet attached -- attach one to "
          "get the paired comparison the plan asks for (§2.2)")
else:
    tuned = pd.read_parquet(tuned_paths[0]).set_index(config.ID_COL)
    shared = [s for s in gold_ids if s in tuned.index]
    keep = [i for i, s in enumerate(gold_ids) if s in tuned.index]
    tuned_syn = tuned.loc[shared, FINDING].to_numpy()
    print(f"paired against {os.path.basename(tuned_paths[0])} "
          f"({len(shared)} shared studies)")
    for label, kwargs in (("conditional", {"strata": strata[keep]}),
                          ("pooled", {"strata": None})):
        delta = conditional.paired_bootstrap(
            y[keep], scores_syn[keep], tuned_syn, draws=4000, **kwargs)
        print(f"  {label:12s} {delta['delta']:+.3f}  "
              f"95% CI [{delta['lo']:+.3f}, {delta['hi']:+.3f}]  "
              f"P(frozen not better) = {delta['p_a_not_better']:.3f}")
    print()
    print("Ship the frozen path for this finding only if it wins on BOTH.")
    print("A win on pooled alone is a better effusion proxy, not a better")
    print("synovitis model.")
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

# ---------------------------------------------------------------------------
# Colab staging, shared by every Colab copy
# ---------------------------------------------------------------------------
#
# ONE implementation of "how the inputs arrive", emitted into both
# 08_finetune_colab and 10_submit_ensemble_colab. It was briefly two
# hand-trimmed copies, which is how the trimmed one loses a guard the other
# still has -- so the differences between the notebooks are the arguments
# below and nothing else.
#
# Sentinels rather than an f-string or .format(): the cell body is full of
# braces of its own ({pattern}, {name}, f-strings) and every one of them would
# have to be doubled.

COLAB_STAGE_TEMPLATE = """@@HEADER@@
!pip install -q @@PIP@@

import glob
import os
import shutil

import kagglehub

from kagglehub.config import get_kaggle_credentials, set_kaggle_api_token

# --- Authentication ---------------------------------------------------------
# Every source below needs it: the shards and knee-labels are private, and the
# competition additionally needs the rules accepted on this account. An
# unauthenticated call does not say so -- it fails inside the resolver as a
# 401 or a bare 404 on a handle that plainly exists -- so credentials are
# resolved and checked here, before anything touches the network.
#
# kagglehub reads, in order (kagglehub/config.py, get_kaggle_credentials):
#   1. kagglehub.login() or set_kaggle_api_token() earlier in this session
#   2. env KAGGLE_API_TOKEN  -- the KGAT_... bearer token, or a path to a file
#      holding it
#   3. env KAGGLE_USERNAME + KAGGLE_KEY   (legacy pair)
#   4. ~/.kaggle/kaggle.json              (legacy pair)
#   5. Colab Secrets, checked last: a secret named KAGGLE_API_TOKEN first,
#      then KAGGLE_USERNAME + KAGGLE_KEY
#
# EASIEST ON COLAB, and no code to edit: the key icon in the left sidebar ->
# "Add new secret" -> name it KAGGLE_API_TOKEN, paste the KGAT_... string
# (it is in ~/.kaggle/access_token on the machine that runs the Kaggle CLI)
# -> switch "Notebook access" on for this notebook. The bearer token is
# enough; a classic username/key pair is not required.
if get_kaggle_credentials() is None:
    # getpass, NOT kagglehub.login(). Inside a notebook, login() renders an
    # ipywidgets form and returns immediately; the token is stored by the
    # button's on_click callback, which cannot run until this cell has already
    # finished. So the login appears to succeed while this cell sees nothing,
    # and the credential it stored lives only in a module global that a
    # runtime restart discards -- which is how the same widget can be filled
    # in twice and still leave the next run unauthenticated. getpass blocks
    # until the token is in hand, so everything below is synchronous.
    import getpass

    token = getpass.getpass(
        "Kaggle API token (KGAT_...), input hidden; Enter to abort: ").strip()
    if not token:
        raise SystemExit(
            "No token entered. The durable fix is a Colab secret named "
            "KAGGLE_API_TOKEN with notebook access enabled -- kagglehub reads "
            "it automatically, so a reconnected runtime needs no typing.")

    # Both on purpose. set_kaggle_api_token fills the module global this
    # process reads; the environment variable is what survives kagglehub being
    # re-imported, and is checked before Colab secrets.
    os.environ["KAGGLE_API_TOKEN"] = token
    set_kaggle_api_token(token)

# Synchronous, and it fails loudly on a bad token rather than at the first
# mount, where the error is a bare 404 on a handle that plainly exists.
print("authenticated as:", kagglehub.whoami()["username"])

# Leave the cache on local disk. Pointed at Drive, the FUSE layer stages every
# byte on the VM's disk first and fills it long before Drive notices.
os.environ.pop("KAGGLEHUB_CACHE", None)

@@WORKING_NOTE@@
os.makedirs("/kaggle/working", exist_ok=True)
os.makedirs("/kaggle/input", exist_ok=True)


def stage(path, name):
    # Make anything that had to be downloaded look like a mount, so the one
    # find_inputs implementation works on both platforms.
    destination = f"/kaggle/input/{name}"
    if not os.path.exists(destination):
        os.symlink(path, destination)
    return destination


# Each shard is wanted as a DATASET, which Colab mounts without copying
# anything. Until the one-time conversion has been run they exist only as
# notebook outputs, which have no Colab resolver and must be downloaded --
# ~9.7 GB each, once per session, because the VM is ephemeral.
#
# That download is survivable. kagglehub resumes a partial transfer with an
# HTTP Range request and verifies the MD5 afterwards, so the ~5.5 GB
# IncompleteRead ceiling recorded in PROJECT.md -- measured against the older
# CLI, which had no resume -- costs retries rather than the whole file. Each
# retry picks up where the last stopped.
CACHE_DATASETS = (
    ("pssv27/knee-cache-shard-1", "pssv27/knee-cache-224-01-shard-1"),
    ("pssv27/knee-cache-shard-2", "pssv27/knee-cache-224-01-shard-2"),
    ("pssv27/knee-cache-shard-3", "pssv27/knee-cache-224-01-shard-3"),
)
DOWNLOAD_ATTEMPTS = 10

free_gb = shutil.disk_usage("/").free / 1e9
print(f"local disk free: {free_gb:.0f} GB "
      f"({'mounts need almost none' if free_gb > 40 else 'TIGHT for downloads'})")


def fetch_shard(dataset_handle, notebook_handle):
    # Prefer the mount. Fall back to the copy, and keep resuming it.
    try:
        path = kagglehub.dataset_download(dataset_handle)
        print(f"  mounted  {dataset_handle}")
        return path, dataset_handle.split("/")[1]
    except Exception as exc:  # noqa: BLE001 - any failure means "not a dataset yet"
        print(f"  {dataset_handle}: {type(exc).__name__} -- "
              f"downloading {notebook_handle} instead")

    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            path = kagglehub.notebook_output_download(notebook_handle)
            print(f"  downloaded {notebook_handle}")
            return path, notebook_handle.split("/")[1]
        except Exception as exc:  # noqa: BLE001 - resume and try again
            print(f"    attempt {attempt}/{DOWNLOAD_ATTEMPTS} "
                  f"{type(exc).__name__}: {exc}")
    raise SystemExit(
        f"{notebook_handle} did not complete in {DOWNLOAD_ATTEMPTS} attempts. "
        "Re-running this cell resumes from what already arrived.")


# --- A Drive copy of the kagglehub cache, if a previous session left one ----
# Copying from Drive beats fetching from Kaggle again, but it must be a COPY
# on to local disk, never used in place: the shards are opened with
# np.load(..., mmap_mode="r") and read by random access from several
# dataloader workers, and over the Drive FUSE mount every page fault is a
# round trip. The client's answer to random reads is to pull the whole 9.7 GB
# file into its local cache anyway, so training in place pays the transfer
# regardless while filling the VM disk -- the failure already seen once.
#
# Set to None to skip Drive entirely and go straight to Kaggle.
DRIVE_CACHE = "/content/drive/MyDrive/kagglehub"

# What the notebook below actually opens, by the same patterns find_inputs
# globs. Anything missing after this cell is fetched from Kaggle instead.
REQUIRED = ("cache_train_series_*.npy", "manifest_train_series_*.parquet",
            "train_labels.parquet", "side_predictions.parquet")
OPTIONAL = ("report_embeddings.parquet",)   # report loss only; lambdas are 0.0

LOCAL = "/kaggle/input/from_drive"


def copy_from_drive(root, patterns):
    # Search by name rather than by cache layout: what matters is the file the
    # notebook opens, not which handle it arrived under.
    os.makedirs(LOCAL, exist_ok=True)
    found = []
    for pattern in patterns:
        for source in sorted(glob.glob(f"{root}/**/{pattern}", recursive=True)):
            name = os.path.basename(source)
            target = os.path.join(LOCAL, name)
            size = os.path.getsize(source)
            if os.path.exists(target) and os.path.getsize(target) == size:
                print(f"  already local  {name}")
                found.append(name)
                continue
            print(f"  copying        {name}  ({size / 1e9:.2f} GB)", flush=True)
            shutil.copy2(source, target)
            # A truncated .npy memmaps without complaint and trains on
            # whatever bytes are there, so compare sizes rather than trust it.
            if os.path.getsize(target) != size:
                raise SystemExit(f"{name} copied short -- delete {target} "
                                 "and re-run this cell")
            found.append(name)
    return found


if DRIVE_CACHE and not os.path.isdir("/content/drive/MyDrive"):
    from google.colab import drive

    drive.mount("/content/drive")

if DRIVE_CACHE and os.path.isdir(DRIVE_CACHE):
    print(f"Drive cache found at {DRIVE_CACHE}; copying to local disk")
    copy_from_drive(DRIVE_CACHE, REQUIRED + OPTIONAL)
elif DRIVE_CACHE:
    print(f"no Drive cache at {DRIVE_CACHE} (mount Drive first if you have "
          "one); falling back to Kaggle")


def missing(patterns):
    absent = []
    for pattern in patterns:
        if not glob.glob(f"/kaggle/input/**/{pattern}", recursive=True):
            absent.append(pattern)
    return absent


# --- Whatever Drive did not supply, fetch from Kaggle -----------------------
if missing(("cache_train_series_*.npy",)):
    for dataset_handle, notebook_handle in CACHE_DATASETS:
        stage(*fetch_shard(dataset_handle, notebook_handle))

if missing(("train_labels.parquet",)):
    stage(kagglehub.dataset_download("pssv27/knee-labels"), "knee-labels")

if missing(("side_predictions.parquet",)):
    # A notebook output, so this one downloads. It is small.
    stage(kagglehub.notebook_output_download("pssv27/knee-side-classifier"),
          "knee-side-classifier")

# find_competition_file checks data/<name> first. `path=` fetches one file;
# without it, competition_download pulls the entire 570 GB DICOM archive.
os.makedirs("data", exist_ok=True)
for name in ("train.csv", "train_series.csv"):
    if os.path.exists(f"data/{name}"):
        continue
    local = glob.glob(f"{DRIVE_CACHE}/**/{name}", recursive=True) if DRIVE_CACHE else []
    source = local[0] if local else kagglehub.competition_download(
        "rsna-knee-abnormality-detection", path=name)
    shutil.copy(source, f"data/{name}")
for name in ("train.csv", "train_series.csv"):
    print(f"  {name}: {os.path.getsize(f'data/{name}') // 1024} KB")

# --- Verify, because every failure mode here is silent ----------------------
still_absent = missing(REQUIRED)
if still_absent:
    raise SystemExit(f"not staged: {still_absent}")

shards = sorted(glob.glob("/kaggle/input/**/cache_train_series_*.npy",
                          recursive=True))
print(f"\\nshards staged: {len(shards)}")
for path in shards:
    print(f"  {os.path.basename(path)}  {os.path.getsize(path) / 1e9:.1f} GB")
if len(shards) < 3:
@@SHORT_SHARDS@@
@@TAIL@@"""


def colab_stage_cell(pip="timm lightning kagglehub", header=None,
                     working_note=None, short_shards=None, tail=""):
    """The Colab input-staging cell.

    Mount first, download second, Drive third -- see the comments in the body.
    Only four things vary between notebooks: which packages to install, what
    the cell says it is doing, what /kaggle/working is for, and what a
    short-shard set means for that notebook. `tail` appends notebook-specific
    staging after the shared part.
    """
    header = header or (
        "# --- Colab staging: mount the Kaggle data, then run the cells below "
        "unchanged.\n"
        "# Nothing here writes to Drive. dataset_download MOUNTS on Colab (no "
        "copy) and\n"
        "# lands under /kaggle/input/<slug>, which is exactly where "
        "find_inputs looks,\n"
        "# so the rest of this notebook is byte-identical to the Kaggle one.")
    working_note = working_note or (
        "# Notebook 08 writes the checkpoint, metrics.csv and the OOF dump "
        "here.")
    short_shards = short_shards or (
        '    print("WARNING: fewer than three shards. A short run looks '
        'entirely "\n'
        '          "healthy and trains on part of the data -- the assemble '
        'cell must "\n'
        '          "report 4,349 studies.")')
    cell = COLAB_STAGE_TEMPLATE
    for sentinel, value in (("@@HEADER@@", header), ("@@PIP@@", pip),
                            ("@@WORKING_NOTE@@", working_note),
                            ("@@SHORT_SHARDS@@", short_shards),
                            ("@@TAIL@@", tail)):
        cell = cell.replace(sentinel, value)
    return cell


def build_finetune_notebook(colab=False):
    nb = Notebook(gpu=True)

    nb.md("""
# 08 — Fine-Tuned Backbone

Trains a ConvNeXt-T end to end instead of reading frozen DINOv2 features. Prior
RSNA competitions were won by fine-tuned ImageNet CNNs, so this is the arm most
likely to top the leaderboard, and the one the frozen path exists to be
measured against.
""")

    if colab:
        nb.md("""
**This is the Colab copy of notebook 08.** Generated from the same source as
`08_finetune.ipynb` -- edit `tools/make_notebook.py` and regenerate, never
this file. Everything below the staging cell is identical to the Kaggle
notebook, so a result from either is comparable with the other.

**Why a copy exists at all.** The Kaggle GPU quota is 25 h/week and one
session at a time; when it is spent, Colab is the only GPU left. The blocker
was never compute, it was the 29 GB cache -- and pulling that over HTTP fails
at ~5.5 GB with IncompleteRead, consistently.

**The route that works is a mount, not a download.** `kagglehub` resolves a
Kaggle *dataset* on Colab by asking Colab's backend to mount it at
`/kaggle/input/<slug>` -- no copy, no Drive, no transfer. That is the same
path Kaggle itself uses, which is why `find_inputs` needs no change. There is
no such resolver for notebook *outputs* or for competitions, so:

- the three cache shards must exist as **datasets** (one-time conversion, run
  `kagglehub.dataset_upload()` from a Kaggle **CPU** notebook -- CPU sessions
  do not touch the GPU quota),
- `knee-side-classifier` is a notebook output and downloads, which is fine at
  its size,
- the competition contributes exactly two CSVs, fetched per-file with
  `path=`. Never call `competition_download` without it: the bare call pulls
  the full 570 GB DICOM archive.

**Authentication is required and is checked first.** The shards and
`knee-labels` are private, and the competition needs its rules accepted on
this account; an unauthenticated call fails as a 401 or a bare 404 on a
handle that plainly exists, which names nothing. Add a Colab secret called
`KAGGLE_API_TOKEN` holding the `KGAT_...` bearer token (the key icon in the
sidebar, with notebook access enabled) and the staging cell needs no edit.
kagglehub checks Colab Secrets last, after `KAGGLE_API_TOKEN` in the
environment, the legacy `KAGGLE_USERNAME`/`KAGGLE_KEY` pair, and
`~/.kaggle/kaggle.json` -- the bearer token alone is enough at any of them.

**Do not point `KAGGLEHUB_CACHE` at Drive.** Every byte written through the
Drive FUSE mount is staged on the VM's local disk first, and kagglehub writes
each archive twice (download, then extract beside it). That fills a ~100 GB
VM regardless of how much Drive quota exists.

**One GPU is expected here**, so `STRATEGY` resolves to `"auto"` rather than
`ddp_notebook`: there is no fork, and the CUDA-poisoning trap that governs
the Kaggle run does not apply. `ACCUM_STEPS` adapts to hold the effective
batch constant, so the run stays numerically comparable to a two-GPU one.

**`/kaggle/working` is ephemeral on Colab.** The last cell copies the
checkpoint, `metrics.csv` and the OOF dump to Drive; those are small. Run it.
""")
    else:
        nb.md("""
**Runs on Kaggle.** Fine-tuning needs raw volumes, and one cache shard is
9.7 GB. Pulling that to Colab over HTTP failed twice at ~5.5 GB with
IncompleteRead -- a consistent ceiling rather than a flaky connection. On
Kaggle the cache is already mounted and nothing transfers. A T4 is slower than
an A100, but not slower than a download that never finishes.

If the GPU quota is spent, `08_finetune_colab.ipynb` is the same notebook with
a staging cell that *mounts* the shards on Colab instead of copying them.

**Attach:** one cache dataset, `knee-labels`, `knee_side_classifier`, and the
competition data. GPU on. The auxiliary report loss additionally wants
`report_embeddings.parquet` (built locally by `src/labels/report_embed.py`)
in the `knee-labels` dataset; without it that loss silently disables, which
the assemble cell prints.
""")

    nb.md("""

**This run measures before it commits.** One study is ~5.5 series x 24 slices
= ~130 slice-forwards, so a full epoch over 4,349 studies is ~570,000. The
training cell reports actual throughput and projects the full sweep. Read that
before launching five folds.

Subsampling controls the cost, and both knobs are real trade-offs:

    SLICES_PER_SERIES  16 of 24     the reference geometry (gold 0.861, LB 0.889)
    MAX_SERIES          5 of ~5.5   6 was measured worse (LB 0.885 vs 0.888)

The capacity ladder is finished (0.845 -> 0.869 -> 0.889 -> 0.891) and 16
slices is where experiments iterate: 24 holds the best LB by +0.002 and overfit
to get it (val peaked epoch 7, gold fell 0.865 -> 0.842), so it stays reserved
for a final AUC-slot run.

**This run is P0.4 and the series cap is the only change in it.** 5 -> 6 lifts
mean series/study 6.7% and cuts truncation from 32.5% of studies to 16.7% --
about half the remaining headroom on this axis. Compare against the P1 run at
16 x 5 (val 0.874 / gold 0.882 at epoch 10).

**Budget ~1 h per fold.** 4.7 min/epoch measured on an A100 at 16 x 5, twelve
epochs. Read the throughput line the training cell prints after epoch 1 before
walking away. If it OOMs, drop `MAX_SERIES` to 4.

**Mirroring is used here**, unlike the frozen path: raw volumes can be
reflected, so left knees are mapped onto the canonical side along the axis the
acquisition plane dictates. That makes this a comparison of two things at once,
backbone and laterality handling, which is worth remembering when reading the
result.
""")

    nb.md("## 1. Setup")

    if colab:
        nb.code(colab_stage_cell())

    setup_source = """
!pip install -q timm

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
"""

    if colab:
        # One GPU is the expected shape on Colab, so the two-GPU advice would
        # be noise; and the accelerator lives in a different menu.
        setup_source = setup_source.replace(
            'raise SystemExit("Settings -> Accelerator -> GPU T4 x2")',
            'raise SystemExit("Runtime -> Change runtime type -> GPU")')
        setup_source = setup_source.replace(
            '''if DEVICES < 2:
    print("NOTE: one GPU visible. Settings -> Accelerator -> GPU T4 x2 "
          "roughly halves wall-clock. The run is correct either way -- "
          "ACCUM_STEPS below adapts to hold the effective batch constant.")''',
            '''if DEVICES < 2:
    print("one GPU: STRATEGY resolves to 'auto', so there is no fork and no "
          "CUDA-poisoning trap here. ACCUM_STEPS adapts to hold the effective "
          "batch constant, so this stays comparable to a two-GPU run.")''')
    nb.code(setup_source)

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

# SILENCE REWEIGHTING -- the Synovitis fix.
#
# 82.2% of studies say nothing about Synovitis. Its mined target averages
# 0.087; its true prevalence on the gold 58 is 46.6%. So ~1,666 of the 3,576
# silent studies are true positives being trained toward zero, and because
# silence is so common for this finding they carry **57% of its gradient**.
# The model is taught to rank real positives low, and duly scores 0.682 on
# gold -- the only finding where fine-tuning is WORSE than the retired frozen
# path (0.746). That is what fitting label noise harder looks like.
#
# The fix multiplies the silent weight by (1 - silence_rate), measured on the
# 4,349 mined studies. One rule, no free parameters, no gold. Silence about
# ACL (9.2% of studies) keeps nearly its full weight, because a report that
# discusses ligaments and omits the ACL is saying something; silence about
# Synovitis falls to 0.044 because a corpus that mentions it once in six
# studies is not. Targets are untouched -- fabricating P(positive | silent) as
# a target was measured worse on every variant tried, and that stands.
#
# Near-neutral for 8 of 12 findings by construction: silence already carries
# only 3-5% of the gradient where reports are thorough. Synovitis goes 57% ->
# 19%.
# MEASURED 2026-08-29 AND FAILED. Synovitis 0.682 -> 0.694 against a gate of
# 0.78; per-finding gold SE is 0.045-0.08, so +0.012 is noise. The correlation
# between how much weight a finding lost and how its AUC moved is -0.08 --
# i.e. nothing happened in either direction, to any finding.
#
# What that rules out: silent false negatives dragging the ranking down was
# NOT the mechanism. Removing 57% -> 19% of Synovitis's gradient from silent
# studies changed nothing, so the studies that remain must be where the
# problem is -- the 18% of reports that DO mention synovitis are presumably
# the severe cases, and a model trained on those cannot rank the mild ones
# that the radiologist graded positive. Loss weighting cannot fix a biased
# positive sample; only image-derived labels for the silent studies can.
SILENCE_REWEIGHT = False

if SILENCE_REWEIGHT:
    _rates = fuse.silence_rates(labels)
    labels = fuse.rescale_silent_weights(labels)
    print("silence reweighting ON")
    for _n in sorted(config.LABELS, key=lambda x: -_rates[x])[:3]:
        print(f"  {_n:18s} silent {_rates[_n]:5.1%} -> weight "
              f"{fuse.WEIGHT_SILENT * (1 - _rates[_n]):.3f}")
else:
    print("silence reweighting OFF (reference labels)")
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

# GROUPED FOLDS -- leave-scanner-out instead of leave-study-out.
#
# The corpus spans 16-19 sites, and a CNN reads scanner signature as readily
# as it reads pathology. A random split puts the same equipment on both sides
# of the fold and reports an accuracy that will not survive a new site: the
# figure in the plan is ~0.053 of inflation, with one measured model showing a
# 0.136 grouped-vs-random gap. Nothing about it raises.
#
# DEFAULT OFF, deliberately. Turning it on re-cuts every fold, and every
# reference number this project has (fold 1: val 0.872 / gold 0.861 / LB
# 0.889) belongs to the ungrouped split -- so an experiment compared against
# those numbers must NOT flip this. Turn it on when the five-fold ensemble is
# rebuilt from scratch, and expect the validation numbers to drop: that drop
# is the inflation being removed, not a regression.
#
# Needs notebook 01's index.parquet attached, for the scanner headers. It is
# not part of the cache dataset, so this falls back with a warning rather than
# failing a training run over a missing input.
GROUPED_FOLDS = False

groups = None
if GROUPED_FOLDS:
    _index_files = find_inputs("index_train.parquet") + find_inputs("index.parquet")
    if _index_files:
        groups = splits.scanner_key(pd.read_parquet(_index_files[0]))
        print(f"grouped folds: {groups.nunique()} scanner groups over "
              f"{len(groups)} studies")
    else:
        print("GROUPED_FOLDS is on but no index parquet was found -- attach "
              "notebook 01's output. Falling back to the ungrouped split.")

folded = splits.make_folds(labels, groups=groups)
splits.assert_no_gold_leakage(folded["StudyInstanceUID"], set(gold["StudyInstanceUID"]))
if groups is not None:
    _summary = splits.summarise(folded)
    print(_summary[["studies"] + config.LABELS[:4]].to_string())
    print("  fold sizes are uneven by construction -- a scanner group cannot "
          "be split. Check the positive rates before trusting a comparison.")

# Capacity ladder, step 2. Doubling slices was worth +0.024 on the
# leaderboard, so the other axis the study is subsampled on is the next thing
# to test. Most studies hold ~5.5 series and three were being kept; PF OA is
# one of only two findings where this model still loses to frozen DINOv2
# features, and the patella may simply not be in the three that survive.
#
# CURRENT: 5. P0.4 raised this to 6 and it was MEASURED NEGATIVE -- LB 0.885
# against the same configuration's 0.888 at cap 5, with gold -0.005 and +4%
# runtime. Reverted 2026-08-29. The series axis is closed: the plan put total
# remaining headroom at +14% and 5 -> 6 was half of it.
#
# Historical note kept because the gate passed a null:
#
# 5 -> 6 buys about half the remaining headroom on the series axis: mean
# series/study +6.7%, truncation down from 32.5% of studies to 16.7%. The mean
# is 5.53 series per study, so uncapping would add only another ~7% for the
# last 2.3%. Cheap: a load-time knob, every series already cached.
#
# Held at 5 through the P1 run so gold differed from the reference on run
# length alone. P1 is done -- no divergence, selection cost 0.000 -- so the
# hold is released.
#
# THE COMPARISON is the P1 run: fold 1, 20 epochs, 16 x 5, val 0.874 / gold
# 0.882 at epoch 10. The plan's gate is "gold up or flat with no cost blowup".
# Read val as well: on 58 gold studies 0.02 is one standard error, and val has
# ~870.
#
# Notebook 10's TUNED_MAX_SERIES tracks this constant; a test asserts it.
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
# CURRENT: 16, the reference geometry -- fold 1 at 16 x 5 is val 0.872 /
# gold 0.861 / LB 0.889, the only split with a measured baseline.
#
# **24 is not the better setting, despite holding the best LB.** It scored
# 0.891 against 16's 0.889 -- +0.002 for +57% training and +48% inference --
# and it overfit while doing it: val peaked at epoch 7 and fell to 0.871 while
# **gold fell 0.865 -> 0.842**. Both curves down together is capacity against
# 4,349 studies, not headroom.
#
# That matters specifically for the run this notebook is currently set up for.
# P1 asks whether a late gold decline comes from a biased selector rather than
# from overfitting; at 24 slices ConvNeXt has already produced the overfitting
# shape once, so the two explanations are indistinguishable there. At 16 the
# late decay was mild (0.854 -> 0.849 over epochs 9-12, and that was with an
# auxiliary loss on), which is the condition under which the answer is legible.
#
# The plan agrees: "past ~16-24 slices, expect flattening ... treat slice count
# as roughly saturated and spend compute elsewhere". The headroom for the small
# structures is resolution, which needs a cache rebuild at CACHE_SIZE 288 or
# 384 -- not a change here.
#
# Notebook 10's TUNED_SLICES tracks this constant and a test asserts the two
# agree, because a mismatch scores worse without erring.
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

# Notebook 10's TUNED_BACKBONE must match this, along with the slice and
# series counts. `net.build_backbone` handles the ViT-family input-size
# argument, so a DINOv2 name works here unchanged.
#
# CURRENT: the incumbent, "convnext_tiny.fb_in22k_ft_in1k" (768 features,
# 27.8M params). FEATURE_DIM printed below is the check that an edit here
# reached the running notebook: 768 features and 27.8M params is ConvNeXt-T.
#
# **The DINOv2 axis is closed.** Fine-tuned DINOv2-S measured gold 0.820
# against fold 1's 0.861 gate, and at 24 slices it converged level with
# ConvNeXt while losing to its own frozen self on three findings. That matches
# the published evidence rather than contradicting it: DINOv2 transfers "on
# par or slightly better" than supervised CNNs on medical *classification*
# (Bassi et al., arXiv:2312.02366), its gains are a segmentation story, and
# even DINOv3 only overtakes ImageNet initialisation at 512px -- where
# ConvNeXt-B still beat ViT-B/16 (arXiv:2510.07191). Do not re-run it.
#
# The backbone experiments still open, in order: ConvNeXt-V2-Base
# ("convnextv2_base.fcmae_ft_in22k_in1k", +1.1-1.5% ImageNet top-1 over V1
# from FCMAE pretraining + GRN, CVPR 2023) as a drop-in, then EfficientNet-V2-S
# as a second trunk for a rank ensemble. Both keep BACKBONE_DEFAULTS' convnet
# path, so each is a one-variable change.
BACKBONE = "convnext_tiny.fb_in22k_ft_in1k"

# Built once here before the trainer spawns its workers. Each rank constructs
# its own backbone, and two processes racing to download the same weights into
# the same cache is a failure that only appears on multi-GPU runs -- warming
# the cache first turns that race into two cache hits.
_warm = net.build_backbone(BACKBONE)
FEATURE_DIM = _warm.num_features
print(f"{BACKBONE}: {FEATURE_DIM} features, "
      f"{sum(p.numel() for p in _warm.parameters()) / 1e6:.1f}M params")
del _warm

# A pretrained ViT needs the statistics it was trained under and a slower rate
# on the trunk than on the from-scratch aggregation; a ConvNeXt needs neither
# and gets {} back, so its results stay comparable. See net.backbone_defaults.
BACKBONE_DEFAULTS = net.backbone_defaults(BACKBONE)
print("backbone defaults:", BACKBONE_DEFAULTS)

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

# THIS CELL DEPENDS ON EVERY CELL ABOVE IT.
#
# The five-fold sweep is run by changing FOLD here and re-running from this
# point, which works only while the kernel still holds the earlier cells. A
# restarted or reclaimed Colab runtime loses them, and the first symptom is
# `NameError: name 'make_loader' is not defined` twenty lines down -- which
# says nothing about what to do. Name the real problem instead.
_required = {
    "DEVICES": "1. Setup",
    "knee_lightning": "2. Repo modules",
    "index": "3. Assemble",
    "make_loader": "4. Dataset",
    "FEATURE_DIM": "5. Model",
}
_missing = {name: cell for name, cell in _required.items()
            if name not in globals()}
if _missing:
    raise SystemExit(
        "This cell needs the cells above it, and this kernel has not run "
        "them:\\n"
        + "\\n".join(f"    {name!r} comes from section {cell}"
                     for name, cell in _missing.items())
        + "\\n\\nRun the notebook from the top (Runtime -> Run all). Changing "
          "FOLD and re-running only this cell works within a LIVE kernel; a "
          "restarted runtime has to re-run everything.")

# The measurement run reached 0.729 at epoch 4 and was still rising
# (0.584 -> 0.679 -> 0.708 -> 0.729), so four was undertrained.
#
# CURRENT: 12, back from the 20 the P1 experiment used.
#
# Run length is measured and closed. Twelve epochs scored LB 0.889 and twenty
# scored 0.888 at the same geometry; across three 20-epoch runs val was flat
# from epoch 10 and gold moved +0.003 over epochs 12-20. Nothing is bought
# after 12.
#
# This is NOT "stop the 20-epoch run early" -- OneCycleLR is sized by EPOCHS,
# so a 12-epoch run anneals into its low-rate phase by epoch 12 rather than
# being cut off mid-schedule. It is a different, complete run.
#
# At 12 the absolute warmup below reproduces the historic schedule exactly:
# 3.6 / 12 = pct_start 0.30, OneCycleLR's default, which is what every
# reference number on record was trained under.
#
# Saves 38 min per fold, ~1.9 h across the three outstanding folds.
EPOCHS = 12
LR = 1e-4

# Warmup as an ABSOLUTE step count, not OneCycleLR's default 0.3 fraction.
# The fraction restretches with run length: at EPOCHS=12 the learning rate
# peaks near epoch 3.6, at EPOCHS=20 near epoch 6. Two runs of different
# lengths are then at different points of differently shaped schedules at the
# same epoch number, so "the best epoch moved" means nothing. Pinning the peak
# makes the 12- and 20-epoch runs comparable, which is what P1 needs.
#
# 0.3 x 12 epochs' worth of steps, so a twelve-epoch run is unchanged and a
# longer one warms up over the same absolute distance.
WARMUP_EPOCHS = 3.6
SCHEDULE = "onecycle"   # "cosine" removes the restretch entirely

# Seeded so every fold starts the aggregation from the same random
# initialisation. Reproducibility is the small reason. The real one is that
# weight averaging across folds -- a model soup -- is only meaningful for
# models in the same loss basin, and two runs whose heads were initialised
# differently are not. Averaging them produces a worse model, silently.
SEED = 0
# Safe to call here: seed_everything defers the CUDA seeding through
# torch.cuda._lazy_call rather than initialising a context, so the fork this
# notebook depends on is not poisoned. The STRATEGY guard further down checks
# torch.cuda.is_initialized() anyway and stops with an instruction if anything
# ever changes that.
pl.seed_everything(SEED, workers=True)

# POS_WEIGHT -- scale the positive term per finding by sqrt(neg/pos), capped.
#
# The metric is macro AUC, so a finding present in 2.5% of studies counts
# exactly as much as one present in 32.5%; without this the loss is dominated
# by whatever is prevalent. `KneeLitModule` has taken `pos_weight` since it was
# written and **no run has ever set it** -- every fold's hparams.yaml says
# None.
#
# Measured motivation (2026-08-31), out-of-fold over 4,349 studies: every
# medial/lateral pair has the rarer lateral side worse by the same margin --
# Meniscus 0.869 / 0.825, OA 0.883 / 0.832 -- and it is not laterality (the gap
# is identical among the 2,544 studies whose side came from the DICOM tag, so
# no mirroring error is possible there) and not label quality (on gold the
# lateral labels are the *better* ones, 0.816 against 0.765). What separates
# them is positives: on Medial Meniscus the model beats its own labels by
# 0.123, and on Lateral Meniscus it lands on them exactly -- enough data to
# imitate the extractor, not enough to generalise past it.
#
# The cap is load-bearing. Raw neg/pos reaches 38.9 for Fracture and 18.7 for
# MCL; at those weights the rare findings would swamp the loss the way the
# stratified ranking term did on 2026-08-31. The square root is the standard
# damping and 4.0 binds only the three rarest findings.
# Back to None. 4.0 was run on 2026-08-31 and measured null twice: paired
# -0.001 on the gold 58 and -0.000 across 871 out-of-fold studies, with the
# pre-registered diagnostic backwards -- the rare findings that took the
# largest weights fell 0.019 on average while the common ones rose 0.007, and
# Lateral Meniscus, the finding it was aimed at, read -0.114 on gold against
# -0.008 out of fold. Left at None so a retrained ensemble member is
# on-reference; an average mixing weighted and unweighted folds is a blend of
# two models and the averaging hides it.
POS_WEIGHT_CAP = None
if POS_WEIGHT_CAP is None:
    POS_WEIGHT = None
else:
    _positives = (labels[config.LABELS] >= 0.5).sum()
    _negatives = len(labels) - _positives
    POS_WEIGHT = [float(min((n / max(p, 1)) ** 0.5, POS_WEIGHT_CAP))
                  for p, n in zip(_positives, _negatives)]
    print("pos_weight (sqrt of neg/pos, capped at "
          f"{POS_WEIGHT_CAP}):")
    for _name, _w in zip(config.LABELS, POS_WEIGHT):
        _p = int((labels[_name] >= 0.5).sum())
        print(f"  {_name:18s} {_w:5.2f}   ({_p} positives, "
              f"{_p / len(labels):.1%})")

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

# --- Stage 1: the conditional synovitis objective -------------------------
# All three knobs are off by default, so a reference fold is unchanged.
#
# Why they exist: on the gold 58 the *effusion* head scores 0.722 as a
# synovitis classifier against the synovitis head's 0.701, and the gold
# effusion label alone scores 0.698. Pooled AUC on this finding is therefore
# obtainable with no synovitis-specific skill, which is why seven consecutive
# experiments read as null. The metric that can see the finding is
# effusion-stratified conditional AUC -- concordance over positive/negative
# pairs drawn from the same effusion stratum -- and it stands at 0.630.
#
# LAMBDA_STRATIFIED_RANK: RankNet restricted to pairs whose predicted effusion
# scores lie within STRATIFIED_CALIPER. Conditional AUC *is* within-stratum
# pair concordance, so this optimises the metric rather than a proxy for it.
# The known risk: stripping effusion removes the one reliable cue and leaves a
# harder task supervised by labels silent in 82.2% of reports against a true
# prevalence of 46.6%. Watch `gold_Synovitis_conditional`, which every epoch
# now logs beside `gold_Synovitis_pooled`.
#
# RESIDUAL_BETA: builds `syn_logit = beta * stopgrad(eff_logit) + g(features)`
# into the model, so g can only learn what effusion does not explain. Not
# derived from the model's own +0.93 self-coupling (that is the thing being
# replaced) and not from r_true = 0.403 (a correlation between two binary
# labels is not a logit coefficient). Fitted on labels, out of fold, by
# `conditional.calibrate_coupling`: 0.64 against the 58 gold labels, 1.14
# against the mined ones, which are 82% silent and biased toward severity.
# The coupling travels in the checkpoint, so notebook 10 needs no change.
#
# LAMBDA_COUPLING: a decorrelation penalty that aims at COUPLING_TARGET rather
# than at zero. The findings genuinely co-occur (gold odds ratio 6.09,
# p = 0.003, label correlation 0.403) while the model couples them at 0.65,
# so independence is the wrong target and the module refuses a penalty
# without an explicit one.
#
# Pre-registered success criterion, before any of this is run: conditional
# AUC 0.630 -> >= 0.75 on the gold 58, tested by paired bootstrap
# (tools/synovitis_gate.py). That is pooled ~0.755 on the same studies.
# 0.1 was tried on 2026-08-31 and **diverged**: BCE climbed 0.60 -> 5.99 over
# twelve epochs, macro fell 0.86 -> 0.71, and the synovitis head saturated so
# hard that every study tied and conditional AUC printed exactly 0.5000 from
# epoch 3 on. The mechanism is in `net.pairwise_rank_loss`: within-stratum
# synovitis pairs contradict each other, and inflating the logit scale was the
# cheapest way to satisfy the majority of them.
#
# Three changes, because one would not have been enough:
#   * the loss now clamps its margin (max_margin 8.0), so no pair can pay for
#     unbounded scale -- the mechanism fix, and the one that matters;
#   * the weight drops to 0.02;
#   * AUX_MODE is "cosine", which applies the auxiliary gradient only while it
#     points downhill for the main task. That gate exists in this project
#     precisely for this and would have switched the term off around epoch 1.
# Return this to 0.0 for any ensemble member.
LAMBDA_STRATIFIED_RANK = 0.0
STRATIFIED_FINDINGS = ("Synovitis",)
STRATIFY_BY = "Effusion"
STRATIFIED_CALIPER = 0.15
# Back to None. 0.64 was run on 2026-08-31 -- the gold synovitis label's
# coefficient on the predicted effusion logit, 0.636 +/- 0.245 -- and measured
# null on every instrument: paired conditional -0.024 on gold, +0.011 on 871
# out-of-fold studies, macro +0.004. Synovitis is descoped (worklog 2026-08-31
# (g)); this stays at None so the next ensemble member trains on-reference.
RESIDUAL_BETA = None
LAMBDA_COUPLING = 0.0
COUPLING_TARGET = 0.403     # the observed truth coupling, not independence

# AUX_MODE selects how the lambdas above are applied.
#
#   "fixed"  -- the weight is the constant above, every step. What C2 ran.
#   "cosine" -- Du et al., "Adapting Auxiliary Losses Using Gradient
#               Similarity" (arXiv:1812.02224), Algorithm 1: apply the
#               auxiliary gradient only while its cosine against the main
#               gradient is non-negative, i.e. only while it is a descent
#               direction for the main task. The paper's whole argument is
#               that a *constant* lambda is the bug -- an auxiliary task's
#               usefulness changes during training, so a fixed weight keeps
#               pushing after it has started to hurt. Set LAMBDA_REPORT = 1.0
#               with this mode to reproduce the paper exactly; the lambda
#               then scales the auxiliary only on the steps it is switched on.
#
# Measured 2026-08-26 and CLOSED. Three configurations on fold 1 at 16 x 5,
# against the aux-free reference's gold 0.861:
#   report 0.2 + rank 0.2   val 0.868  gold 0.854
#   report 0.5, fixed       val 0.881  gold 0.867  -> LB 0.880 (aux-free 0.889)
#   report 1.0, cosine gate val 0.874  gold 0.856
# None beat the reference on gold, and the only one submitted lost 0.009 on
# the leaderboard. The gating paper's promise held -- no collapse -- but there
# was nothing to gain. Both lambdas ship at 0.0; the code stays for a future
# capacity phase.
# Back to "fixed": no auxiliary loss is active in the residual run, so the
# gate would gate nothing, and a second changed constant would stop this being
# a one-variable experiment. Set it to "cosine" together with any future
# LAMBDA_STRATIFIED_RANK -- that pairing is what the divergence bought.
AUX_MODE = "fixed"
AUX_GATE = "binary"   # "weighted" uses max(0, cos) instead of an on/off gate
REPORT_DIM = (len([c for c in reports.columns if c != "StudyInstanceUID"])
              if reports is not None else None)

# Which fold this session trains.
#
# THE FOLD AND THE AUX SETTINGS MOVE TOGETHER. This notebook serves two runs
# and they are mutually exclusive:
#
#   Ensemble fold   LAMBDA_REPORT = 0.0, AUX_MODE = "fixed".
#                   Fold 3 done 2026-08-26 (val 0.869, verified aux-free);
#                   fold 1 exists as lit_fold1_old.ckpt (16 x 5 reference,
#                   LB 0.889). Outstanding: FOLD = 0, then 2, then 4.
#   Backbone probe  FOLD = 1, BACKBONE = a different trunk, lambdas 0.0.
#                   Fold 1 because it is the ONLY split with a measured
#                   baseline (val 0.872 / gold 0.861 / LB 0.889); the same
#                   run on any other fold has nothing to be compared against.
#                   Fold 3's LB 0.877 against fold 1's 0.889 on an identical
#                   configuration is what that costs when ignored.
#
# Changing the fold and the configuration together has already cost this
# project one ambiguous run, and an ensemble fold trained with an auxiliary
# loss or a different backbone is worse than useless -- it silently makes one
# member of the average a different model. The banner below states which run
# this is.
#
# CURRENT RUN: fold 1, an EXPERIMENT and not an ensemble member. The
# five-fold ensemble is complete (folds 0-4 all trained aux-free) and scored
# **LB 0.906** on 2026-08-31 -- the project's best result, and the reason
# nothing here is waiting on a fold.
#
# **Fold 1, and the fold is not free to choose.** It is the only split with a
# measured baseline (val 0.8726 / gold 0.8747 / LB 0.889 at 16 x 5), and
# per-finding scores move more across splits than any of these treatments do.
# The same run on another fold has nothing to be compared against; fold 3's
# LB 0.877 against fold 1's 0.889 on an identical configuration is what that
# costs when ignored.
#
# NOTHING ELSE CHANGES IN AN EXPERIMENT. Eight single-variable changes have
# now measured null against a gold set whose macro SE is 0.015-0.025 and whose
# per-finding SE is 0.045-0.08; the instrument cannot read them, so do not
# slip a second variable in beside the one being tested.
FOLD = 1
CHECKPOINT = f"/kaggle/working/convnext_fold{FOLD}.pt"

# Where the EMA and SWA weights land. Deliberately NOT convnext_fold*_ema.pt:
# notebook 10 averages every `convnext_fold*.pt` it can see, so a name under
# that glob would silently join the fold ensemble as an extra member of the
# same run. `avg_fold{N}_ema.pt` and `avg_fold{N}_swa.pt` are picked up only
# when TUNED_PATTERNS asks for them by name.
AVERAGE_PREFIX = f"/kaggle/working/avg_fold{FOLD}"

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

WARMUP_STEPS = int(round(TOTAL_STEPS / EPOCHS * WARMUP_EPOCHS))
print(f"warmup: {WARMUP_STEPS} steps ({WARMUP_EPOCHS} epochs' worth), "
      f"schedule {SCHEDULE}")

module = knee_lightning.KneeLitModule(
    backbone=net.build_backbone(BACKBONE),
    feature_dim=FEATURE_DIM, lr=LR, total_steps=TOTAL_STEPS,
    report_dim=REPORT_DIM, lambda_report=LAMBDA_REPORT,
    lambda_rank=LAMBDA_RANK, aux_mode=AUX_MODE, aux_gate=AUX_GATE,
    schedule=SCHEDULE, warmup_steps=WARMUP_STEPS,
    lambda_stratified_rank=LAMBDA_STRATIFIED_RANK,
    stratified_findings=STRATIFIED_FINDINGS, stratify_by=STRATIFY_BY,
    stratified_caliper=STRATIFIED_CALIPER,
    lambda_coupling=LAMBDA_COUPLING,
    coupling_target=COUPLING_TARGET if LAMBDA_COUPLING > 0 else None,
    residual_beta=RESIDUAL_BETA, pos_weight=POS_WEIGHT,
    **BACKBONE_DEFAULTS)
print(f"trunk lr: {LR * BACKBONE_DEFAULTS['backbone_lr_scale']:.1e}  "
      f"head lr: {LR:.1e}  "
      f"input norm: {'on' if BACKBONE_DEFAULTS['input_mean'] else 'off'}")
print(f"aux losses: report {LAMBDA_REPORT} (dim {REPORT_DIM})  "
      f"rank {LAMBDA_RANK}  mode {AUX_MODE}"
      + (f"/{AUX_GATE}" if AUX_MODE == "cosine" else ""))
print(f"synovitis: stratified rank {LAMBDA_STRATIFIED_RANK} "
      f"(caliper {STRATIFIED_CALIPER})  residual beta {RESIDUAL_BETA}  "
      f"coupling {LAMBDA_COUPLING}"
      + (f" -> {COUPLING_TARGET}" if LAMBDA_COUPLING > 0 else ""))

# Which of the notebook's two mutually exclusive runs this is. An ensemble
# fold trained with an auxiliary loss makes one member of the average a
# different model, and the averaging hides it -- so say it before the hours
# are spent, not after the leaderboard disagrees.
# --- Ensemble bookkeeping -------------------------------------------------
# Which folds are already trained at the reference config. Update this list
# when a fold finishes and its .pt is safely off the VM -- it is the only
# record a cold session has of what is left to run.
#
# Verified by opening each checkpoint's stored hyper_parameters rather than
# trusting the filenames: folds 0, 1, 2 all report total_steps 5220 (12 epochs
# at 435 steps/epoch), both auxiliary lambdas 0.0, gated_pool False.
#
# Fold 3 is the older 2026-08-26 checkpoint: 5232 steps, also 12 epochs and
# also aux-free, but predating both `gated_pool` and `seed_everything`. Kept.
# The 12-step difference is nothing, and the missing seed matters only for
# weight-space souping -- notebook 10 averages probabilities, so it does not
# apply. Retraining it would buy homogeneity nobody can measure.
ENSEMBLE_DONE = {0: "2026-08-30, 12 ep, val 0.8814",
                 1: "2026-08-30, 12 ep, val 0.8726",
                 2: "2026-08-30, 12 ep, val 0.8675",
                 3: "2026-08-26, 12 ep, val 0.8686 (LB 0.877 solo)",
                 4: "2026-08-30, 12 ep, val 0.8697"}
ENSEMBLE_TODO = [f for f in range(5) if f not in ENSEMBLE_DONE]

# The reference config, spelled out. Every member of an average must be the
# same model; one fold trained with a different knob makes the ensemble a
# blend of two things and the averaging hides it. Checked, not trusted.
_expected = {"SLICES_PER_SERIES": (SLICES_PER_SERIES, 16),
             "MAX_SERIES": (MAX_SERIES, 5),
             "EPOCHS": (EPOCHS, 12),
             "LAMBDA_REPORT": (LAMBDA_REPORT, 0.0),
             "LAMBDA_RANK": (LAMBDA_RANK, 0.0),
             "LAMBDA_STRATIFIED_RANK": (LAMBDA_STRATIFIED_RANK, 0.0),
             "RESIDUAL_BETA": (RESIDUAL_BETA, None),
             "LAMBDA_COUPLING": (LAMBDA_COUPLING, 0.0),
             "POS_WEIGHT_CAP": (POS_WEIGHT_CAP, None),
             "SILENCE_REWEIGHT": (SILENCE_REWEIGHT, False),
             "BACKBONE": (BACKBONE, "convnext_tiny.fb_in22k_ft_in1k")}
_wrong = {k: v for k, (v, want) in _expected.items() if v != want}

print("=" * 68)
if _wrong:
    print(f"  OFF-REFERENCE FOLD {FOLD}. These differ from the ensemble config:")
    for k, v in _wrong.items():
        print(f"    {k} = {v!r}, reference is {_expected[k][1]!r}")
    print("  An ensemble member trained off-reference makes the average a")
    print("  blend of two models, and the averaging hides it. Fix these, or")
    print("  accept that this run is an experiment and not a member.")
elif FOLD in ENSEMBLE_DONE:
    print(f"  FOLD {FOLD} IS ALREADY DONE -- {ENSEMBLE_DONE[FOLD]}")
    print(f"  Retraining it is fine, but the folds still missing are "
          f"{ENSEMBLE_TODO}.")
else:
    print(f"  ENSEMBLE FOLD {FOLD}: reference config, no auxiliary loss")
    print(f"  done: {sorted(ENSEMBLE_DONE)}   outstanding after this: "
          f"{[f for f in ENSEMBLE_TODO if f != FOLD]}")
    print(f"  export convnext_fold{FOLD}.pt and COPY IT OFF THE VM before the")
    print("  session ends -- /kaggle/working does not survive.")
print("=" * 68)

checkpointer = pl.callbacks.ModelCheckpoint(
    dirpath="/kaggle/working", filename=f"lit_fold{FOLD}",
    monitor="val_macro", mode="max", save_top_k=1)
report = knee_lightning.EpochReport()

# Selection-free weights, accumulated alongside the monitored checkpoint so
# one run produces three candidates instead of one:
#
#   lit_fold{N}.ckpt      the best mined-val epoch      <- selected, suspect
#   avg_fold{N}_ema.pt    exponential moving average    <- selection-free
#   avg_fold{N}_swa.pt    equal average, last 25%       <- selection-free
#
# The argument for the two averages is that the selector is broken, not that
# the optimiser is. Mined validation shares its ~18% systematic disagreement
# with gold with the training labels, so late in training it rewards
# memorising exactly the labels that are wrong -- Liu et al., ELR, NeurIPS
# 2020. Averaging asks nothing of any metric. SWA (Izmailov et al., UAI 2018)
# lands in a flatter region than any single point on the trajectory; EMA
# (Tarvainen & Valpola, NeurIPS 2017) is the cheap continuous version.
#
# Scored against each other in the cell below. Nothing here selects.
averager = knee_lightning.WeightAverage(
    prefix=AVERAGE_PREFIX, decay=0.999, swa_fraction=0.25,
    total_steps=TOTAL_STEPS)

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
    callbacks=[checkpointer, report, averager,
               knee_lightning.BatchProgress()],
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
def recover_checkpoint(fold):
    \"\"\"The checkpoint this run wrote, when `best_model_path` came back empty.

    NOT `sorted(glob(...))[0]`, which is what this was. That picks the first
    name ALPHABETICALLY, and "-" (45) sorts before "." (46), which sorts
    before "_" (95): `lit_fold1-v1.ckpt` beats `lit_fold1.ckpt`, which beats
    `lit_fold1_old.ckpt`. Lightning writes those `-vN` suffixes on every name
    collision, so a session that resumed after a crash silently recovered a
    checkpoint from the earlier attempt.

    The exact name wins when it exists, and otherwise the newest file does --
    which is this run's, since this cell runs seconds after fit returned.
    \"\"\"
    exact = f"/kaggle/working/lit_fold{fold}.ckpt"
    if os.path.exists(exact):
        return exact
    found = glob.glob(f"/kaggle/working/lit_fold{fold}*.ckpt")
    return max(found, key=os.path.getmtime) if found else None


checkpoint_path = checkpointer.best_model_path or ""
if not checkpoint_path or not os.path.exists(checkpoint_path):
    checkpoint_path = recover_checkpoint(FOLD)
    if checkpoint_path is None:
        raise SystemExit(
            "training finished but no checkpoint was written -- check the "
            "epoch log for whether val_macro was ever computed")
    print(f"best_model_path was empty; recovered {checkpoint_path}")
print("\\nbest checkpoint:", checkpoint_path)

# Convert Lightning's checkpoint into the bare KneeNet state dict notebook 10
# loads. Read from the file, not from `module`: the process that called fit is
# not the process that trained.
n_tensors = knee_lightning.export_from_checkpoint(checkpoint_path, CHECKPOINT)
print(f"wrote {CHECKPOINT} ({n_tensors} tensors)")

# The averages were written by the callback inside fit, in on_train_end. They
# are read back off disk here for exactly the reason best_model_score comes
# back None: under ddp_notebook the object that accumulated them lives in a
# forked child, and `averager.average` in THIS process is still None. Asking
# it directly would report an empty average with no error at all.
AVERAGES = {kind: f"{AVERAGE_PREFIX}_{kind}.pt"
            for kind in ("ema", "swa")
            if os.path.exists(f"{AVERAGE_PREFIX}_{kind}.pt")}
for kind, path in AVERAGES.items():
    print(f"wrote {path} ({kind.upper()} weights, selection-free)")
if not AVERAGES:
    print("NOTE: no averaged weights on disk -- the WeightAverage callback "
          "did not run to on_train_end. The monitored checkpoint is fine; "
          "only the P2 comparison is unavailable.")

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

    # P1's verdict, read off the curves this run just produced.
    #
    # The claim under test: mined validation is a BIASED selector, not merely
    # a noisy one. It carries the same ~18% systematic disagreement with
    # image-derived truth that the training labels do, so once the network
    # starts memorising that disagreement, mined-val agreement keeps rising
    # while true performance does not. The signature is a val peak that
    # arrives while gold is still flat or still climbing, and a val curve that
    # then falls away from a gold curve that does not.
    #
    # Not a decision procedure: gold on 58 studies has a macro SE around 0.02,
    # so no single epoch's gold reading means anything. The shape across
    # eleven epochs does.
    _val = per_epoch_rows["val_macro"]
    _gold = per_epoch_rows["gold_macro"]
    _tail = _gold.loc[best_epoch:]
    print("\\nP1 -- val/gold divergence")
    print(f"  val   peak {_val.max():.3f} at epoch {_val.idxmax() + 1}, "
          f"final {_val.iloc[-1]:.3f}  ({_val.iloc[-1] - _val.max():+.3f})")
    print(f"  gold  peak {_gold.max():.3f} at epoch {_gold.idxmax() + 1}, "
          f"final {_gold.iloc[-1]:.3f}  ({_gold.iloc[-1] - _gold.max():+.3f})")
    print(f"  gold from the selected epoch on: range "
          f"{_tail.max() - _tail.min():.3f} over {len(_tail)} epochs")
    if _val.iloc[-1] < _val.max() - 0.005 and _tail.max() - _tail.min() < 0.02:
        print("  -> DIVERGENCE REPRODUCES. Mined val fell while gold held "
              "flat: the selector is biased, and every fold selected on it "
              "is paying for it. Ship the averaged weights.")
    else:
        print("  -> no divergence in this run. Selecting on mined val is "
              "defensible here; re-read before adopting the averages "
              "fleet-wide.")
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
    scoring_path = recover_checkpoint(FOLD)
    if scoring_path is None:
        raise SystemExit(f"neither {CHECKPOINT} nor a lit_fold{FOLD} "
                         "checkpoint is present")
    print(f"{CHECKPOINT} missing; scoring from {scoring_path}")

# BACKBONE_DEFAULTS, exactly as the training module applied them. These now
# also travel IN the checkpoint -- the buffers were registered
# persistent=False until 2026-08-28, so a strict load restored every weight
# and silently left normalisation off, which is how this cell once printed
# 0.823 for weights that logged 0.864 during training. Passed here anyway:
# the model should be right before it is loaded, not because it was loaded.
def load_for_scoring(path):
    model = net.KneeNet(
        net.build_backbone(BACKBONE, pretrained=False), FEATURE_DIM,
        input_mean=BACKBONE_DEFAULTS["input_mean"],
        input_std=BACKBONE_DEFAULTS["input_std"]).to(device)
    model.load_state_dict(net.load_knee_state(path, map_location=device))
    return model.eval()


def gold_predictions(model):
    ids, rows = [], []
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for batch in gold_loader:
            logits = model(batch["volumes"].to(device),
                           batch["meta"].to(device),
                           batch["mask"].to(device))
            rows.append(torch.sigmoid(logits.float()).cpu().numpy())
            ids += batch[config.ID_COL]
    return ids, np.concatenate(rows)


print("scoring with input normalisation:",
      "on" if BACKBONE_DEFAULTS["input_mean"] else "off")
scoring_model = load_for_scoring(scoring_path)
gold_ids, gold_probs = gold_predictions(scoring_model)
truth = gold.set_index("StudyInstanceUID").loc[gold_ids]

# The frozen DINOv2 path, retired 2026-08-23 at LB 0.787. Kept only as
# historical context -- beating it is not news and has not been since the
# first capacity rung.
FROZEN = dict(zip(config.LABELS,
                  [0.728, 0.671, 0.698, 0.680, 0.895, 0.899,
                   0.772, 0.897, 0.746, 0.862, 0.711, 0.683]))

# The comparisons that actually decide anything. Both are fold 1 at 16 x 5,
# so a fold-1 run is directly comparable and any other fold is not.
#
# REFERENCE_GOLD was 0.861 until 2026-08-31 and that number was stale by
# 0.014: it came from the 2026-08-26 era, while the fold-1 checkpoint every
# current comparison is actually made against -- the 2026-08-30 run in
# models_train/fold1 -- logged gold_macro 0.8747 at its selected epoch, and
# its saved predictions score macro 0.876. Using 0.861 flattered the
# pos_weight run by exactly the gap: it read +0.014 against the constant and
# -0.001 against the predictions, paired. Read the paired number.
REFERENCE_GOLD = 0.875     # fold 1, 16 x 5, no aux losses, 2026-08-30 run
BEST_GOLD = 0.875          # the same; the 24 x 5 run's 0.865 predates it

print(f"{'finding':18s} {'this run':>9s}  {'frozen (retired)':>16s}")
print("-" * 47)
scores = []
for i, name in enumerate(config.LABELS):
    value = auc(truth[name].to_numpy().astype(int), gold_probs[:, i])
    scores.append(value)
    print(f"{name:18s} {value:9.3f}  {FROZEN[name]:16.3f}")
print("-" * 47)
macro = float(np.nanmean(scores))
print(f"{'macro':18s} {macro:9.3f}  {np.mean(list(FROZEN.values())):16.3f}")

# Beating the retired baseline is not the question; beating the model that
# currently holds the leaderboard slot is. Gold's own noise floor is ~0.02,
# so print the deltas next to it rather than inviting a read off the table.
print(f"\\nthe comparisons that matter (fold 1, 16 x 5):")
print(f"  vs reference  gold {REFERENCE_GOLD:.3f} (LB 0.889): "
      f"{macro - REFERENCE_GOLD:+.3f}")
print(f"  vs best       gold {BEST_GOLD:.3f} (LB 0.891): "
      f"{macro - BEST_GOLD:+.3f}")
if abs(macro - REFERENCE_GOLD) < 0.02:
    print("  -> inside the gold 58's ~0.02 noise floor: this is a null "
          "against the reference, whatever the frozen column says.")
if FOLD != 1:
    print(f"  -> NOTE: this is fold {FOLD}; both references are fold 1, so "
          "these deltas are not comparable. Judge fold members by the "
          "ensemble's leaderboard score, not by gold here.")

# P2 -- the three candidates this run produced, side by side.
#
# The selected checkpoint against the two averages that selected nothing. Read
# the DIRECTION, not the numbers: 58 studies put a standard error of roughly
# 0.02 on the macro, so a 0.005 gap between these is not a result. What would
# be a result is the averages sitting consistently at or above the selected
# checkpoint across several folds -- which is what the literature predicts
# when the selector is the thing that is broken.
if AVERAGES:
    print("\\nP2 -- selection-free weights (gold macro)")
    print(f"  {'best mined-val epoch':24s} {macro:.3f}")
    for kind, path in AVERAGES.items():
        _ids, _probs = gold_predictions(load_for_scoring(path))
        _truth = gold.set_index("StudyInstanceUID").loc[_ids]
        _macro = float(np.nanmean([
            auc(_truth[name].to_numpy().astype(int), _probs[:, i])
            for i, name in enumerate(config.LABELS)]))
        print(f"  {kind.upper() + ' (selection-free)':24s} {_macro:.3f}  "
              f"({_macro - macro:+.3f} vs selected)")
    print("  gold's macro SE on 58 studies is ~0.02; treat anything under "
          "that as a tie and decide on the leaderboard.")

# The predictions themselves, keyed by UID. Every synovitis comparison in the
# plan is paired on these 58 studies, and a paired test needs both models'
# predictions rather than two printed macros -- which is what made the frozen
# path's apparent +0.045 unreadable for a week.
GOLD_PREDICTIONS = f"/kaggle/working/gold_predictions_fold{FOLD}.parquet"
pd.DataFrame(gold_probs, columns=config.LABELS).assign(
    **{config.ID_COL: gold_ids})[config.SUBMISSION_COLUMNS].to_parquet(
        GOLD_PREDICTIONS, index=False)
print("\\nwrote", GOLD_PREDICTIONS)
""")

    nb.md("""
## 8b. Synovitis, on the metric that can measure it

Macro cannot read this run and neither can pooled AUC on the finding itself:
the gold effusion label alone scores 0.698 as a synovitis classifier against
the fine-tuned head's 0.701. If `LAMBDA_STRATIFIED_RANK` or `RESIDUAL_BETA` is
set, **this is the cell the run is judged on** — everything above is context.

Pre-registered before the run: conditional AUC 0.630 → **≥ 0.75**, paired
bootstrap on the same 58 studies, which is pooled ≈ 0.755. Two runs of an
identical configuration differ by about 0.011 here, so the target is roughly
eleven times the run-to-run jitter; the sampling interval is ±0.17, which is
why a pass will still not be *significant* on this set.

Also watch `stratified_rank_loss` in the epoch table. The previous ranking
term collapsed to near-zero within a few epochs — it memorised its own queue
rather than learning an ordering — and that is the failure mode to catch here
before reading the AUC at all.
""")

    nb.code("""
FINDING, STRATIFY_BY = "Synovitis", "Effusion"
BASELINE_CONDITIONAL = 0.630   # fold 1, 16 x 5, aux-free -- the thing to beat
TARGET_CONDITIONAL = 0.75      # pre-registered, not chosen after the fact

y = truth[FINDING].to_numpy().astype(int)
scores_syn = gold_probs[:, config.LABELS.index(FINDING)]
strata = truth[STRATIFY_BY].to_numpy().astype(int)

stats = conditional.bootstrap_conditional(y, scores_syn, strata, draws=4000)
print(f"conditional  {stats['auc']:.3f}  "
      f"95% CI [{stats['lo']:.3f}, {stats['hi']:.3f}]  "
      f"P(<= 0.5) = {stats['p_at_or_below_chance']:.3f}")
print(f"pooled       {conditional.auc(y, scores_syn):.3f}")
print(f"the {STRATIFY_BY} head as a {FINDING} classifier: "
      f"{conditional.auc(y, gold_probs[:, config.LABELS.index(STRATIFY_BY)]):.3f}"
      "   <- if this is higher, the head is still gated")
print()
print(f"vs the {BASELINE_CONDITIONAL:.3f} baseline: "
      f"{stats['auc'] - BASELINE_CONDITIONAL:+.3f}   "
      f"target was {TARGET_CONDITIONAL:.2f} "
      f"({'MET' if stats['auc'] >= TARGET_CONDITIONAL else 'not met'})")

# Paired against the baseline run's predictions, if they are attached. This is
# the comparison the plan's standing rules require: the same studies, the same
# resamples, so between-study variance cancels instead of hiding the effect.
baseline_paths = [p for p in find_inputs("gold_predictions*.parquet")
                  if "frozen" not in os.path.basename(p)
                  and os.path.abspath(p) != os.path.abspath(GOLD_PREDICTIONS)]
if not baseline_paths:
    print("\\nno baseline gold_predictions*.parquet attached -- attach the "
          "aux-free fold-1 run's to get the paired comparison")
else:
    base = pd.read_parquet(baseline_paths[0]).set_index(config.ID_COL)
    keep = [i for i, s in enumerate(gold_ids) if s in base.index]
    shared = [gold_ids[i] for i in keep]
    print(f"\\npaired against {os.path.basename(baseline_paths[0])} "
          f"({len(shared)} shared studies)")
    for label, kwargs in (("conditional", {"strata": strata[keep]}),
                          ("pooled", {"strata": None})):
        delta = conditional.paired_bootstrap(
            y[keep], scores_syn[keep],
            base.loc[shared, FINDING].to_numpy(), draws=4000, **kwargs)
        print(f"  {label:12s} {delta['delta']:+.3f}  "
              f"95% CI [{delta['lo']:+.3f}, {delta['hi']:+.3f}]  "
              f"P(not better) = {delta['p_a_not_better']:.3f}")
""")

    nb.md("""
## 9. Out-of-fold predictions

Writes this fold's validation predictions to `oof_fold{FOLD}.parquet`. The
studies here were never trained on by this checkpoint, so five folds tile the
whole 4,349-study corpus with predictions from a reader that saw the images and
never saw the study it is scoring.

This is the only label source in the project that can rank studies inside the
blocks the reports leave silent -- 27% of gold positives are never written down
at all, and no extractor can reach those. Whether that is worth anything is
Phase E's question; this cell just stops the answer from costing another pass,
since the forward pass has already happened.

**Attach the resulting file to the `knee-labels` dataset alongside the
checkpoint.** A fold whose OOF dump is missing has to be re-run to get it.
""")

    nb.code("""
# The gold studies are excluded from training everywhere, so they are absent
# from every fold's validation split and no OOF row can be one of them. That
# keeps this file usable for label work without a leak -- but it also means
# the gold 58 never appear here, so Phase E's gate still reads gold from the
# labels it builds, not from this dump.
oof_ids, oof_rows = [], []
with torch.no_grad(), torch.amp.autocast("cuda"):
    for batch in val_loader:
        logits = scoring_model(batch["volumes"].to(device),
                               batch["meta"].to(device),
                               batch["mask"].to(device))
        oof_rows.append(torch.sigmoid(logits.float()).cpu().numpy())
        oof_ids += batch[config.ID_COL]

oof = pd.DataFrame(np.concatenate(oof_rows), columns=config.LABELS)
oof.insert(0, config.ID_COL, oof_ids)
oof["fold"] = FOLD

OOF_PATH = f"/kaggle/working/oof_fold{FOLD}.parquet"
oof.to_parquet(OOF_PATH, index=False)
print(f"wrote {OOF_PATH}: {len(oof)} studies x {len(config.LABELS)} findings")
print(f"mean predicted rate: {oof[config.LABELS].to_numpy().mean():.3f}")

# A dump that is all one value is a broken export, not a cautious model, and
# it would quietly poison any blend built on it.
_spread = oof[config.LABELS].to_numpy().std(axis=0)
if float(_spread.min()) < 0.01:
    _flat = [n for n, s in zip(config.LABELS, _spread) if s < 0.01]
    print(f"  WARNING: near-constant predictions for {_flat} -- "
          "check before using this dump for label work")
""")

    if colab:
        nb.md("""
## 10. Save the artefacts off the VM

`/kaggle/working` disappears with the runtime, and a reclaimed session takes
the checkpoint with it. These three files are small.
""")

        nb.code("""
from google.colab import drive

drive.mount("/content/drive")

DRIVE_OUT = "/content/drive/MyDrive/knee_runs"
os.makedirs(DRIVE_OUT, exist_ok=True)

saved = []
for pattern in (f"convnext_fold{FOLD}.pt", f"lit_fold{FOLD}*.ckpt",
                f"avg_fold{FOLD}_*.pt", f"oof_fold{FOLD}.parquet"):
    for path in glob.glob(f"/kaggle/working/{pattern}"):
        shutil.copy(path, DRIVE_OUT)
        saved.append(os.path.basename(path))

metrics = os.path.join(csv_logger.log_dir, "metrics.csv")
if os.path.exists(metrics):
    shutil.copy(metrics, os.path.join(DRIVE_OUT, f"metrics_fold{FOLD}.csv"))
    saved.append(f"metrics_fold{FOLD}.csv")

print("saved to", DRIVE_OUT)
for name in saved:
    print("  ", name)
if not saved:
    print("  NOTHING SAVED -- the training cell did not complete")
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

    nb.write(NOTEBOOKS / ("08_finetune_colab.ipynb" if colab
                          else "08_finetune.ipynb"))


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

# --- Where notebook 10's Colab copy looks for its inputs --------------------
# The Kaggle notebook scans /kaggle/working and /kaggle/input/*. On Colab the
# Drive copy is somewhere else entirely, so the search is rewritten rather
# than extended: the roots overlap (from_drive is inside /kaggle_local/input),
# and /kaggle/input entries are symlinks made by the staging cell, so the same
# file is reachable by two or three paths. Loading it twice is not a crash --
# it silently doubles that checkpoint's weight in the fold average.

COLAB_FIND_INPUTS_OLD = '''def find_inputs(pattern):
    found = sorted(glob.glob(f"/kaggle/working/{pattern}"))
    for root in sorted(glob.glob("/kaggle/input/*")):
        if os.path.basename(root) == "competitions":
            continue
        found += sorted(glob.glob(f"{root}/**/{pattern}", recursive=True))
    return found'''

COLAB_FIND_INPUTS_NEW = '''# Search roots in priority order. Edit this one tuple to move the cache, the
# checkpoints, or anything else -- find_inputs is what every later cell calls,
# so nothing downstream needs to know where a file came from.
INPUT_ROOTS = (
    "data",                              # competition CSVs, from the staging cell
    "/content/downloads",                # a checkpoint dropped in by hand
    "/content/drive/MyDrive/Downloads",  # ... or downloaded into Drive
    "/kaggle/working",                   # a checkpoint trained in this session
    "/kaggle_local/input/from_drive",    # the Drive copy on this machine
    "/kaggle_local/input",
    "/kaggle/input",                     # where staging symlinks its mounts
)


def find_inputs(pattern):
    """Every distinct file matching `pattern`, nearest root first.

    Deduplicated by realpath. The roots deliberately overlap and /kaggle/input
    holds symlinks into the kagglehub cache, so a single checkpoint is
    reachable several ways; returning it twice would load it twice and weight
    it twice in the averaged prediction, with nothing in the output saying so.
    """
    found = []
    for root in INPUT_ROOTS:
        # Skip absent roots rather than walking them. Over the Drive FUSE
        # mount a recursive glob is slow enough to look like a hang, and most
        # of these roots are absent in any given session.
        if not os.path.isdir(root):
            continue
        for path in sorted(glob.glob(f"{root}/**/{pattern}", recursive=True)):
            # Competition mounts are addressed by find_competition_file.
            if f"{os.sep}competitions{os.sep}" in path:
                continue
            found.append(path)

    seen, unique = set(), []
    for path in found:
        key = os.path.realpath(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique'''


# --- Notebook 10's four differences from notebook 08's staging --------------
# Everything else about how the inputs arrive is the shared cell above.

COLAB_STAGE_HEADER_10 = """\
# --- Colab staging: mount the Kaggle data, then run the cells below unchanged.
# The same method 08_finetune_colab uses -- one implementation, emitted into
# both notebooks -- so what this run reads is what the training run read.
# dataset_download MOUNTS on Colab (no copy) and lands under
# /kaggle/input/<slug>, which is exactly where find_inputs looks.
#
# This notebook scores instead of training, so it stages the cache the
# training run read plus the checkpoints to score. It never stages test
# DICOM: there is none off Kaggle, which is why the copy ends at the gold 58
# rather than at a submission."""

COLAB_STAGE_WORKING_10 = """\
# Nothing is written to /kaggle/working here, but a checkpoint trained in an
# earlier Colab session is read from it, and both directories must exist for
# the one find_inputs implementation to work unchanged on both platforms."""

COLAB_STAGE_SHORT_SHARDS_10 = """\
    print("WARNING: fewer than three shards. A gold study whose series live "
          "in a missing shard is dropped silently -- the scoring cell must "
          "report 58 of 58.")"""

COLAB_STAGE_TAIL_10 = '''

# --- What this notebook needs and notebook 08 does not ----------------------
# Same three-step method as everything above: prefer the mount, fall back to
# the notebook output, and say so when neither exists.

# 1. The tuned checkpoints. One attempt, not fetch_shard's ten: a handle that
#    does not exist is the normal case here, because a checkpoint trained in
#    this session is already in /kaggle/working and needs no staging at all.
CHECKPOINT_DATASETS = ("pssv27/knee-convnext",)

local_checkpoints = (glob.glob("/kaggle/working/*.pt")
                     + glob.glob("/kaggle/working/*.ckpt"))
if local_checkpoints:
    print(f"\\ncheckpoints already local: {len(local_checkpoints)} in "
          "/kaggle/working")
else:
    for handle in CHECKPOINT_DATASETS:
        try:
            stage(kagglehub.dataset_download(handle), handle.split("/")[1])
            print(f"  mounted  {handle}")
        except Exception as exc:  # noqa: BLE001 - not staged is a valid state
            print(f"  {handle}: {type(exc).__name__} -- skipped. A checkpoint "
                  "in /kaggle/working or Drive is fine; the listing cell "
                  "below reports whatever is actually found.")

# 2. side_model.pt. The load cell reads it unconditionally, and it arrives
#    only with the side-classifier output -- which the block above fetches
#    only when side_predictions.parquet was missing. Supplied from knee-labels
#    or from Drive, that parquet is present and the model file is not, and the
#    failure lands several cells later on a name that says nothing about
#    staging.
if not glob.glob("/kaggle/input/**/side_model.pt", recursive=True):
    stage(kagglehub.notebook_output_download("pssv27/knee-side-classifier"),
          "knee-side-classifier")
if not glob.glob("/kaggle/input/**/side_model.pt", recursive=True):
    raise SystemExit(
        "side_model.pt not staged. The load cell below reads it whatever the "
        "blend is set to, so this fails here rather than there.")
print("side_model.pt: staged")
'''

COLAB_CHECKPOINT_CELL = '''
# --- Which weights are we about to score? -----------------------------------
# The Kaggle notebook globs for checkpoints and scores whatever turns up. That
# is fine when one dataset is attached and actively misleading otherwise:
# `sorted(glob(...))[0]` picks the first name ALPHABETICALLY, and "-" (45)
# sorts before "." (46), which sorts before "_" (95). So lit_fold1-v1.ckpt
# beats lit_fold1.ckpt, which beats lit_fold1_old.ckpt -- and Lightning writes
# those -vN suffixes on every name collision.
#
# Here every candidate is listed with its size, mtime and content hash before
# anything is loaded, and CHECKPOINTS pins the choice explicitly. Set it to a
# list of paths to score exactly those; leave it None to score everything
# found, which is the five-fold ensemble case.
import hashlib

CHECKPOINTS = None

# Exactly the roots find_inputs uses -- not a superset. A cell that lists a
# checkpoint the loader cannot see is worse than one that lists nothing, and
# bare /content or MyDrive would recursively walk the whole Drive mount.
SEARCH = INPUT_ROOTS

# The three families notebook 08 writes per fold. Listing all of them here is
# deliberate -- the point of this cell is to show what is on the machine --
# but the load cell below refuses to score more than one family at once, so
# pin CHECKPOINTS when several are present. `dinov2_fold*.pt` is gone: that
# axis closed at gold 0.820 against a 0.861 gate.
PATTERNS = ("convnext_fold*.pt", "lit_fold*.ckpt",
            "avg_fold*_ema.pt", "avg_fold*_swa.pt")


def find_checkpoints():
    hits = []
    for root in SEARCH:
        for pattern in PATTERNS:
            hits += glob.glob(f"{root}/**/{pattern}", recursive=True)
    return sorted({os.path.realpath(h) for h in hits})


def digest(path, limit=64 << 20):
    # First 64 MB only. Enough to separate two checkpoints, fast enough to run
    # on every candidate without thinking about it.
    sha = hashlib.sha256()
    with open(path, "rb") as handle:
        sha.update(handle.read(limit))
    return sha.hexdigest()[:12]


candidates = CHECKPOINTS if CHECKPOINTS is not None else find_checkpoints()
if not candidates:
    raise SystemExit(
        f"no checkpoint matched {PATTERNS} under {SEARCH}. Upload the .pt or "
        ".ckpt, or set CHECKPOINTS to its path.")

print(f"{'checkpoint':52s} {'MB':>7s} {'modified':>17s}  sha256[:12]")
for path in candidates:
    stat = os.stat(path)
    stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime))
    print(f"{path[-52:]:52s} {stat.st_size / 1e6:7.0f} {stamp:>17s}  "
          f"{digest(path)}")
print(f"\\nscoring {len(candidates)} checkpoint(s) as one averaged ensemble")
'''


COLAB_GOLD_CELL = '''
# --- Assemble the gold 58 out of the cache ----------------------------------
# Same inputs notebook 08 used to print its per-epoch gold, so a disagreement
# between that number and this one is the export or the checkpoint, not the
# data.
labels_frame = pd.read_parquet(find_one("train_labels.parquet"))
sides = pd.read_parquet(find_one("side_predictions.parquet"))
series_meta_train = pd.read_csv(find_competition_file("train_series.csv"))
train_csv = pd.read_csv(find_competition_file("train.csv"))
gold = train_csv[train_csv["ACL"].notna()].reset_index(drop=True)

frames, shard_arrays = [], {}
for cache_path in find_inputs("cache_train_series_*.npy"):
    manifest_path = cache_path.replace("cache_", "manifest_").replace(".npy",
                                                                     ".parquet")
    if not os.path.exists(manifest_path):
        stem = os.path.basename(cache_path).replace("cache_", "").replace(".npy", "")
        hits = [m for m in find_inputs("manifest_train_series_*.parquet")
                if os.path.basename(m).startswith(f"manifest_{stem}")]
        if not hits:
            continue
        manifest_path = hits[0]
    frame = pd.read_parquet(manifest_path)
    frame["cache_path"] = cache_path
    shard_arrays[cache_path] = np.load(cache_path, mmap_mode="r")
    frames.append(frame)

manifest = pd.concat(frames, ignore_index=True)

gold_labels = gold[[config.ID_COL] + config.LABELS].copy()
for name in config.LABELS:
    gold_labels[f"{name}_weight"] = 1.0

# max_series is TUNED_MAX_SERIES on purpose: this is the submission geometry
# being measured, not the training one. They are the same number today.
gold_index = knee_dataset.StudyIndex(manifest, series_meta_train, gold_labels,
                                     sides, max_series=TUNED_MAX_SERIES)
print(f"gold studies with cached series: {len(gold_index)} of {len(gold)}")
if len(gold_index) < len(gold):
    print("  NOTE: a gold study whose shard is missing is dropped here, and a "
          "macro over fewer than 58 is not comparable to the references below.")
'''


COLAB_SCORE_CELL = '''
from tqdm.auto import tqdm

KEEP = np.linspace(0, config.CACHE_SLICES - 1, TUNED_SLICES).round().astype(int)
print(f"slice subsample: {TUNED_SLICES} of {config.CACHE_SLICES} -> {KEEP.tolist()}")


def score_study(study):
    """One study through the tuned path, at the submission geometry.

    The cache route rather than notebook 10's DICOM route -- there is no test
    DICOM off Kaggle. The two used to disagree on WHICH series survive the
    cap: the cache path (StudyIndex.series_for) spreads its pick with
    np.linspace while notebook 10 took volumes[:TUNED_MAX_SERIES], a head
    truncation, which selected differently for the 32.5% of studies carrying
    more than five series. Notebook 10's `pick_series` now applies the
    linspace rule too, so this measurement and the submission read the same
    images.
    """
    item = knee_dataset.load_study(gold_index, study, shard_arrays)
    volumes = item["volumes"]
    if volumes.shape[1] > TUNED_SLICES:
        volumes = volumes[:, KEEP]
    elif volumes.shape[1] < TUNED_SLICES:
        raise SystemExit(
            f"cache supplies {volumes.shape[1]} slices and TUNED_SLICES is "
            f"{TUNED_SLICES}. This is notebook 09's thinned export; scoring at "
            "24 needs the full shards.")

    batch = torch.from_numpy(np.ascontiguousarray(volumes)).unsqueeze(0).to(device)
    meta = torch.from_numpy(item["meta"]).unsqueeze(0).to(device)
    mask = torch.ones(1, volumes.shape[0], dtype=torch.bool, device=device)
    with torch.no_grad(), torch.amp.autocast("cuda"):
        return np.mean([torch.sigmoid(m(batch, meta, mask).float()).cpu().numpy()[0]
                        for m in tuned_models], axis=0)


gold_ids, rows = [], []
for study in tqdm(gold_index.studies, desc="gold", unit="study"):
    rows.append(score_study(study))
    gold_ids.append(study)

gold_probs = np.stack(rows)
truth = gold.set_index(config.ID_COL).loc[gold_ids]
print(f"scored {len(gold_ids)} studies")
'''


COLAB_TABLE_CELL = '''
from sklearn.metrics import roc_auc_score


def auc(y_true, scores):
    return (float("nan") if len(np.unique(y_true)) < 2
            else float(roc_auc_score(y_true, scores)))


# The frozen DINOv2 path, retired 2026-08-23 at LB 0.787. Historical context;
# beating it has not been news since the first capacity rung.
FROZEN = dict(zip(config.LABELS,
                  [0.728, 0.671, 0.698, 0.680, 0.895, 0.899,
                   0.772, 0.897, 0.746, 0.862, 0.711, 0.683]))

# Both fold 1 at 16 x 5, so a fold-1 run is directly comparable and any other
# fold is not.
REFERENCE_GOLD = 0.861     # fold 1, 16 x 5, no aux losses -> LB 0.889
BEST_GOLD = 0.865          # fold 1, 24 x 5              -> LB 0.891 (best)

print(f"{'finding':18s} {'this run':>9s}  {'frozen (retired)':>16s}")
print("-" * 47)
scores = []
for i, name in enumerate(config.LABELS):
    value = auc(truth[name].to_numpy().astype(int), gold_probs[:, i])
    scores.append(value)
    print(f"{name:18s} {value:9.3f}  {FROZEN[name]:16.3f}")
print("-" * 47)
macro = float(np.nanmean(scores))
print(f"{'macro':18s} {macro:9.3f}  {np.mean(list(FROZEN.values())):16.3f}")

print(f"\\ngeometry scored: {TUNED_SLICES} slices x {TUNED_MAX_SERIES} series"
      f"  |  backbone: {TUNED_BACKBONE}")
print(f"the comparisons that matter (fold 1, 16 x 5):")
print(f"  vs reference  gold {REFERENCE_GOLD:.3f} (LB 0.889): "
      f"{macro - REFERENCE_GOLD:+.3f}")
print(f"  vs best       gold {BEST_GOLD:.3f} (LB 0.891): "
      f"{macro - BEST_GOLD:+.3f}")
if abs(macro - REFERENCE_GOLD) < 0.02:
    print("  -> inside the gold 58's ~0.02 noise floor: this is a null "
          "against the reference, whatever the frozen column says.")

# The reconciliation this notebook exists for. Training logs a gold score per
# epoch from the in-memory model; this scores the same studies from weights
# read off disk. They must agree. If they do not, the export or the checkpoint
# choice is wrong, and it is worth knowing here rather than after a submission.
TRAINING_GOLD = None    # paste the selected epoch's logged gold
if TRAINING_GOLD is not None:
    drift = macro - TRAINING_GOLD
    verdict = "agrees" if abs(drift) < 0.005 else "DISAGREES -- export is wrong"
    print(f"\\nvs training log {TRAINING_GOLD:.3f}: {drift:+.3f}  {verdict}")
'''


def _colab_ensemble_cells(nb):
    """The Colab copy's tail: score the gold 58 instead of writing a submission.

    Notebook 10 proper reads raw test DICOM, which exists only inside a Kaggle
    submission run, so the copy cannot end the way the original does. It ends
    at the one measurement Kaggle cannot make -- labelled studies through the
    submission's own geometry -- which is what a checkpoint has to pass before
    a Kaggle session is spent on it.
    """
    nb.md("""
## 5. The gold 58, out of the cache

Same shards, same manifest, same `StudyIndex` notebook 08 scores with, so a
disagreement between the epoch log and this table is the export or the
checkpoint choice -- never the data.
""")
    nb.code(COLAB_GOLD_CELL)
    nb.code(COLAB_SCORE_CELL)

    nb.md("""
## 6. Per-finding table

`TRAINING_GOLD` at the bottom is the reconciliation: paste the selected epoch's
logged gold and the cell says outright whether the export agrees with it.
""")
    nb.code(COLAB_TABLE_CELL)

    nb.md("""
## Next

A checkpoint that reproduces its training gold here is ready for a Kaggle
session; one that does not has an export problem, and submitting it spends
~2 h to learn something this cell just said for free.

**Two things this notebook cannot tell you.** It never touches raw DICOM, so
it does not exercise `build_volumes`, `predict_side`, or the fallback path --
a study that fails to decode on Kaggle fails there and not here. And the gold
58 were already spent on checkpoint selection, so a macro read off this table
is not an unbiased estimate of anything; it is a consistency check against a
number that came from the same 58 studies.
""")


def build_ensemble_notebook(colab=False):
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

    if colab:
        nb.md("""
**This is the Colab copy of notebook 10.** Generated from the same source as
`10_submit_ensemble.ipynb` -- edit `tools/make_notebook.py` and regenerate,
never this file.

**It cannot produce a submission, and that is not a limitation of the copy.**
Notebook 10 builds its volumes from raw test DICOM under
`config.DATA_ROOT / "test_series"`. The test set is hidden: it exists only
inside a Kaggle submission run, and there is no route to it from Colab. A
literal copy would find no directory, fall back to the training prior for
every study, write a `submission.csv` of twelve constants and report zero
errors -- the failure this project keeps paying for, one more time.

**So the copy stops where the data stops and does the measurement instead.**
The tuned path runs at the submission's own geometry over the 58 gold studies,
read from the cache shards. That is the check Kaggle's notebook can never run
-- it has no labels there -- and it answers the question that actually gates a
submission: *does this checkpoint, loaded from disk, score what training said
it scored?*

**Geometry: 16 slices x 5 series**, the same constants notebook 08 trains at,
and the reference geometry every fold-1 number on record belongs to. The cache
holds 24 slices per series, so 24 is the ceiling without rebuilding the shards
and notebook 09's thinned export (8 slices) cannot feed a 16- or 24-slice run
-- the scoring cell raises rather than scoring quietly on short volumes. Set
both constants to match whatever the checkpoint was trained at: the 24-slice
LB-0.891 checkpoint needs 24 **and** 5.
""")

        nb.md("## 0. Stage the inputs")
        nb.code(colab_stage_cell(
            pip="timm kagglehub",
            header=COLAB_STAGE_HEADER_10,
            working_note=COLAB_STAGE_WORKING_10,
            short_shards=COLAB_STAGE_SHORT_SHARDS_10,
            tail=COLAB_STAGE_TAIL_10))

    nb.md("## 1. Repo modules" if colab else "## Setup")
    nb.code(bootstrap_cell(MODEL_MODULES, require_raw_dicom=not colab))

    if colab:
        nb.md("## 2. Geometry and constants")

    setup_source = """
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
# at 8 slices and 3 series must be scored with 8 and 3 here.
#
# CURRENT: 16 x 5, matching notebook 08. P0.4's cap of 6 measured LB 0.885
# against 0.888 at 5 and was reverted.
#
# **Every checkpoint trained before P0.4 was trained at 5 and must be scored
# at 5.** That is convnext_fold1.pt from the P1 run, the 16 x 5 LB-0.889
# reference, and the 24 x 5 LB-0.891 best (which needs 24 here too).
#
# **Score a checkpoint at the geometry it was trained at.** Nothing enforces
# it: the aggregation masks the slice axis and pads the series axis, so a
# mismatch runs clean and simply scores worse.
TUNED_SLICES = 16
TUNED_MAX_SERIES = 5

# Must match notebook 08's BACKBONE. This was hard-coded to ConvNeXt here
# while notebook 08 exposed it as a constant, so a backbone experiment would
# have trained one trunk and scored another; the strict load would have caught
# it, but only after the hours were spent. Now it travels with the slice and
# series counts -- the three are re-uploaded together.
#
# CURRENT: ConvNeXt-T (768 features, 27.8M params), matching notebook 08.
# The DINOv2 axis is closed -- fine-tuned it measured gold 0.820 against a
# 0.861 gate -- so a ViT name here would be scoring weights this project no
# longer trains.
TUNED_BACKBONE = "convnext_tiny.fb_in22k_ft_in1k"

# Combine fold members by rank rather than by mean probability. Free, and
# strictly better aligned to a rank metric -- but it changes the submission's
# numbers, and every LB score on record was produced by the probability mean,
# so it is off until one submission measures it. No effect with one member.
MEMBER_RANK_AVERAGE = False

# Test-time augmentation: predict on the mirrored knee too, swap the
# medial/lateral findings back, average. DOUBLES inference time -- at ~1.65 h
# per fold that is the difference between five folds fitting the 9 h cap and
# not -- so it is off until it is shown to pay for itself.
FLIP_TTA = False

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
"""

    if colab:
        # The Colab copy's markdown states the geometry it measures at, so it
        # has to be checked against the constant rather than assumed. Asserted
        # rather than replaced: a str.replace that no longer matches is a
        # silent no-op, and this is the constant whose mismatch scores worse
        # without erroring.
        assert "TUNED_SLICES = 16" in setup_source, (
            "notebook 10's TUNED_SLICES is no longer 16; update the Colab "
            "copy's geometry paragraph to match before changing this")
        # Every place an input can be on this machine, in priority order, and
        # the whole of find_inputs rewritten around it rather than one glob
        # bolted on: the Drive copy lives outside /kaggle/input, the roots
        # overlap, and one of them is reached through a symlink.
        assert COLAB_FIND_INPUTS_OLD in setup_source, (
            "find_inputs in notebook 10 has changed; COLAB_FIND_INPUTS_OLD no "
            "longer matches and the Colab copy would silently keep the "
            "Kaggle-only search")
        setup_source = setup_source.replace(COLAB_FIND_INPUTS_OLD,
                                            COLAB_FIND_INPUTS_NEW)
        # find_competition_file's Kaggle globs never match on Colab, where the
        # two CSVs are copied into ./data by the staging cell.
        setup_source = setup_source.replace(
            '    for candidate in (f"/kaggle/input/competitions/*/{name}",',
            '    for candidate in (f"data/{name}",\n'
            '                      f"/kaggle/input/competitions/*/{name}",')
    nb.code(setup_source)

    if colab:
        nb.md("""
## 3. Which checkpoint

Before the load, not after it: this cell decides what gets loaded and prints
size, mtime and a content hash for each candidate. `CHECKPOINTS` pins the
choice, and the load cell below consumes this list rather than globbing again.
""")
        nb.code(COLAB_CHECKPOINT_CELL)

    nb.md(("## 4. Load both models" if colab else "## Load both models") + """

Everything from disk. The ConvNeXt checkpoint carries its own backbone weights,
so `pretrained=False` is correct here -- timm cannot reach the network anyway.
""")

    models_source = """
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
#
# TUNED_DEFAULTS is the trunk's input statistics. They travel in checkpoints
# written from 2026-08-28 onward, and `KneeNet` adopts a checkpoint's own
# statistics over whatever is passed here -- so an older file scored with the
# right TUNED_BACKBONE is still correct, and a newer one cannot be scored
# wrongly at all. Derived from TUNED_BACKBONE so the two cannot disagree.
TUNED_DEFAULTS = net.backbone_defaults(TUNED_BACKBONE)
print("input normalisation:", "on" if TUNED_DEFAULTS["input_mean"] else "off")

# WHICH FAMILY OF WEIGHTS TO SCORE. Notebook 08 now writes three per fold:
#
#   convnext_fold{N}.pt    the best mined-val epoch    <- selected on a
#                                                         biased metric
#   avg_fold{N}_ema.pt     EMA of the trajectory       <- selection-free
#   avg_fold{N}_swa.pt     equal average, last 25%     <- selection-free
#
# **Score one family at a time.** These are three views of the same run, so a
# glob that catches two of them averages a fold with itself and quietly gives
# it double weight in the ensemble. The guard below refuses that.
TUNED_PATTERNS = ("convnext_fold*.pt", "lit_fold*.ckpt")

tuned_paths = []
for pattern in TUNED_PATTERNS:
    tuned_paths += find_inputs(pattern)

import re


# (family, fold) for a checkpoint, from its filename.
#
# Two files describing the same fold's weights must never both be loaded. The
# averaging here is unweighted, so a fold present twice gets twice the say --
# 2/6 of a five-fold ensemble instead of 1/5 -- and nothing in the output says
# so.
#
# The family comes from the suffix notebook 08 writes. The fold comes from the
# digits after "fold", which every name this project produces carries:
# convnext_fold1.pt, lit_fold1.ckpt, lit_fold1-v1.ckpt, lit_fold1_old.ckpt and
# avg_fold1_ema.pt are five names for two distinct sets of weights, and the
# first four are all fold 1 "selected".
#
# A comment rather than a docstring on purpose: a triple-quoted string in a
# cell body terminates the generator's own cell source. PROJECT.md lists it.
def checkpoint_identity(path):
    name = os.path.basename(path)
    family = ("ema" if name.endswith("_ema.pt") else
              "swa" if name.endswith("_swa.pt") else "selected")
    match = re.search(r"fold(\\d+)", name)
    # No fold number means the identity cannot be established, so the file is
    # keyed by its own name and can only collide with itself.
    return family, (match.group(1) if match else name)


_seen = {}
for path in tuned_paths:
    _seen.setdefault(checkpoint_identity(path), []).append(
        os.path.basename(path))

# 1. More than one family: three views of the same runs, averaged together.
_families = sorted({family for family, _ in _seen})
if len(_families) > 1:
    _listing = "; ".join(
        f"{fam}={sorted(n for (f, _), names in _seen.items() if f == fam for n in names)}"
        for fam in _families)
    raise SystemExit(
        f"checkpoints from more than one family matched TUNED_PATTERNS: "
        f"{_listing}\\nThese are the same runs' weights several ways, so "
        "averaging across them double-counts every fold. Narrow "
        "TUNED_PATTERNS to one family.")

# 2. One fold under two names inside a family. This is the standing trap:
#    `convnext_fold{N}.pt` is a bit-exact export of `lit_fold{N}.ckpt`, and
#    both match TUNED_PATTERNS. Lightning's `-vN` collision suffixes and the
#    `_old` files kept on disk land here too.
_duplicated = {k: v for k, v in _seen.items() if len(v) > 1}
if _duplicated:
    raise SystemExit(
        "the same fold matched TUNED_PATTERNS under more than one name: "
        + "; ".join(f"fold {fold} -> {sorted(names)}"
                    for (_, fold), names in sorted(_duplicated.items()))
        + "\\nThese are one set of weights each, and averaging a fold with "
        "itself gives it double weight in the ensemble. Attach exactly one "
        "file per fold -- the exported .pt, not the .ckpt it came from.")

print(f"checkpoint family: {_families[0] if _families else 'none'}  "
      f"({len(_seen)} fold(s))")

# SOUP averages the checkpoints in WEIGHT space and runs one model, instead of
# running each and averaging probabilities. Wortsman et al. (ICML 2022) report
# that this usually matches or beats the ensemble, and it is free at inference
# -- five folds for the cost of one, which is the whole 9-hour budget question
# and the Efficiency Prize's as well.
#
# **Only valid for models in the same loss basin.** Notebook 08 seeds every
# fold identically so they share an initialisation; checkpoints from before
# that seeding was added do not, and averaging them produces a worse model
# with nothing in the output saying so. Default off: probability averaging is
# what every leaderboard number on record was produced by, and this switches
# the submission's semantics.
SOUP = False

tuned_states = [net.load_knee_state(path, map_location="cpu")
                for path in tuned_paths]
for path in tuned_paths:
    print("  tuned:", os.path.basename(path))

if SOUP and len(tuned_states) > 1:
    tuned_states = [averaging.average_states(tuned_states)]
    print(f"  souped {len(tuned_paths)} checkpoints into one set of weights")

tuned_models = []
for state in tuned_states:
    backbone = net.build_backbone(TUNED_BACKBONE, pretrained=False)
    model = net.KneeNet(backbone, backbone.num_features,
                        input_mean=TUNED_DEFAULTS["input_mean"],
                        input_std=TUNED_DEFAULTS["input_std"]).to(device)
    model.load_state_dict(state)
    model.eval()
    tuned_models.append(model)

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
"""

    if colab:
        # Load the candidates the cell above listed and hashed, in that order.
        # A second glob here would ignore a pinned CHECKPOINTS and could load
        # a different set than the one whose hashes were just printed -- the
        # exact ambiguity that cell exists to remove.
        old_glob = """TUNED_PATTERNS = ("convnext_fold*.pt", "lit_fold*.ckpt")

tuned_paths = []
for pattern in TUNED_PATTERNS:
    tuned_paths += find_inputs(pattern)"""
        assert old_glob in models_source
        models_source = models_source.replace(old_glob, """TUNED_PATTERNS = PATTERNS
tuned_paths = list(candidates)""")
    nb.code(models_source)

    if colab:
        # Everything below this point reads raw test DICOM, which does not
        # exist off Kaggle. The copy ends with the measurement instead.
        _colab_ensemble_cells(nb)
        nb.write(NOTEBOOKS / "10_submit_ensemble_colab.ipynb")
        return

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

# FAST_DECODE -- stop decoding images the model never looks at.
#
# Runtime is a SCORED quantity in the Efficiency Prize, not merely a ceiling:
# Efficiency = AUC / (Benchmark - maxAUC) + RuntimeSeconds / 32400, minimised,
# which at Benchmark 0.5 and maxAUC 0.95 makes one hour of runtime worth
# 0.050 AUC. For scale, the five-fold ensemble's whole advantage over a single
# fold is +0.017 -- twenty minutes.
#
# The reference path decodes EVERY series in the study at CACHE_SLICES depth,
# and only afterwards does `pick_series` keep TUNED_MAX_SERIES of them and
# `KEEP` keep TUNED_SLICES of the 24. On a seven-series study that is 7 x 24 =
# 168 slices decoded where 5 x 16 = 80 are used -- 52% of the pixel decoding
# thrown away, plus a header read for every file of two series that are never
# scored. 32.5% of studies carry more than five series.
#
# With this on, series are chosen before anything is read, so a seven-series
# study reads and decodes five. Predictions are unchanged: `pick_series` needs
# only the count, and both sides order by SeriesInstanceUID -- the sorted
# directory name -- so exactly the series the reference path would have kept
# are the ones decoded.
#
# **The slice-depth half of this is blocked, and it was the bigger half.**
# Decoding each series to TUNED_SLICES instead of CACHE_SLICES would drop a
# third of the remaining pixel work, and `cache.build_series_volume(headers,
# slices=...)` implements it correctly -- `submission_indices` composes the
# coarse and fine selections so the same raw files are read, which matters
# because picking 16 evenly across the raw stack in one step lands on
# different files for 41 of the 44 series lengths between 16 and 60.
#
# What stops it is the SIDE CLASSIFIER: it takes the slice axis as input
# channels and its first convolution has 24 of them, so a 16-deep volume makes
# every study raise and fall back to the training prior -- a submission-wide
# AUC of 0.5, which is exactly what tools/check_submission.py caught. The
# laterality DICOM tag would sidestep it, but only 58.5% of studies carry it
# on any series; the other 1,828 need the model. Unblocking this means
# retraining the side classifier at TUNED_SLICES depth (notebook 04, minutes),
# and that is a behaviour change to laterality -- which silently transposes
# four of the twelve findings when it is wrong -- so it is its own experiment,
# not a rider on this one.
FAST_DECODE = True
AB_CHECK = True     # score both paths and report the difference


def build_volumes(study_uid, fast=False):
    directory = test_root / study_uid
    if not directory.exists():
        return [], []
    series_dirs = sorted(p for p in directory.iterdir() if p.is_dir())

    # Choose the series BEFORE reading anything. `pick_series` needs only the
    # count, and both sides order by SeriesInstanceUID -- which is the sorted
    # directory name -- so this selects exactly what the reference path keeps.
    if fast:
        series_dirs = [series_dirs[i] for i in pick_series(len(series_dirs))]

    # Full depth either way: the side classifier reads CACHE_SLICES as input
    # channels. See the FAST_DECODE note above for what that costs.
    volumes, uids = [], []
    for series_dir in series_dirs:
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


def pick_series(count):
    \"\"\"Which series survive the cap -- the SAME rule training uses.

    `StudyIndex.series_for` spreads its pick with np.linspace; this took
    `volumes[:TUNED_MAX_SERIES]`, a head truncation. For a seven-series study
    that is positions [0, 2, 3, 4, 6] against [0, 1, 2, 3, 4], so the model was
    scored on a different subset of the study than it was trained to read.
    **32.5% of studies carry more than five series**, and nothing about the
    mismatch raises: the shapes agree and the prediction is merely built from
    the wrong images.

    Both sides order series by SeriesInstanceUID -- `build_volumes` sorts the
    directory names, the cache manifest is written in that order -- so once
    the rule matches, the selection matches.
    \"\"\"
    if count <= TUNED_MAX_SERIES:
        return list(range(count))
    keep = np.linspace(0, count - 1, TUNED_MAX_SERIES).round().astype(int)
    return np.unique(keep).tolist()


def tuned_predict(volumes, uids, side, fast=False):
    # Mirrored and subsampled exactly as during training.
    #
    # Under FAST_DECODE the series were already chosen in build_volumes, so
    # picking again would take a subset of a subset and silently score the
    # model on a fifth of the intended images. Slice depth is untouched, so
    # KEEP still applies in both paths.
    chosen = range(len(volumes)) if fast else pick_series(len(volumes))
    picked = [volumes[i] for i in chosen]
    picked_uids = [uids[i] for i in chosen]
    mirrored, planes = [], []
    for volume, uid in zip(picked, picked_uids):
        plane = series_meta.loc[uid]["Anatomical_Plane"] if uid in series_meta.index else None
        planes.append(plane)
        mirrored.append(knee_dataset.mirror_volume(volume, side, plane)[KEEP])

    meta = meta_tensor(picked_uids, side)
    mask = torch.ones(1, len(mirrored), dtype=torch.bool, device=device)

    def run(stack):
        tensor = torch.from_numpy(
            np.ascontiguousarray(np.stack(stack))).unsqueeze(0).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            # (n_members, n_labels) -- NOT averaged here. Members are combined
            # once, over the whole test set, so they can be rank-averaged.
            return np.stack([
                torch.sigmoid(m(tensor, meta, mask).float()).cpu().numpy()[0]
                for m in tuned_models])

    out = run(mirrored)

    if FLIP_TTA:
        # Mirror the knee and read the prediction back through the
        # medial/lateral swap. The plan calls hflip "safe post-laterality
        # normalisation" -- it is not, on its own: a mirrored knee puts the
        # medial compartment where the lateral one was, so four of the twelve
        # findings come back transposed. augment.lateral_swap_order() is the
        # permutation that undoes it, and augment.flip_volume picks the axis
        # from the same table mirror_volume canonicalises with.
        flipped = [knee_augment.flip_volume(v, plane)
                   for v, plane in zip(mirrored, planes)]
        out = 0.5 * (out + run(flipped)[:, knee_augment.lateral_swap_order()])

    return out


ids, frozen_rows, tuned_rows, failures = [], [], [], 0

# Timed separately from setup. Model loading is a fixed cost, and amortising
# it over three public studies inflates the per-study figure enormously --
# it is noise against a real test set.
loop_start = time.time()

# Decode and forward timed apart. The only runtime figure this project has
# ever had is a single projected total, which cannot answer either question
# the efficiency work turns on: how much a sixth fold costs (forward only) and
# how much FAST_DECODE saves (decode only). One dry run now answers both.
decode_seconds = 0.0
forward_seconds = 0.0
ab_deltas = []

for study in tqdm(studies, desc="studies", unit="study"):
    try:
        _t = time.time()
        volumes, uids = build_volumes(study, fast=FAST_DECODE)
        if not volumes:
            raise ValueError("no readable series")
        side = predict_side(volumes)
        decode_seconds += time.time() - _t

        meta = meta_tensor(uids, side)
        mask = torch.ones(1, len(volumes), dtype=torch.bool, device=device)

        _t = time.time()
        frozen_rows.append(frozen_predict(volumes, meta, mask)
                           if (frozen_models and USE_FROZEN) else None)
        row = (tuned_predict(volumes, uids, side, fast=FAST_DECODE)
               if (tuned_models and USE_TUNED) else None)
        forward_seconds += time.time() - _t
        tuned_rows.append(row)

        # Score the same study the reference way and compare. FAST_DECODE
        # reads the same raw files but normalises over a shallower stack, so
        # the predictions move a little; this reports how much instead of
        # arguing about it.
        if AB_CHECK and FAST_DECODE and row is not None:
            slow_volumes, slow_uids = build_volumes(study, fast=False)
            slow = tuned_predict(slow_volumes, slow_uids,
                                 predict_side(slow_volumes), fast=False)
            ab_deltas.append(np.abs(row - slow).max())
    except Exception as exc:  # noqa: BLE001
        failures += 1
        print(f"  fallback for {study[-12:]}: {str(exc)[:80]}")
        prior = np.array([PRIORS[n] for n in config.LABELS])
        frozen_rows.append(prior)
        # One row per member, so the stack below keeps its shape whether a
        # study succeeded or fell back.
        tuned_rows.append(np.repeat(prior[None], max(len(tuned_models), 1),
                                    axis=0))
    ids.append(study)

loop_elapsed = time.time() - loop_start
elapsed = time.time() - START
setup = elapsed - loop_elapsed
print(f"\\n{len(ids)} studies in {loop_elapsed / 60:.1f} min "
      f"({failures} fallbacks); setup took {setup / 60:.1f} min")
print(f"per study: {loop_elapsed / max(len(ids), 1):.2f}s")

_n = max(len(ids), 1)
print(f"  decode  {decode_seconds / _n:.2f}s/study"
      f"   ({decode_seconds / max(decode_seconds + forward_seconds, 1e-9):.0%} of the loop)")
print(f"  forward {forward_seconds / _n:.2f}s/study"
      f"   over {len(tuned_models)} model(s)"
      + (f", {forward_seconds / _n / len(tuned_models):.2f}s each"
         if tuned_models else ""))
print(f"  FAST_DECODE {'on' if FAST_DECODE else 'off'}")
if AB_CHECK and FAST_DECODE:
    print("  AB_CHECK is ON: every study is also scored the reference way, "
          "which is untimed but\n  inflates the loop total and therefore the "
          "projection below. Set AB_CHECK = False\n  before reading a runtime "
          "figure; the decode and forward lines above are unaffected.")
if ab_deltas:
    print(f"\\nA/B against the reference decode path, {len(ab_deltas)} studies:")
    print(f"  max absolute prediction difference {max(ab_deltas):.4f}"
          f"   mean {sum(ab_deltas) / len(ab_deltas):.4f}")
    print("  This is the intensity-percentile shift, not a different slice "
          "selection -- the raw files are the same. Anything above ~0.005 "
          "means stop and re-read cache.submission_indices.")
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

# The fold members are combined HERE, over the whole test set, rather than
# inside tuned_predict per study -- because rank-averaging needs every study's
# score for a member before it can rank any of them.
#
# Why ranks: the metric is macro AUC, which reads only the ordering of studies
# within a finding. A probability mean lets a member with a confident output
# distribution outvote one with a compressed distribution even when the two
# rank studies identically. Ranking first gives every member one vote.
# to_ranks below already did this for the frozen-vs-tuned blend; it was never
# applied ACROSS FOLDS, which is the ensemble that matters now that BLEND=1.0
# retires the frozen path.
tuned_matrix = None
if have_tuned:
    # (studies, members, labels) -> (members, studies, labels)
    per_member = np.stack(tuned_rows).transpose(1, 0, 2)
    if MEMBER_RANK_AVERAGE and per_member.shape[0] > 1 and len(ids) > 1:
        tuned_matrix = averaging.rank_average(list(per_member))
        print(f"combining {per_member.shape[0]} members by RANK average")
    else:
        tuned_matrix = per_member.mean(axis=0)
        print(f"combining {per_member.shape[0]} member(s) by probability mean")

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



# ---------------------------------------------------------------------------
# 11 -- error analysis: which studies fail, and what do they look like
# ---------------------------------------------------------------------------

def build_error_analysis_notebook():
    nb = Notebook(gpu=True)

    nb.md("""
# 11 — Error Analysis

**Deliberately separate from notebook 08.** That notebook trains; this one
looks. Keeping them apart means a diagnostic can be re-run against any
checkpoint without spending a training session, and a training run is not
slowed down by rendering images nobody asked for.

Seven consecutive changes have been judged on one aggregate number that
provably cannot resolve them: the gold set's macro standard error is
0.015-0.025 and every remaining idea is worth less than that. This notebook
asks the questions the number cannot.

1. **Per finding**, with the Hanley-McNeil standard error beside each one, so
   a gap is never read without its noise.
2. **By study descriptor** — series count, fluid-sensitive count, planes
   present. Is the model uniformly mediocre, or losing a definable slice?
3. **The failing studies as pictures**, rendered exactly as the model sees
   them: after laterality mirroring and the slice subsample.

The third is the one no metric can substitute for. The two worst bugs this
project has had — slice order reversed by `InstanceNumber`, sagittal mirroring
flipping the wrong axis — raised nothing, scored plausibly, and cost about
0.03 each. They are visible in a contact sheet and nowhere else.

**Attach:** the cache shards, `knee-labels`, `knee_side_classifier`, the
competition data, and whichever checkpoint you want to examine.
""")

    nb.md("## 1. Setup")
    nb.code("""
!pip install -q timm

import glob
import os

import numpy as np
import pandas as pd
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
""")

    nb.md("## 2. Repo modules")
    nb.code(bootstrap_cell(INSPECT_MODULES, require_raw_dicom=False))

    nb.md("""
## 3. Which checkpoint, and at what geometry

**The geometry must match what the checkpoint was trained at.** Nothing
enforces it: the aggregation masks the slice axis and pads the series axis, so
a mismatch runs clean and simply scores worse — and this notebook would then
report failures that are artefacts of the mismatch rather than of the model.
""")

    nb.code("""
# Point this at the checkpoint to examine. Either format loads: the exported
# .pt or Lightning's .ckpt.
CHECKPOINT = None          # None -> take the newest convnext_fold*.pt found

SLICES = 16                # notebook 08's SLICES_PER_SERIES for this checkpoint
MAX_SERIES = 5             # notebook 08's MAX_SERIES for this checkpoint
BACKBONE = "convnext_tiny.fb_in22k_ft_in1k"


def find_inputs(pattern):
    found = sorted(glob.glob(f"data/**/{pattern}", recursive=True))
    found += sorted(glob.glob(f"/kaggle/working/{pattern}"))
    for root in sorted(glob.glob("/kaggle/input/*")):
        if os.path.basename(root) == "competitions":
            continue
        found += sorted(glob.glob(f"{root}/**/{pattern}", recursive=True))
    seen, unique = set(), []
    for path in found:
        key = os.path.realpath(path)
        if key not in seen:
            seen.add(key); unique.append(path)
    return unique


def find_one(pattern):
    hits = find_inputs(pattern)
    if not hits:
        raise SystemExit(f"missing input: {pattern}")
    return hits[0]


def find_competition_file(name):
    for candidate in (f"data/{name}", f"/kaggle/input/competitions/*/{name}",
                      f"/kaggle/input/*/{name}"):
        hits = sorted(glob.glob(candidate))
        if hits:
            return hits[0]
    raise SystemExit(f"missing competition file: {name}")


if CHECKPOINT is None:
    candidates = find_inputs("convnext_fold*.pt") + find_inputs("lit_fold*.ckpt")
    if not candidates:
        raise SystemExit("no checkpoint found; set CHECKPOINT explicitly")
    CHECKPOINT = max(candidates, key=os.path.getmtime)
print("examining:", CHECKPOINT)
print(f"geometry : {SLICES} slices x {MAX_SERIES} series   backbone {BACKBONE}")
""")

    nb.md("## 4. Assemble the gold studies")
    nb.code("""
labels = pd.read_parquet(find_one("train_labels.parquet"))
sides = pd.read_parquet(find_one("side_predictions.parquet"))
series_meta = pd.read_csv(find_competition_file("train_series.csv"))
train_csv = pd.read_csv(find_competition_file("train.csv"))
gold = train_csv[train_csv["ACL"].notna()].reset_index(drop=True)

frames, shards = [], {}
for cache_path in find_inputs("cache_train_series_*.npy"):
    stem = os.path.basename(cache_path).replace("cache_", "").replace(".npy", "")
    hits = [m for m in find_inputs("manifest_train_series_*.parquet")
            if os.path.basename(m).startswith(f"manifest_{stem}")]
    if not hits:
        continue
    frame = pd.read_parquet(hits[0])
    frame["cache_path"] = cache_path
    shards[cache_path] = np.load(cache_path, mmap_mode="r")
    frames.append(frame)
if not frames:
    raise SystemExit("no cache shards found -- attach the cache dataset")
manifest = pd.concat(frames, ignore_index=True)

gold_labels = gold[[config.ID_COL] + config.LABELS].copy()
for name in config.LABELS:
    gold_labels[f"{name}_weight"] = 1.0
gold_index = knee_dataset.StudyIndex(manifest, series_meta, gold_labels, sides,
                                     max_series=MAX_SERIES)
print(f"gold studies with cached series: {len(gold_index)} of {len(gold)}")
if len(gold_index) < len(gold):
    print("  NOTE: a study whose shard is missing is dropped, and a macro over")
    print("  fewer than 58 is not comparable to anything on record.")
""")

    nb.md("## 5. Score them")
    nb.code("""
KEEP = np.linspace(0, config.CACHE_SLICES - 1, SLICES).round().astype(int)
DEFAULTS = net.backbone_defaults(BACKBONE)

backbone = net.build_backbone(BACKBONE, pretrained=False)
model = net.KneeNet(backbone, backbone.num_features,
                    input_mean=DEFAULTS["input_mean"],
                    input_std=DEFAULTS["input_std"]).to(device)
model.load_state_dict(net.load_knee_state(CHECKPOINT, map_location=device))
model.eval()
print("input normalisation:", "on" if DEFAULTS["input_mean"] else "off")

ids, rows = [], []
for study in gold_index.studies:
    item = knee_dataset.load_study(gold_index, study, shards)
    volumes = item["volumes"]
    if volumes.shape[1] > SLICES:
        volumes = volumes[:, KEEP]
    elif volumes.shape[1] < SLICES:
        raise SystemExit(
            f"cache has {volumes.shape[1]} slices and SLICES is {SLICES} -- "
            "this is the thinned export; attach the full shards")
    batch = torch.from_numpy(np.ascontiguousarray(volumes)).unsqueeze(0).to(device)
    meta = torch.from_numpy(item["meta"]).unsqueeze(0).to(device)
    mask = torch.ones(1, volumes.shape[0], dtype=torch.bool, device=device)
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device == "cuda")):
        rows.append(torch.sigmoid(model(batch, meta, mask).float()).cpu().numpy()[0])
    ids.append(study)

preds = pd.DataFrame(np.stack(rows), columns=config.LABELS, index=ids)
truth = gold.set_index(config.ID_COL).loc[ids, config.LABELS].astype(float)

# Dumped so the same analysis can be re-run off the VM with
# tools/error_analysis.py, which shares this notebook's error definition.
out = preds.reset_index().rename(columns={"index": config.ID_COL})
out.to_parquet("/kaggle/working/gold_predictions.parquet", index=False)
print(f"scored {len(ids)} studies -> /kaggle/working/gold_predictions.parquet")
""")

    nb.md("""
## 6. Per finding, with the noise beside it

`se` is the Hanley-McNeil standard error. **If the difference you are looking
at is smaller than the SE, there is nothing there** — which has been true of
every change measured on this set so far.
""")

    nb.code("""
# The same rank-statistic AUC the training loop uses. Copied rather than
# imported: `src/model/lightning.py` is bootstrapped only into notebook 08,
# because it imports the Lightning framework at module scope and this notebook
# has no reason to need it.
def auc(y, s):
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    pos = y == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    uniq, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(uniq)); np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


# Hanley & McNeil (1982) -- the number that says whether a gap is real.
# A comment, not a docstring: a triple-quoted string in a cell body terminates
# the generator's own cell source. PROJECT.md lists this trap.
def auc_se(a, n_pos, n_neg):
    if not np.isfinite(a) or n_pos < 1 or n_neg < 1:
        return float("nan")
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    num = a * (1 - a) + (n_pos - 1) * (q1 - a * a) + (n_neg - 1) * (q2 - a * a)
    return float(np.sqrt(max(num, 0.0) / (n_pos * n_neg)))


scores = []
print(f"{'finding':18s} {'auc':>7s} {'+/- se':>8s} {'pos':>5s} {'neg':>5s}")
for name in config.LABELS:
    y, s = truth[name].to_numpy(), preds[name].to_numpy()
    value = auc(y, s)
    n_pos, n_neg = int((y == 1).sum()), int((y != 1).sum())
    scores.append(value)
    print(f"{name:18s} {value:7.3f} {auc_se(value, n_pos, n_neg):8.3f} "
          f"{n_pos:5d} {n_neg:5d}")
print(f"{'macro':18s} {np.nanmean(scores):7.3f}")
""")

    nb.md("""
## 7. Is the loss concentrated on a describable slice of studies?

Buckets with fewer than eight studies print `--`. An eye-catching 0.55 computed
from six studies, sitting beside numbers from fifty, is how this project has
fooled itself before.
""")

    nb.code("""
plane = series_meta["Anatomical_Plane"].astype(str)
enriched = series_meta.assign(
    _sag=(plane == "Sagittal").astype(int),
    _cor=(plane == "Coronal").astype(int),
    _ax=(plane == "Axial").astype(int),
    _fluid=series_meta["Fluid_Sensitive"].fillna(0).astype(int))
grouped = enriched.groupby(config.ID_COL)
features = pd.DataFrame({"n_series": grouped.size(),
                         "n_sagittal": grouped["_sag"].sum(),
                         "n_coronal": grouped["_cor"].sum(),
                         "n_axial": grouped["_ax"].sum(),
                         "n_fluid_sensitive": grouped["_fluid"].sum()})


def macro_on(mask):
    if mask.sum() < 8:
        return float("nan")
    return float(np.nanmean([auc(truth[n].to_numpy()[mask],
                                 preds[n].to_numpy()[mask])
                             for n in config.LABELS]))


for column, edges in (("n_series", [0, 4, 5, 6, 8, 99]),
                      ("n_fluid_sensitive", [0, 1, 2, 3, 99]),
                      ("n_sagittal", [0, 1, 2, 3, 99]),
                      ("n_coronal", [0, 1, 2, 99]),
                      ("n_axial", [0, 1, 2, 99])):
    values = features.reindex(truth.index)[column]
    parts = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = ((values >= low) & (values < high)).to_numpy()
        macro = macro_on(mask)
        parts.append(f"[{low},{high}):"
                     + (f"{macro:.3f}" if np.isfinite(macro) else "--")
                     + f"(n={int(mask.sum())})")
    print(f"{column:20s} " + "  ".join(parts))
""")

    nb.md("""
## 8. Synovitis, on the metric that can measure it

Pooled AUC cannot see this finding. On these 58 studies the **effusion** head
scores 0.722 as a synovitis classifier against the synovitis head's 0.701, and
the gold effusion label alone scores 0.698 — a number in that region is
obtainable with no synovitis-specific skill at all.

**Effusion-stratified conditional AUC** keeps only positive/negative pairs from
the same effusion stratum, which is the part of the ranking effusion does not
already decide. Everything below comes from `src/model/conditional.py`, the
same code the training loop logs every epoch and `tools/synovitis_gate.py`
reads the Stage 0 gate from.
""")

    nb.code("""
FINDING, STRATIFY_BY = "Synovitis", "Effusion"

y = truth[FINDING].to_numpy()
scores_syn = preds[FINDING].to_numpy()
strata_gold = truth[STRATIFY_BY].to_numpy()
strata_pred = conditional.strata_from_scores(preds[STRATIFY_BY].to_numpy())

for label, strata in (("gold effusion label", strata_gold),
                      ("predicted effusion (median split)", strata_pred)):
    stats = conditional.bootstrap_conditional(y, scores_syn, strata, draws=4000)
    counts = conditional.pair_counts(y, strata)
    print(f"{label:34s} conditional {stats['auc']:.3f}  "
          f"95% CI [{stats['lo']:.3f}, {stats['hi']:.3f}]  "
          f"P(<= 0.5) = {stats['p_at_or_below_chance']:.3f}  "
          f"({counts['within']} within-stratum pairs)")

print(f"{'pooled':34s} {conditional.auc(y, scores_syn):.3f}")
print(f"{'the effusion head, used directly':34s} "
      f"{conditional.auc(y, preds[STRATIFY_BY].to_numpy()):.3f}   "
      "<- if this is close to pooled, pooled is not measuring the finding")
print()

print("where the error sits")
for row in conditional.pair_decomposition(y, scores_syn, strata_gold):
    mark = "  <- within stratum, addressable" if row["within"] else ""
    print(f"  pos eff={row['pos_stratum']} vs neg eff={row['neg_stratum']}: "
          f"{row['pairs']:4d} pairs  {row['misranked']:5.1%} misranked  "
          f"{row['error_share']:5.1%} of the error{mark}")
print()

print("what a conditional target is worth in pooled terms, on these studies")
for row in conditional.pooled_from_conditional(
        y, scores_syn, strata_gold, [0.70, 0.75, 0.80, 0.85, 1.00]):
    print(f"  conditional {row['conditional']:.2f} -> pooled {row['pooled']:.3f}")
print("  the last line is the ceiling under an unchanged gating: arithmetic,")
print("  not a reachable target -- one model produces both halves.")
""")

    nb.md("""
### 8b. Contrast, and series composition

Two checks on the input rather than the model, from the plan's Stage 0.
Post-contrast series fed to the model as ordinary fluid-sensitive ones would
be worth more than any modelling change; genuinely absent contrast is a hard
ceiling on the finding and belongs in the write-up. **Absence of a string is
not absence of contrast** — about a fifth of this collection's series carry no
usable description, so the blind spot is printed beside the result.

Needs `index_train.parquet` (the header scan from notebook 01). Skipped with a
message if it is not attached, rather than reporting a clean bill it has not
earned.
""")

    nb.code("""
CONTRAST_MARKERS = r"\\+\\s*c\\b|gad|contrast|enhanc|subtract|arthro"

index_paths = find_inputs("index_train*.parquet")
if not index_paths:
    print("no index_train.parquet attached -- contrast cannot be ruled in or")
    print("out from the competition's series table, which carries no")
    print("SeriesDescription. Attach the notebook 01 output to run this.")
else:
    header = pd.read_parquet(index_paths[0], columns=[
        config.ID_COL, "SeriesInstanceUID", "SeriesDescription",
        "ProtocolName"]).drop_duplicates("SeriesInstanceUID")
    text = (header["SeriesDescription"].fillna("") + " | "
            + header["ProtocolName"].fillna("")).str.lower()
    hits = text.str.contains(CONTRAST_MARKERS, regex=True, na=False)
    blind = header["SeriesDescription"].fillna("").str.strip().str.lower().isin(
        ["", "dummyseriesdesc!"])

    flagged = header[hits].merge(series_meta, on=[config.ID_COL,
                                                  "SeriesInstanceUID"],
                                 how="left")
    print(f"{len(header)} series, {header[config.ID_COL].nunique()} studies")
    print(f"{len(flagged)} series in {flagged[config.ID_COL].nunique()} "
          "studies name contrast")
    for _, row in flagged.iterrows():
        print(f"  {str(row[config.ID_COL])[-12:]}  "
              f"{str(row['SeriesDescription'])[:34]:34s} "
              f"fluid={row.get('Fluid_Sensitive')} "
              f"fatsat={row.get('Fat_Suppression')}")
    routed = int((flagged.get("Fluid_Sensitive", pd.Series(dtype=float))
                  .fillna(0) == 1).sum())
    print(f"routed to the model as fluid-sensitive: {routed}"
          + ("   <- fix this before anything else" if routed else
             "   (no routing bug)"))
    print(f"blind spot: {int(blind.sum())} series ({blind.mean():.1%}) carry "
          "no usable description")

# 0d: does the score scale with how many fluid-sensitive fat-suppressed series
# a study happens to have? Compared against the LABEL's own correlation with
# the same count -- sicker knees are scanned with more sequences, so the raw
# number alone cannot say whether the model is reading protocol.
both = ((series_meta["Fluid_Sensitive"].fillna(0) == 1)
        & (series_meta["Fat_Suppression"].fillna(0) == 1)).astype(int)
per_study = series_meta.assign(_both=both).groupby(config.ID_COL)["_both"].sum()
count = per_study.reindex(truth.index).to_numpy(dtype=float)


def _logit(p):
    return np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))


print()
print(f"corr(n fluid+fatsat series, {FINDING} logit) "
      f"{np.corrcoef(count, _logit(scores_syn))[0, 1]:+.3f}")
print(f"the label's own correlation with the same count  "
      f"{np.corrcoef(count, y)[0, 1]:+.3f}")
print(f"on {len(count)} studies a correlation carries SE "
      f"{1 / np.sqrt(max(len(count) - 3, 1)):.2f} -- read this on the "
      "out-of-fold predictions, not here.")
""")

    nb.md("""
## 9. The failing studies, as pictures

Eight worst and two best, the latter as controls — without them there is no way
to tell whether the odd thing in a failure is odd, or just what knee MRI looks
like.

Rendered **as the model sees them**: after laterality mirroring and the slice
subsample. Look for a blank series; slices running the wrong way through the
knee; one study's sagittals mirrored against the others; `side?` on every row,
meaning laterality was unknown so no mirroring was applied; a plane label that
does not match the picture.
""")

    nb.code("""
FAILURE_DIR = "/kaggle/working/failures"

sheets = knee_inspect.render_failures(gold_index, shards, truth, preds,
                                      FAILURE_DIR, limit=8, best=2)
print(f"wrote {len(sheets)} sheets to {FAILURE_DIR}")

from IPython.display import Image, display
for path in sheets:
    print(os.path.basename(path))
    display(Image(filename=path))
""")

    nb.md("""
## 10. Take them off the VM

`/kaggle/working` dies with the runtime. These are small.
""")

    nb.code("""
import shutil

try:
    from google.colab import drive
    drive.mount("/content/drive")
    OUT = "/content/drive/MyDrive/knee_runs/error_analysis"
    os.makedirs(OUT, exist_ok=True)
    for path in sheets + ["/kaggle/working/gold_predictions.parquet"]:
        shutil.copy(path, OUT)
    print("copied to", OUT)
except ImportError:
    print("not on Colab; the files are in /kaggle/working and will be saved "
          "with the notebook version")
""")

    nb.write(NOTEBOOKS / "11_error_analysis.ipynb")


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
    build_finetune_notebook(colab=True)
    build_export_notebook()
    build_ensemble_notebook()
    build_ensemble_notebook(colab=True)
    build_error_analysis_notebook()


if __name__ == "__main__":
    main()
