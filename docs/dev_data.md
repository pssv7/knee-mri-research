# Local development data

The full competition dataset is ~570 GB and never touches this machine. For
local work — verifying DICOM parsing, slice ordering, laterality, and the cache
pipeline — a single real series is enough.

## What is checked in

Nothing. `data/` is gitignored. The small metadata CSVs and the dev series are
downloaded on demand.

## Metadata CSVs (~9 MB)

```bash
cd <project root>
for f in train.csv train_series.csv test.csv test_series.csv sample_submission.csv; do
    kaggle competitions download rsna-knee-abnormality-detection -f "$f" -p data/meta -q
done
```

## One dev series (~37 MB, 30 slices)

The public test set holds 5 series across 3 studies. Any one works; the sagittal
series below is a left knee with complete geometry, which exercises both the
mirroring path and the geometric sort.

```bash
python3 - <<'EOF'
import collections, os, time
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi(); api.authenticate()
listing = api.competition_list_files("rsna-knee-abnormality-detection", page_size=200)
paths = [f.name for f in listing.files if f.name.startswith("test_series/")]

by_series = collections.defaultdict(list)
for p in paths:
    parts = p.split("/")
    if len(parts) == 4:
        by_series[parts[2]].append(p)

target = by_series["1.2.826.0.1.3680043.8.498.17811502614030631664517622518906646132"]
os.makedirs("data/dev", exist_ok=True)
for p in sorted(target):
    api.competition_download_file(
        "rsna-knee-abnormality-detection", p, path="data/dev", quiet=True)
    time.sleep(0.15)   # the API rate-limits aggressively; 429s are easy to trigger
EOF
```

Files land flat in `data/dev/`, not in the nested `test_series/study/series/`
layout — the API strips the prefix. The tests glob `data/dev/*.dcm` accordingly.

## Running the tests

```bash
python3 -m pytest tests/ -q
```

Tests under `TestRealSeries` skip automatically when `data/dev/` is empty, so the
synthetic geometry and laterality tests still run in a clean checkout.

## Known properties of the dev series

Useful as regression anchors:

| Property | Value |
|---|---|
| Series | `…22518906646132`, sagittal |
| Slices | 30, regular 3.3 mm spacing |
| Acquisition | 800×800, 0.1875 mm in-plane, Siemens 1.5 T, `t2_tse_sag_d` |
| `ImageLaterality` | **absent** — side comes from the `Laterality` tag |
| Side | `L`, so the cache pipeline mirrors it |
| `InstanceNumber` | runs **opposite** to geometric order (30 → 1) |

That last row is the reason slices are sorted by projected
`ImagePositionPatient` rather than `InstanceNumber`: on this series the naive
sort silently reverses the volume.
