# Restore the six original Kaggle 384 receipts beside their shards.
import hashlib, json
from pathlib import Path

ROWS = [
    (0, 4064, 1293.063320476,
        "35ccac10bd16d8d7ef009f554aa3a383"
        "44f945d1339b0cea0bb7dbbdf207b8c9",
        "0f4c7c8173d4c806bd1b7c7cd44c6315"
        "e20de3bee4b8a40dc6d30b25d7709fa9"),
    (1, 4064, 1601.7194189820002,
        "cad897ec5bee934a5941ee65dba63e70"
        "c84128fe786e4ec0e83fdd8a3d494710",
        "9400e89ef330d21f83ab3c721aa39e48"
        "40c80d1fed91354e7e33e0aa8d49ae72"),
    (2, 4063, 1588.316613795,
        "a5f6e11685226e605f9c71c03eec673f"
        "4a403755d7f9d4762e4e9c89845f28af",
        "e6244919e2c94bd302c294a51d43bd61"
        "074d2b8ffbc087b02e48c5926581b791"),
    (3, 4060, 1213.978375951,
        "02c890384f838fc3fc12da1473e0bf40"
        "5f6b9081e67a9222b87d100993049755",
        "73bf980d421684985d6e7eace6d7551f"
        "75c7caae5f14ac15dc2a6845cdd5e6e8"),
    (4, 4060, 1746.893612485,
        "14731cda128c3ad111af8ef80c4d1c16"
        "b261e0e344285232e680272d3be6acfa",
        "a8b6d73e9e79da2ac8aa88830db35758"
        "c132a102f2dd1f15ad27415e65c81baf"),
    (5, 4060, 1747.8137639689999,
        "a1c57309891e9ae6a7a0dce91324f1e6"
        "c7e29e87d618df1d92f25511f1386dea",
        "337141b8c6c400c356aeabd5f25e29ab"
        "b9b35409c97939b3ac195d59d8fc0f3e"),
]
PLAN = ("53acb1b146ccb178a172553457f91d8e"
        "49ced5efca03240d9ffe3610b94c5eb9")

def receipt(shard, n, elapsed, volume, manifest):
    return {"state": "complete", "shard": shard, "shards": 6, "size": 384,
            "depth": 24, "dtype": "uint8", "mirrored": False,
            "shape": [n, 24, 384, 384], "source": "raw_dicom", "completed": n,
            "plan_sha256": PLAN, "elapsed_seconds": elapsed,
            "volume_sha256": volume, "manifest_sha256": manifest,
            "raw_reproduction_checks": 2}

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()

root = Path(DRIVE_CACHE_384)
for row in ROWS:
    shard, rec = row[0], receipt(*row)
    name = f"resolution_receipt_{shard:02d}.json"
    arrays = list(root.rglob(f"cache_train_series_{shard:02d}.npy"))
    manifests = list(root.rglob(f"manifest_train_series_{shard:02d}.parquet"))
    assert len(arrays) == 1 and len(manifests) == 1, (shard, arrays, manifests)
    folder = arrays[0].parent
    assert manifests[0].parent == folder, f"shard {shard}: manifest not with array"
    assert sha(manifests[0]) == rec["manifest_sha256"], f"shard {shard}: manifest"
    for old in root.rglob(name):
        if json.loads(old.read_text()).get("plan_sha256") != PLAN:
            old.rename(old.with_name(name + ".synthetic_backup"))
            print("moved aside:", old)
    target = folder / name
    if target.exists():
        assert json.loads(target.read_text()) == rec, f"shard {shard}: differs"
    else:
        target.write_text(json.dumps(rec, indent=2) + "\n")
    print(f"shard {shard}: original receipt in place")
