"""Bounded, versioned raw-DICOM cache construction for resolution experiments."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

from src import config
from src.data import cache, dicom_io


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def shard_plan(series, size=384, shards=6, budget_bytes=16 * 2**30):
    """Balance by study, deterministic under input reordering; enforce disk cap."""
    if size < 1 or shards < 1 or series.empty:
        raise ValueError('positive geometry, shard count and nonempty series required')
    if series.SeriesInstanceUID.duplicated().any():
        raise ValueError('duplicate series')
    counts = series.groupby(config.ID_COL).size()
    order = sorted(counts.index, key=lambda uid: (-counts[uid], hashlib.sha256(str(uid).encode()).hexdigest()))
    loads = [0] * shards
    assignment = {}
    for uid in order:
        chosen = min(range(shards), key=lambda i: (loads[i], i))
        loads[chosen] += int(counts[uid])
        assignment[uid] = chosen
    payloads = [count * config.CACHE_SLICES * size * size + 4096 for count in loads]
    if any(n > budget_bytes for n in payloads) or not all(loads):
        raise ValueError('shard payload exceeds budget or a shard is empty')
    result = series[[config.ID_COL, 'SeriesInstanceUID']].copy()
    result['shard'] = result[config.ID_COL].map(assignment)
    return result.sort_values('SeriesInstanceUID').reset_index(drop=True)


def audit_index(index, series):
    """Full header-index geometry inventory, with strict source coverage."""
    pairs = index[[config.ID_COL, 'SeriesInstanceUID']].drop_duplicates()
    expected = series[[config.ID_COL, 'SeriesInstanceUID']]
    if set(map(tuple, pairs.to_numpy())) != set(map(tuple, expected.to_numpy())):
        raise ValueError('header index does not cover exactly the official series')
    if index.path.duplicated().any():
        raise ValueError('duplicate indexed DICOM path')
    dims = index[['Rows', 'Columns']].apply(pd.to_numeric, errors='coerce')
    if dims.isna().any().any() or (dims <= 0).any().any():
        raise ValueError('missing or invalid original image dimensions')
    summary = dims.value_counts().rename('slices').reset_index()
    per_series = dims.assign(SeriesInstanceUID=index.SeriesInstanceUID).groupby('SeriesInstanceUID').min()
    return {'studies': int(pairs[config.ID_COL].nunique()), 'series': len(pairs),
            'slices': len(index), 'geometry': summary.to_dict('records'),
            'series_both_dimensions_above_224': int((per_series > 224).all(axis=1).sum()),
            'series_both_dimensions_at_least_384': int((per_series >= 384).all(axis=1).sum())}


def relocate_headers(rows, raw_root):
    result = []
    for row in rows:
        item = dict(row)
        path = Path(raw_root) / str(item[config.ID_COL]) / str(item['SeriesInstanceUID']) / Path(item['path']).name
        if not path.is_file():
            raise FileNotFoundError(path)
        item['path'] = str(path)
        result.append(item)
    return result


def validate_cache_inputs(paths, expected_size=384, expected_shards=6):
    """Require complete, compatible shards with bound manifests and disjoint IDs."""
    if len(paths) != expected_shards:
        raise ValueError('incomplete resolution cache')
    seen, plans, uids = set(), set(), set()
    for path in map(Path, paths):
        suffix = path.stem.removeprefix('cache_train_series_')
        receipt = json.loads((path.parent / f'resolution_receipt_{suffix}.json').read_text())
        if (receipt['state'] != 'complete' or receipt['size'] != expected_size
                or receipt['depth'] != 24 or receipt['shards'] != expected_shards
                or receipt['source'] != 'raw_dicom' or receipt['mirrored']
                or receipt['dtype'] != 'uint8'):
            raise ValueError('incomplete or incompatible resolution cache receipt')
        if receipt['shard'] in seen:
            raise ValueError('duplicate resolution shard')
        seen.add(receipt['shard'])
        plans.add(receipt['plan_sha256'])
        manifest_path = path.parent / f'manifest_train_series_{suffix}.parquet'
        if sha256(manifest_path) != receipt['manifest_sha256']:
            raise ValueError('cache manifest hash mismatch')
        volume = np.load(path, mmap_mode='r')
        if list(volume.shape) != receipt['shape'] or volume.dtype != np.uint8:
            raise ValueError('cache array geometry mismatch')
        if volume.shape[1:] != (24, expected_size, expected_size):
            raise ValueError('cache dimensions mismatch')
        frame = pd.read_parquet(manifest_path)
        if (len(frame) != len(volume) or receipt['completed'] != len(frame)
                or frame.error.notna().any() or frame.position.tolist() != list(range(len(frame)))):
            raise ValueError('cache rows incomplete or misaligned')
        ids = set(frame.SeriesInstanceUID)
        if len(ids) != len(frame) or ids & uids:
            raise ValueError('duplicate cache series')
        uids.update(ids)
    if seen != set(range(expected_shards)) or len(plans) != 1:
        raise ValueError('shard plan mismatch')


def build_shard(index, series, raw_root, output, shard, size=384, shards=6,
                workers=2, budget_bytes=16 * 2**30):
    """Stream a shard to memmap; failures leave a non-complete receipt."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not 0 <= shard < shards:
        raise ValueError('invalid shard')
    plan = shard_plan(series, size, shards, budget_bytes)
    audit = audit_index(index, series)
    selected = plan[plan.shard == shard]
    shape = (len(selected), config.CACHE_SLICES, size, size)
    required = int(np.prod(shape)) + 4096
    if shutil.disk_usage(output).free < required + 2 * 2**30:
        raise ValueError('insufficient disk space including 2 GiB reserve')
    volume_path = output / f'cache_train_series_{shard:02d}.npy'
    manifest_path = output / f'manifest_train_series_{shard:02d}.parquet'
    receipt_path = output / f'resolution_receipt_{shard:02d}.json'
    if any(p.exists() for p in (volume_path, manifest_path, receipt_path)):
        raise FileExistsError('refusing to overwrite a previous cache attempt')
    plan.to_parquet(output / 'resolution_plan.parquet', index=False)
    (output / 'resolution_geometry_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    receipt = {'state': 'building', 'shard': shard, 'shards': shards, 'size': size,
               'depth': config.CACHE_SLICES, 'dtype': 'uint8', 'mirrored': config.MIRROR_IN_CACHE,
               'shape': shape, 'source': 'raw_dicom', 'completed': 0,
               'plan_sha256': sha256(output / 'resolution_plan.parquet')}
    started = time.monotonic()
    def save():
        receipt['elapsed_seconds'] = time.monotonic() - started
        receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    save()
    packed = np.lib.format.open_memmap(volume_path, mode='w+', dtype=np.uint8, shape=shape)
    grouped = {uid: frame for uid, frame in index[index.SeriesInstanceUID.isin(selected.SeriesInstanceUID)].groupby('SeriesInstanceUID')}
    def build(uid):
        headers = relocate_headers(grouped[uid].to_dict('records'), raw_root)
        result = cache.build_series_volume(headers, size=size)
        return uid, headers, result
    records = []
    try:
        # Bounded batches prevent the executor queuing several GiB of completed volumes.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            uids = selected.SeriesInstanceUID.tolist()
            for start in range(0, len(uids), workers):
                for uid, headers, result in pool.map(build, uids[start:start + workers]):
                    position = len(records)
                    volume = result.pop('volume')
                    if volume.shape != shape[1:] or volume.dtype != np.uint8:
                        raise ValueError('unexpected built geometry')
                    packed[position] = volume
                    result.update(position=position, error=None)
                    records.append(result)
                    # Verify the serialized volume through the same raw inference entrypoint.
                    if position < 2:
                        again = cache.build_series_volume(headers, size=size)['volume']
                        np.testing.assert_array_equal(packed[position], again)
                    receipt['completed'] = len(records)
                if start % (workers * 50) == 0:
                    packed.flush()
                    save()
                    print(f"shard {shard}: {len(records)}/{len(uids)} series", flush=True)
        packed.flush()
        manifest = pd.DataFrame(records)
        sides = {study: dict(zip(rows.SeriesInstanceUID, rows.laterality))
                 for study, rows in manifest.groupby(config.ID_COL)}
        resolved = dicom_io.resolve_study_laterality(sides)
        lookup = {uid: side for study in resolved.values() for uid, side in study.items()}
        manifest['laterality_resolved'] = manifest.SeriesInstanceUID.map(lookup)
        manifest.to_parquet(manifest_path, index=False)
        receipt.update(state='complete', volume_sha256=sha256(volume_path),
                       manifest_sha256=sha256(manifest_path), raw_reproduction_checks=min(2, len(records)))
    except BaseException as exc:
        receipt.update(state='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save()
    print(json.dumps(receipt, indent=2), flush=True)
    return receipt
