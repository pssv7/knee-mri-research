"""Resumable Kaggle downloads and Drive-backed resolution training in Colab."""

import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
import uuid
import zipfile

from tools.colab_protocol_runtime import relocate, run_process
from tools.colab_slice_runtime import relocate_inputs


def digest(path):
    path = Path(path)
    if path.stat().st_size < 256 * 2**20:
        with path.open('rb') as handle:
            return hashlib.file_digest(handle, 'sha256').hexdigest()
    # Verifying the six 384 arrays reads ~80 GB from Drive; show it.
    return _hash_with_progress(path, f'hash {path.name}')


def output_urls(client, kernel, version='1'):
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    owner, slug = kernel.split('/')
    found, token = {}, ''
    while True:
        request = ApiListKernelSessionOutputRequest()
        request.user_name, request.kernel_slug = owner, slug
        request.version_label, request.page_size, request.page_token = str(version), 100, token
        response = client.kernels.kernels_api_client.list_kernel_session_output(request)
        for item in response.files or []:
            name = Path(item.file_name).name
            if name in found:
                raise RuntimeError(f'Ambiguous output filename: {name}')
            found[name] = item.url
        token = response.next_page_token
        if not token:
            return found


def download_stream(url_factory, destination, expected_hash=None, attempts=8,
                    request_get=None):
    """Resume HTTP ranges, renew signed URLs, and hash before exposing a file."""
    import requests
    request_get = requests.get if request_get is None else request_get
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and expected_hash and digest(destination) == expected_hash:
        print(f'Verified existing {destination.name}', flush=True)
        return destination
    partial = destination.with_name(destination.name + '.partial')
    for attempt in range(attempts):
        offset = partial.stat().st_size if partial.exists() else 0
        # A disconnect can occur after the last byte but before the response closes.
        if expected_hash and offset and digest(partial) == expected_hash:
            partial.replace(destination)
            return destination
        headers = {'Accept-Encoding': 'identity'}
        if offset:
            headers['Range'] = f'bytes={offset}-'
        try:
            with request_get(url_factory(), headers=headers, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                if response.status_code == 206:
                    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
                    if not match or int(match[1]) != offset:
                        raise RuntimeError('Server returned an incorrect byte range')
                    total = int(match[3])
                    mode = 'ab' if offset else 'wb'
                elif response.status_code == 200:
                    # Some servers ignore Range: restart, never append the full file.
                    offset, mode = 0, 'wb'
                    total = int(response.headers.get('Content-Length', 0))
                else:
                    raise RuntimeError('Unexpected download response')
                if total and shutil.disk_usage(destination.parent).free < total - offset + 2 * 2**30:
                    raise RuntimeError('Insufficient disk space for this download plus reserve')
                print(f'{destination.name}: starting at {offset / 2**30:.2f} GiB', flush=True)
                reported = offset
                with partial.open(mode) as handle:
                    for block in response.iter_content(chunk_size=8 * 2**20):
                        if block:
                            handle.write(block)
                            offset += len(block)
                        if offset - reported >= 2**30:
                            print(f'  {destination.name}: {offset / 2**30:.1f} GiB', flush=True)
                            reported = offset
                if total and partial.stat().st_size != total:
                    raise requests.exceptions.ChunkedEncodingError('incomplete file')
            if expected_hash and digest(partial) != expected_hash:
                # Task-owned incomplete file is replaced on the next attempt.
                with partial.open('wb'):
                    pass
                raise requests.exceptions.ChunkedEncodingError('checksum mismatch')
            partial.replace(destination)
            return destination
        except requests.RequestException as exc:
            # Never print signed download URLs or authorization material.
            status = getattr(getattr(exc, 'response', None), 'status_code', None)
            if status in (401, 403):
                raise RuntimeError(
                    f'Kaggle denied {destination.name} (HTTP {status}). '
                    'Re-run the authentication cell with the account that owns the private cache shards.') from None
            print(f'{destination.name}: {type(exc).__name__}'
                  + (f' HTTP {status}' if status else '')
                  + f'; retry {attempt + 1}/{attempts}', flush=True)
            if attempt + 1 == attempts:
                raise RuntimeError(f'Download interrupted: {destination.name}. Rerun this cell to resume.') from None
            time.sleep(min(2 ** attempt, 15))
    raise RuntimeError('download attempts exhausted')


def fetch_output(api, kernel, version, name, directory, expected_hash=None):
    # Kaggle output links work unauthenticated in many environments, but Colab
    # can reject them without the authenticated SDK session's headers. Keep the
    # session alive for the full transfer, including every resumed range.
    with api.build_kaggle_client() as client:
        def url():
            files = output_urls(client, kernel, version)
            if name not in files:
                raise RuntimeError(f'{kernel} version {version} has no {name}. Wait for that cache job to complete, then rerun.')
            return files[name]
        return download_stream(url, Path(directory) / name, expected_hash,
                               request_get=client.http_client()._session.get)


def unpack_metadata(folder, name):
    """Extract only the named small metadata member; never unpack arbitrary paths."""
    folder = Path(folder)
    target = folder / name
    if target.is_file():
        return target
    archives = list(folder.glob('*.zip'))
    for archive in archives:
        with zipfile.ZipFile(archive) as bundle:
            matches = [n for n in bundle.namelist() if Path(n).name == name]
            if len(matches) == 1:
                with bundle.open(matches[0]) as source, target.open('wb') as output:
                    shutil.copyfileobj(source, output)
                return target
    raise RuntimeError(f'Missing downloaded metadata: {name}')


def download_to_drive(api, drive_cache, hashes, versions):
    import numpy as np
    import pandas as pd
    from src.data.resolution import validate_cache_inputs
    if len(versions) != 6:
        raise ValueError('Exactly six cache versions are required')
    root = Path(drive_cache)
    root.mkdir(parents=True, exist_ok=True)
    receipts = []
    # Fetch ALL small receipts first; a pending shard stops before image transfers.
    for shard, version in enumerate(versions):
        kernel = f'pssv27/knee-cache-384-shard-{shard}'
        directory = root / 'cache' / f'shard{shard}'
        receipt_path = fetch_output(api, kernel, version, f'resolution_receipt_{shard:02d}.json', directory)
        receipt = json.loads(receipt_path.read_text())
        if (receipt['state'] != 'complete' or receipt['size'] != 384 or receipt['depth'] != 24
                or receipt['shard'] != shard or receipt['shards'] != 6
                or receipt['source'] != 'raw_dicom' or receipt['mirrored']):
            raise RuntimeError(f'Shard {shard} is incomplete or incompatible')
        receipts.append((kernel, version, directory, receipt))
    required = 0
    for _, _, directory, receipt in receipts:
        path = directory / f"cache_train_series_{receipt['shard']:02d}.npy"
        total = int(np.prod(receipt['shape'])) + 4096
        existing = path.stat().st_size if path.exists() else 0
        partial = path.with_name(path.name + '.partial')
        if not existing and partial.exists():
            existing = partial.stat().st_size
        required += max(0, total - existing)
    free = shutil.disk_usage(root).free
    print(f'Additional Drive cache space: {required / 2**30:.1f} GiB; reported free: {free / 2**30:.1f} GiB', flush=True)
    if free < required + 8 * 2**30:
        raise RuntimeError('Need Drive space for all six shards plus 8 GiB reserve.')
    cache_paths = []
    for kernel, version, directory, receipt in receipts:
        shard = receipt['shard']
        fetch_output(api, kernel, version, f'manifest_train_series_{shard:02d}.parquet', directory, receipt['manifest_sha256'])
        path = fetch_output(api, kernel, version, f'cache_train_series_{shard:02d}.npy', directory, receipt['volume_sha256'])
        cache_paths.append(path)
    validate_cache_inputs(cache_paths)
    meta, auxiliary = root / 'metadata', root / 'auxiliary'
    meta.mkdir(exist_ok=True)
    auxiliary.mkdir(exist_ok=True)
    for name in ('train.csv', 'train_series.csv', 'train_labels.parquet'):
        folder = auxiliary if name.endswith('.parquet') else meta
        path = folder / name
        if not path.exists() or digest(path) != hashes[name]:
            with tempfile.TemporaryDirectory(dir=folder, prefix='metadata_download_') as temporary:
                if name.endswith('.parquet'):
                    api.dataset_download_file('pssv27/knee-labels', name, path=temporary, force=True, quiet=False)
                else:
                    api.competition_download_file('rsna-knee-abnormality-detection', name, path=temporary, force=True, quiet=False)
                downloaded = unpack_metadata(temporary, name)
                if digest(downloaded) != hashes[name]:
                    raise RuntimeError(f'{name} differs from the pinned experiment metadata')
                downloaded.replace(path)
        if digest(path) != hashes[name]:
            raise RuntimeError(f'{name} differs from the pinned experiment metadata')
    side = fetch_output(api, 'pssv27/knee-side-classifier', 'latest', 'side_predictions.parquet', auxiliary)
    manifests = pd.concat([pd.read_parquet(p.with_name(p.name.replace('cache_', 'manifest_', 1).replace('.npy', '.parquet')))
                           for p in cache_paths], ignore_index=True)
    official = pd.read_csv(meta / 'train_series.csv')
    if (len(manifests) != 24371 or not manifests.SeriesInstanceUID.is_unique
            or set(zip(manifests.StudyInstanceUID, manifests.SeriesInstanceUID)) != set(zip(official.StudyInstanceUID, official.SeriesInstanceUID))):
        raise RuntimeError('Cache does not cover the official series exactly')
    sides = pd.read_parquet(side)
    if not sides.SeriesInstanceUID.is_unique or not set(official.SeriesInstanceUID) <= set(sides.SeriesInstanceUID):
        raise RuntimeError('Side predictions are incomplete or duplicated')
    record = {'state': 'verified', 'cache_versions': list(versions), 'metadata_sha256': hashes,
              'side_predictions_sha256': digest(side), 'shards': [r[3] for r in receipts]}
    (root / 'input_receipt.json').write_text(json.dumps(record, indent=2) + '\n')
    print('All six 384-pixel shards verified on Drive; ready to copy locally.', flush=True)
    return record


# KaggleHub is the supported Colab integration. Redefine the staging entrypoint
# below the original low-level implementation so older generated notebooks stay
# runnable while new notebooks use the stable KaggleHub resolver.
def _hub_file(handle, name, directory, expected_hash=None, kind="notebook"):
    import kagglehub

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / name
    if target.is_file() and (expected_hash is None or digest(target) == expected_hash):
        print(f"Reusing verified Drive {name}", flush=True)
        return target
    if target.exists():
        print(f"Replacing unverified Drive {name} (interrupted or stale download)", flush=True)
    # Reaching here means the file is absent or failed our checksum. Without
    # force_download KaggleHub refuses to overwrite a file it did not mark
    # complete (FileExistsError), and could return a marked-but-wrong file.
    options = dict(path=name, output_dir=str(directory), force_download=True)
    try:
        if kind == "notebook":
            result = kagglehub.notebook_output_download(handle, **options)
        elif kind == "dataset":
            result = kagglehub.dataset_download(handle, **options)
        else:
            result = kagglehub.competition_download(handle, **options)
    except Exception as exc:
        raise RuntimeError(
            f"KaggleHub could not download {name}: {type(exc).__name__}: {exc}. "
            "If this is an authorization error, run kagglehub.login() using the "
            "account that owns the private cache shards.") from None
    result = Path(result)
    matches = ([result] if result.is_file() and result.name == name
               else list(directory.rglob(name)))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {name} after KaggleHub download, found {matches}")
    if expected_hash is not None and digest(matches[0]) != expected_hash:
        raise RuntimeError(f"Downloaded {name} does not match the pinned checksum")
    return matches[0]


def download_to_drive(drive_cache, hashes, versions):
    """KaggleHub download to Drive, then strict cache/metadata verification."""
    import numpy as np
    import pandas as pd
    from src.data.resolution import validate_cache_inputs

    if len(versions) != 6:
        raise ValueError("Exactly six cache versions are required")
    root = Path(drive_cache)
    root.mkdir(parents=True, exist_ok=True)
    receipts = []
    for shard, version in enumerate(versions):
        kernel = f"pssv27/knee-cache-384-shard-{shard}/versions/{version}"
        directory = root / "cache" / f"shard{shard}"
        receipt_path = _hub_file(kernel, f"resolution_receipt_{shard:02d}.json", directory)
        receipt = json.loads(receipt_path.read_text())
        if (receipt["state"] != "complete" or receipt["size"] != 384
                or receipt["depth"] != 24 or receipt["shard"] != shard
                or receipt["shards"] != 6 or receipt["source"] != "raw_dicom"
                or receipt["mirrored"]):
            raise RuntimeError(f"Shard {shard} is incomplete or incompatible")
        receipts.append((kernel, directory, receipt))
    required = sum(int(np.prod(receipt["shape"])) + 4096
                   for _, _, receipt in receipts)
    if shutil.disk_usage(root).free < required + 8 * 2**30:
        raise RuntimeError("Need Drive space for all six shards plus 8 GiB reserve")
    paths = []
    for kernel, directory, receipt in receipts:
        shard = receipt["shard"]
        _hub_file(kernel, f"manifest_train_series_{shard:02d}.parquet", directory,
                  receipt["manifest_sha256"])
        paths.append(_hub_file(kernel, f"cache_train_series_{shard:02d}.npy", directory,
                               receipt["volume_sha256"]))
    validate_cache_inputs(paths)
    meta, auxiliary = root / "metadata", root / "auxiliary"
    for name in ("train.csv", "train_series.csv"):
        _hub_file("rsna-knee-abnormality-detection", name, meta, hashes[name], "competition")
    _hub_file("pssv27/knee-labels", "train_labels.parquet", auxiliary,
              hashes["train_labels.parquet"], "dataset")
    # Use the current published side-classifier output.  The old v1 pin was
    # the stale checkpoint and can silently produce a different laterality
    # table; the unversioned handle resolves to the latest COMPLETE version.
    side = _hub_file("pssv27/knee-side-classifier",
                     "side_predictions.parquet", auxiliary)
    manifest = pd.concat([pd.read_parquet(path.with_name(
        path.name.replace("cache_", "manifest_", 1).replace(".npy", ".parquet")))
                          for path in paths], ignore_index=True)
    official = pd.read_csv(meta / "train_series.csv")
    if (len(manifest) != 24371 or not manifest.SeriesInstanceUID.is_unique
            or set(zip(manifest.StudyInstanceUID, manifest.SeriesInstanceUID))
            != set(zip(official.StudyInstanceUID, official.SeriesInstanceUID))):
        raise RuntimeError("Cache does not cover official series exactly")
    sides = pd.read_parquet(side)
    if not sides.SeriesInstanceUID.is_unique or not set(official.SeriesInstanceUID) <= set(sides.SeriesInstanceUID):
        raise RuntimeError("Side predictions are incomplete or duplicated")
    record = {"state": "verified", "cache_versions": list(versions),
              "metadata_sha256": hashes, "side_predictions_sha256": digest(side),
              "shards": [receipt for _, _, receipt in receipts]}
    (root / "input_receipt.json").write_text(json.dumps(record, indent=2) + "\n")
    print("All six 384-pixel shards verified on Drive; ready to copy locally.", flush=True)
    return record


def _hash_with_progress(path, label):
    from tqdm.auto import tqdm
    checksum, path = hashlib.sha256(), Path(path)
    with path.open('rb') as handle, tqdm(total=path.stat().st_size, desc=label, unit='B',
                                         unit_scale=True, unit_divisor=1024, mininterval=1.0) as bar:
        for block in iter(lambda: handle.read(8 * 2**20), b''):
            checksum.update(block)
            bar.update(len(block))
    return checksum.hexdigest()


def copy_from_drive(source, destination, expected_hash):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == source.stat().st_size \
            and _hash_with_progress(destination, f'hash local {destination.name}') == expected_hash:
        print(f'Reusing verified local {destination.name}', flush=True)
        return
    if shutil.disk_usage(destination.parent).free < source.stat().st_size + 2 * 2**30:
        raise RuntimeError(f'Insufficient local disk for {source.name}')
    partial = destination.with_name(destination.name + '.partial')
    from tqdm.auto import tqdm
    checksum, copied = hashlib.sha256(), 0
    with source.open('rb') as incoming, partial.open('wb') as outgoing, \
            tqdm(total=source.stat().st_size, desc=f'copy {source.name}', unit='B',
                 unit_scale=True, unit_divisor=1024, mininterval=1.0) as bar:
        for block in iter(lambda: incoming.read(8 * 2**20), b''):
            outgoing.write(block)
            checksum.update(block)
            copied += len(block)
            bar.update(len(block))
    if copied != source.stat().st_size or checksum.hexdigest() != expected_hash:
        raise RuntimeError(f'Drive copy failed checksum: {source.name}; rerun the Drive download/verification cell')
    partial.replace(destination)


def copy_inputs_to_colab(drive_cache, local_root):
    from src.data.resolution import validate_cache_inputs
    root, local = Path(drive_cache), Path(local_root)
    if str(local.resolve()).startswith('/content/drive') or local.resolve() == root.resolve():
        raise ValueError('LOCAL_INPUTS must be on Colab VM disk, separate from Drive')
    record = json.loads((root / 'input_receipt.json').read_text())
    if record['state'] != 'verified' or len(record['shards']) != 6:
        raise RuntimeError('Complete Drive download/verification first')
    files = {}
    for receipt in record['shards']:
        shard = receipt['shard']
        prefix = f'cache/shard{shard}'
        files[f'{prefix}/cache_train_series_{shard:02d}.npy'] = receipt['volume_sha256']
        files[f'{prefix}/manifest_train_series_{shard:02d}.parquet'] = receipt['manifest_sha256']
        name = f'{prefix}/resolution_receipt_{shard:02d}.json'
        files[name] = digest(root / record.get('files', {}).get(name, name))
    for name, checksum in record['metadata_sha256'].items():
        prefix = 'auxiliary' if name.endswith('.parquet') else 'metadata'
        files[f'{prefix}/{name}'] = checksum
    files['auxiliary/side_predictions.parquet'] = record['side_predictions_sha256']
    local.mkdir(parents=True, exist_ok=True)
    needed = sum((root / record.get('files', {}).get(name, name)).stat().st_size
                 for name in files if not (local / name).exists())
    if shutil.disk_usage(local).free < needed + 8 * 2**30:
        raise RuntimeError(f'Need {needed / 2**30:.1f} GiB additional local storage plus 8 GiB reserve. Select a larger-disk runtime.')
    for name, checksum in files.items():
        source = root / record.get('files', {}).get(name, name)
        copy_from_drive(source, local / name, checksum)
    validate_cache_inputs(sorted((local / 'cache').rglob('cache_train_series_*.npy')))
    local_record = {**record, 'files': {name: name for name in files}} if 'files' in record else record
    (local / 'input_receipt.json').write_text(json.dumps(local_record, indent=2) + '\n')
    print('Drive → Colab copy verified. Training will read only VM-local images.', flush=True)
    return record


def verify_drive_inputs(drive_cache):
    """Verify a completed Drive cache without contacting Kaggle."""
    from src.data.resolution import validate_cache_inputs

    root = Path(drive_cache)
    record_path = root / "input_receipt.json"
    if not record_path.is_file():
        raise RuntimeError("Missing input_receipt.json in the existing Drive cache")
    record = json.loads(record_path.read_text())
    if 'files' in record:
        return adopt_drive_inputs(root, record['metadata_sha256'])
    if record.get("state") != "verified" or len(record.get("shards", [])) != 6:
        raise RuntimeError("Drive cache receipt is incomplete")
    paths = []
    for receipt in record["shards"]:
        shard = receipt["shard"]
        folder = root / "cache" / f"shard{shard}"
        array = folder / f"cache_train_series_{shard:02d}.npy"
        manifest = folder / f"manifest_train_series_{shard:02d}.parquet"
        if (not array.is_file() or not manifest.is_file()
                or digest(array) != receipt["volume_sha256"]
                or digest(manifest) != receipt["manifest_sha256"]):
            raise RuntimeError(f"Drive cache shard {shard} differs from its receipt")
        paths.append(array)
    validate_cache_inputs(paths)
    for name, checksum in record["metadata_sha256"].items():
        folder = root / ("auxiliary" if name.endswith(".parquet") else "metadata")
        if not (folder / name).is_file() or digest(folder / name) != checksum:
            raise RuntimeError(f"Drive metadata differs from its receipt: {name}")
    side = root / "auxiliary" / "side_predictions.parquet"
    if not side.is_file() or digest(side) != record["side_predictions_sha256"]:
        raise RuntimeError("Drive side predictions differ from its receipt")
    print("Existing Drive cache verified; no Kaggle download will run.", flush=True)
    return record


def adopt_drive_inputs(drive_cache, hashes):
    """Verify original shard receipts; never invent raw-data provenance or Drive links."""
    root = Path(drive_cache)
    import pandas as pd
    from src.data.resolution import validate_cache_inputs
    files, records, arrays, frames = {}, [], [], []

    def unique(name, checksum=None):
        hits = [p for p in root.rglob(name) if p.is_file()]
        if checksum:
            hits = [p for p in hits if digest(p) == checksum]
        if len(hits) != 1:
            raise RuntimeError(f"Expected exactly one original {name}; found {hits}")
        return hits[0]

    for shard in range(6):
        receipt_path = unique(f"resolution_receipt_{shard:02d}.json")
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("plan_sha256") == "adopted-existing-drive-cache":
            raise RuntimeError("Synthetic adoption receipt is not source provenance; restore the original Kaggle receipt")
        array = unique(f"cache_train_series_{shard:02d}.npy")
        manifest = unique(f"manifest_train_series_{shard:02d}.parquet")
        if array.parent != receipt_path.parent or manifest.parent != receipt_path.parent:
            raise RuntimeError("Keep each original receipt, manifest and array together")
        if digest(array) != receipt["volume_sha256"] or digest(manifest) != receipt["manifest_sha256"]:
            raise RuntimeError(f"Original shard {shard} checksum mismatch")
        for path in (array, manifest, receipt_path):
            files[f"cache/shard{shard}/{path.name}"] = str(path.relative_to(root))
        arrays.append(array)
        frames.append(pd.read_parquet(manifest))
        records.append(receipt)
    validate_cache_inputs(arrays)
    for name, checksum in hashes.items():
        folder = "auxiliary" if name.endswith(".parquet") else "metadata"
        files[f"{folder}/{name}"] = str(unique(name, checksum).relative_to(root))
    side = unique("side_predictions.parquet")
    files["auxiliary/side_predictions.parquet"] = str(side.relative_to(root))
    official = pd.read_csv(root / files["metadata/train_series.csv"])
    combined = pd.concat(frames, ignore_index=True)
    if (len(combined) != 24371 or not combined.SeriesInstanceUID.is_unique
            or set(zip(combined.StudyInstanceUID, combined.SeriesInstanceUID))
            != set(zip(official.StudyInstanceUID, official.SeriesInstanceUID))):
        raise RuntimeError("Cache must cover exactly the 24,371 official series")
    sides = pd.read_parquet(side)
    if not sides.SeriesInstanceUID.is_unique or not set(official.SeriesInstanceUID) <= set(sides.SeriesInstanceUID):
        raise RuntimeError("Side predictions are incomplete or duplicated")
    record = {"state": "verified", "cache_versions": ["original_receipts"] * 6,
              "metadata_sha256": hashes, "side_predictions_sha256": digest(side),
              "shards": records, "files": files}
    (root / "input_receipt.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def run_training(sources, drive_root, local_root, fold=1, size=384):
    import torch
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Select exactly one Colab GPU')
    root, local = Path(drive_root), Path(local_root)
    mount = Path('/content/drive/MyDrive')
    if not mount.is_dir() or not root.resolve().is_relative_to(mount.resolve()):
        raise RuntimeError('Mount Drive and place DRIVE_OUT inside MyDrive')
    inputs = json.loads((local / 'input_receipt.json').read_text())
    if inputs['state'] != 'verified':
        raise RuntimeError('Run input verification before training')
    attempt = root / f'attempt_{uuid.uuid4().hex[:12]}'
    attempt.mkdir(parents=True)
    shutil.copy2(local / 'input_receipt.json', attempt / 'input_receipt.json')
    progress = {'state': 'running', 'fold': fold, 'gpu': torch.cuda.get_device_name(),
                'torch': torch.__version__, 'single_gpu': True, 'effective_batch': 8,
                'note': 'Single-GPU run is not bitwise equivalent to Kaggle DDP.'}
    try:
        for arm in ('smoke', 'full'):
            arm_root = attempt / arm
            directory = arm_root / f'resolution{size}_fold{fold}'
            directory.mkdir(parents=True)
            source = relocate_inputs(relocate(sources[arm], arm_root), local)
            script = directory / 'train.py'
            script.write_text(source)
            progress['active'] = arm
            (attempt / 'progress.json').write_text(json.dumps(progress, indent=2))
            receipt = {'state': 'running', 'size': size, 'fold': fold, 'smoke': arm == 'smoke',
                       'script_sha256': hashlib.sha256(source.encode()).hexdigest()}
            started = time.monotonic()
            try:
                print(f'{arm.upper()}: checkpoints and logs in {directory}', flush=True)
                run_process(script, directory)
                if arm == 'smoke':
                    measured = json.loads((directory / 'training_measurement.json').read_text())
                    if not measured['training_memory_passed']:
                        raise RuntimeError('Resolution smoke failed the training memory gate')
                receipt['state'] = 'complete'
            except BaseException as exc:
                receipt.update(state='failed', error=type(exc).__name__)
                raise
            finally:
                receipt['elapsed_seconds'] = time.monotonic() - started
                (arm_root / 'resolution_run.json').write_text(json.dumps(receipt, indent=2) + '\n')
        required = [f'convnext_fold{fold}.pt', f'convnext_fold{fold}.pt.json',
                    f'oof_fold{fold}.parquet', f'gold_predictions_fold{fold}.parquet',
                    'run_contract.json', 'folds.parquet']
        progress.update(state='complete', active=None,
                        output=str(directory), sha256={name: digest(directory / name) for name in required})
        print('TRAINING COMPLETE:', directory, flush=True)
    except BaseException as exc:
        progress.update(state='failed', error=type(exc).__name__)
        raise
    finally:
        (attempt / 'progress.json').write_text(json.dumps(progress, indent=2) + '\n')
    return str(directory)


def run_resolution_folds(sources_by_fold, drive_root, local_root, folds):
    """Run screening folds sequentially, preserving every Drive attempt."""
    folds = list(folds)
    if not folds or len(set(folds)) != len(folds) or not set(folds) <= {1, 3}:
        raise ValueError("Choose unique screening folds from 1 and 3")
    results = {}
    for fold in folds:
        if set(sources_by_fold.get(fold, {})) != {"smoke", "full"}:
            raise ValueError(f"Missing smoke/full sources for fold {fold}")
        results[fold] = run_training(sources_by_fold[fold],
                                     Path(drive_root) / f"fold{fold}", local_root, fold)
    return results
