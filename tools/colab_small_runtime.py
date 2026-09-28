"""Drive staging and single-candidate execution for the frozen Small screen."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def valid_file(path, entry):
    path = Path(path)
    return path.is_file() and path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256']


def copy_pinned(candidates, target, entry):
    """Use full content hashes, including for completed local copies."""
    from tqdm.auto import tqdm
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if valid_file(target, entry):
        print('Verified local copy:', target.name, flush=True)
        return target
    for source in sorted(set(map(Path, candidates))):
        if not source.is_file() or source.stat().st_size != entry['bytes']:
            continue
        # Space for a new temporary copy is needed even if the old file exists.
        if shutil.disk_usage(target.parent).free < entry['bytes'] + 2 * 10**9:
            raise RuntimeError(f'Insufficient local disk to stage {target.name}')
        temporary = target.with_name(target.name + '.partial')
        digest = hashlib.sha256()
        with source.open('rb') as incoming, temporary.open('wb') as outgoing, tqdm(
                total=entry['bytes'], unit='B', unit_scale=True, desc=target.name) as progress:
            for block in iter(lambda: incoming.read(8 << 20), b''):
                outgoing.write(block)
                digest.update(block)
                progress.update(len(block))
        if digest.hexdigest() != entry['sha256'] or not valid_file(temporary, entry):
            temporary.unlink()
            continue
        temporary.replace(target)
        return target
    raise RuntimeError(f'No hash-matching {target.name} found. Use the original full 24-slice '
                       '224-pixel shards and their manifests; thinned exports cannot be used.')


def stage_inputs(drive_cache, local_root, code_root, expected, side_file=''):
    """Images only from Drive; optional authenticated download of small side metadata."""
    root, local, code = Path(drive_cache), Path(local_root), Path(code_root)
    if not root.is_dir():
        raise RuntimeError('Set DRIVE_CACHE to the existing Drive shard folder')
    local.mkdir(parents=True, exist_ok=True)
    records = [expected['series_meta'], expected['side_predictions']]
    records += [entry for shard in expected['shards'] for entry in shard.values()]
    candidates = {}
    for entry in records:
        name = Path(entry['path']).name
        hits = list(root.rglob(name))
        if (code / 'metadata' / name).is_file():
            hits.append(code / 'metadata' / name)
        if name == 'side_predictions.parquet' and side_file:
            hits.insert(0, Path(side_file))
        candidates[name] = hits
    # Catch absent/wrong-sized shards before copying or requesting credentials.
    needed = 0
    for shard in expected['shards']:
        for entry in shard.values():
            name = Path(entry['path']).name
            if valid_file(local / name, entry):
                continue
            if not any(p.is_file() and p.stat().st_size == entry['bytes'] for p in candidates[name]):
                raise RuntimeError(f'Missing original full shard/manifest: {name} in {root}')
            needed += entry['bytes']
    if shutil.disk_usage(local).free < needed + 5 * 10**9:
        raise RuntimeError('Need about 35 GB free local disk for the three original shards')
    bound = json.loads(json.dumps(expected))
    bound_entries = [bound['series_meta'], bound['side_predictions']]
    bound_entries += [entry for shard in bound['shards'] for entry in shard.values()]
    for entry in bound_entries:
        name = Path(entry['path']).name
        if name == 'side_predictions.parquet' and not valid_file(local / name, entry):
            hits = [p for p in candidates[name] if valid_file(p, entry)]
            if not hits:
                import kagglehub
                from kagglehub.config import get_kaggle_credentials, set_kaggle_api_token
                if get_kaggle_credentials() is None:
                    from getpass import getpass
                    token = getpass('Kaggle API token for missing side_predictions.parquet (hidden): ').strip()
                    if not token:
                        raise RuntimeError('Set SIDE_FILE to the saved side predictions, or supply a Kaggle token')
                    set_kaggle_api_token(token)
                    del token
                folder = Path(kagglehub.notebook_output_download(
                    'pssv27/knee-side-classifier', path='side_predictions.parquet'))
                hits = [folder] if folder.is_file() else list(folder.rglob(name))
            candidates[name] = hits
        target = copy_pinned(candidates[name], local / name, entry)
        entry['path'] = str(target.resolve())
    path = local / 'inputs.json'
    path.write_text(json.dumps(bound, indent=2) + '\n')
    print('All original inputs verified; training reads VM-local arrays:', path)
    return path


def run_process(command, cwd, log_path):
    env = os.environ.copy()
    env['PYTHONHASHSEED'] = '0'
    with Path(log_path).open('w', buffering=1) as log:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
            if process.wait() != 0:
                raise RuntimeError(f'Process failed; preserved log: {log_path}')
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            process.stdout.close()


def completion_files():
    return ['candidate/selected.pt', 'candidate/run_contract.json',
            'candidate/outer_predictions.parquet', 'candidate/training/version_0/metrics.csv',
            'smoke/smoke_receipt.json', 'comparison.json', 'screen_decision.json']


def verify_completed(output, bundle_hash):
    output = Path(output)
    marker = output / 'complete.json'
    if not marker.exists():
        return False
    receipt = json.loads(marker.read_text())
    if receipt['bundle_sha256'] != bundle_hash or set(receipt['files']) != set(completion_files()):
        raise ValueError('Completion belongs to a different or incomplete experiment')
    for name, digest in receipt['files'].items():
        if sha(output / name) != digest:
            raise ValueError(f'Completed artifact changed: {name}')
    return True


def compare_candidate(code_root, inputs_path, output):
    """Explicit historical Tiny comparison; do not weaken the matched-pair evaluator."""
    import pandas as pd
    from src.validation.proxy import ID, LABELS, validate_prediction_provenance, paired_proxy_evaluation
    code, output = Path(code_root), Path(output)
    design = code / 'design/candidate'
    spec = json.loads((design / 'spec.json').read_text())
    roles = pd.read_csv(design / 'roles.csv')
    labels = pd.read_parquet(design / 'proxy_labels.parquet')
    for name, key in [('roles.csv', 'roles_sha256'), ('proxy_labels.parquet', 'proxy_labels_sha256')]:
        if sha(design / name) != spec[key]:
            raise ValueError('Frozen evaluation design changed')
    reference = code / 'reference'
    control = json.loads((reference / 'run_contract.json').read_text())
    candidate = json.loads((output / 'candidate/run_contract.json').read_text())
    for key, value in spec.items():
        if candidate.get(key) != value:
            raise ValueError(f'Candidate differs from frozen specification: {key}')
    for key in ['roles_sha256', 'proxy_labels_sha256', 'seed', 'sampling_policy',
                'raw_counts_sha256', 'protected_exclusions_sha256']:
        if candidate[key] != control[key]:
            raise ValueError(f'Historical-control mismatch: {key}')
    expected_recipe = dict(control['training_recipe'], backbone='convnext_small.fb_in22k_ft_in1k')
    if candidate['training_recipe'] != expected_recipe:
        raise ValueError('Training recipe changed beyond backbone')
    if candidate['input_config_sha256'] != sha(inputs_path):
        raise ValueError('Candidate input configuration changed')
    old_inputs = json.loads((reference / 'inputs.json').read_text())
    if sha(reference / 'inputs.json') != control['input_config_sha256']:
        raise ValueError('Historical input configuration changed')
    new_inputs = json.loads(Path(inputs_path).read_text())
    # Host paths differ; actual input identities must not.
    def hashes(config):
        entries = [config['series_meta'], config['side_predictions']]
        entries += [e for shard in config['shards'] for e in shard.values()]
        return sorted((Path(e['path']).name, e['sha256'], e['bytes']) for e in entries)
    if hashes(old_inputs) != hashes(new_inputs):
        raise ValueError('Historical-control input content mismatch')
    if sha(output / 'candidate/selected.pt') != candidate['checkpoint_sha256']:
        raise ValueError('Candidate checkpoint changed')
    smoke = json.loads((output / 'smoke/smoke_receipt.json').read_text())
    if smoke['status'] != 'passed' or smoke['initial_weights_sha256'] != candidate['initial_weights_sha256']:
        raise ValueError('Full run initialization differs from smoke')
    if smoke['runtime_versions'] != candidate['runtime_versions']:
        raise ValueError('Runtime changed after smoke')
    metrics = pd.read_csv(output / 'candidate/training/version_0/metrics.csv')
    if sorted(metrics.epoch.dropna().astype(int).unique()) != list(range(12)):
        raise ValueError('Expected all twelve completed epochs')
    outer = roles[roles.role == 'outer'].sort_values(ID)
    values = []
    for directory, contract in [(reference, control), (output / 'candidate', candidate)]:
        path = directory / 'outer_predictions.parquet'
        if sha(path) != contract['outer_predictions_sha256']:
            raise ValueError('Prediction hash mismatch')
        predictions = pd.read_parquet(path)
        validate_prediction_provenance(roles, contract, predictions)
        values.append(predictions.set_index(ID).loc[outer[ID], LABELS].to_numpy(float))
    truth = (labels.set_index(ID).loc[outer[ID], LABELS].to_numpy(float) >= .5).astype(float)
    report = paired_proxy_evaluation(truth, *values, outer.component.to_numpy(), draws=2000)
    report.update(comparison='exploratory_historical_tiny_vs_colab_small',
                  matched_hardware_runtime=False, runtime_versions={
                      'control': control['runtime_versions'], 'candidate': candidate['runtime_versions']},
                  control_checkpoint_verified_at_packaging=True,
                  image_gain_demonstrated=False)
    ci = report['paired_sampling_95ci']
    point = (report['macro_delta'] is not None and report['macro_delta'] >= .005 and
             all(p['candidate'] is not None and p['control'] is not None and
                 p['candidate'] - p['control'] >= -.02 for p in report['per_finding'].values()))
    decision = {'exploratory_point_gate_passed': point,
                'positive_sampling_lower_bound': ci is not None and ci[0] > 0,
                'worth_proposing_matched_confirmation': point and ci is not None and ci[0] > 0,
                'promotion_allowed': False, 'automatic_followup_training': False,
                'note': 'Historical control and reused outer0: exploratory only. No submission or blend.'}
    (output / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    (output / 'screen_decision.json').write_text(json.dumps(decision, indent=2) + '\n')
    return report, decision


def run_small(code_root, inputs_path, drive_out, bundle_hash):
    code, output = Path(code_root), Path(drive_out)
    mount = Path('/content/drive/MyDrive')
    if not mount.is_dir() or not output.resolve().is_relative_to(mount.resolve()):
        raise ValueError('Mount Drive and choose DRIVE_OUT inside MyDrive')
    if verify_completed(output, bundle_hash):
        print('Already completed; all hashes verified:', output)
        return output
    # Never silently restart a partial fit or overwrite its optimizer/checkpoints.
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f'Partial run preserved at {output}. Do not rerun training from zero; '
                           'inspect its logs/checkpoints and recover the interrupted stage first.')
    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(code, output / 'source')
    shutil.copy2(inputs_path, output / 'inputs.json')
    progress = {'state': 'starting', 'bundle_sha256': bundle_hash}
    try:
        command = [sys.executable, '-u', str(code / 'tools/train_small.py'),
                   '--design', str(code / 'design/candidate'), '--inputs', str(inputs_path), '--execute']
        for phase, flag in [('smoke', ['--smoke']), ('candidate', [])]:
            progress['state'] = phase
            (output / 'progress.json').write_text(json.dumps(progress, indent=2))
            run_process(command + ['--out', str(output / phase)] + flag, code, output / f'{phase}.log')
            if phase == 'smoke':
                smoke = json.loads((output / 'smoke/smoke_receipt.json').read_text())
                if smoke.get('status') != 'passed' or smoke.get('optimizer_steps') != 2:
                    raise RuntimeError('Smoke gate did not pass; full training not started')
        report, decision = compare_candidate(code, inputs_path, output)
        receipt = {'bundle_sha256': bundle_hash,
                   'files': {name: sha(output / name) for name in completion_files()}}
        (output / 'complete.json').write_text(json.dumps(receipt, indent=2) + '\n')
        progress['state'] = 'complete'
        print(json.dumps({k: report[k] for k in ['control_macro', 'candidate_macro', 'macro_delta', 'paired_sampling_95ci']}, indent=2))
        print(json.dumps(decision, indent=2))
    except BaseException as exc:
        progress.update(state='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        (output / 'progress.json').write_text(json.dumps(progress, indent=2) + '\n')
    return output
