"""Stage the silence-masking screen as one private Kaggle T4x2 kernel.

Same pair design as the mirror screen: the unmodified frozen runner (control)
and the frozen runner with silent cells of the label-driven findings weighted 0
in training rows (candidate) train concurrently, one per T4. Declared primary
endpoint and gate: `src/model/silence_mask.py` (explicit-label AUC over the
masked findings, +0.005, unmasked findings no worse than -0.020 on full labels).
The standard macro is reported, not gated.
"""
import base64
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.prepare_colab_perturbation import payload
from tools.prepare_kaggle_mirror_screen import body as pair_body
from tools.prepare_kaggle_perturbation import SOURCES

KERNEL_ID = 'pssv27/knee-silence-screen-v1'
OUT = 'outputs/kaggle_silence_screen_v1'
LABEL_WEIGHTING = 'silence_masked_v1'


def silence_source(baseline):
    """Fail closed if the frozen runner no longer has the expected shape."""
    source = baseline
    edits = {
        "    spec, roles, labels, manifest, series, sides, paths, recipe = data\n":
            "    spec, roles, labels, manifest, series, sides, paths, recipe = data\n"
            "    from src.model.silence_mask import mask_silence\n"
            "    labels, silence_masked = mask_silence(labels, roles)\n"
            "    print('silent training cells weighted 0:', silence_masked, flush=True)\n",
        "'arm': 'control' if spec.get('sampling_policy', 'legacy') == 'legacy' else 'candidate'":
            f"'arm': 'candidate', 'label_weighting': {LABEL_WEIGHTING!r}, 'silence_masked': silence_masked",
    }
    for old, new in edits.items():
        if source.count(old) != 1:
            raise ValueError(f'Frozen runner adapter needs review: {old!r}')
        source = source.replace(old, new)
    compile(source, 'train_silence.py', 'exec')
    return source


def main_block():
    """The mirror pair driver with this candidate and the declared endpoint."""
    base = pair_body().split('\ndef assemble():', 1)[1]
    driver = base.split('\nGATE = ', 1)[1]
    replacements = {
        "('candidate', 'tools/train_mirror.py')": "('candidate', 'tools/train_silence.py')",
        "augmentation=AUGMENTATION, gate=GATE, training_launched=True,": "label_weighting=LABEL_WEIGHTING, training_launched=True,",
        "    decision = {'point_screen_passed': point, 'paired_lower_bound_positive': ci is not None and ci[0] > 0,\n"
        "                'eligible_for_separate_confirmation': point and ci is not None and ci[0] > 0,\n"
        "                'promotion_allowed': False, 'automatic_followup_training': False, 'gate': GATE}\n":
        "    from src.model.silence_mask import explicit_evaluation\n"
        "    labels_frame = pd.read_parquet(code / 'design/control/proxy_labels.parquet')\n"
        "    roles_frame = pd.read_csv(code / 'design/control/roles.csv')\n"
        "    outer_ids = roles_frame[roles_frame.role == 'outer'].sort_values(ID)[ID]\n"
        "    arms = [pd.read_parquet(out / a / 'outer_predictions.parquet').set_index(ID).loc[outer_ids, LABELS].to_numpy(float)\n"
        "            for a in ('control', 'candidate')]\n"
        "    explicit = explicit_evaluation(labels_frame, roles_frame, *arms)\n"
        "    (out / 'explicit_evaluation.json').write_text(json.dumps(explicit, indent=2) + '\\n')\n"
        "    decision = explicit['decision']\n",
    }
    for old, new in replacements.items():
        if driver.count(old) != 1:
            raise ValueError(f'Pair driver changed; review the reuse: {old[:60]!r}')
        driver = driver.replace(old, new)
    # The standard-evaluation imports must precede the explicit block.
    old = "    from tools.evaluate_nested_proxy import evaluate\n    import pandas as pd\n    from src.validation.proxy import ID, LABELS, paired_proxy_evaluation\n"
    if driver.count(old) != 1:
        raise ValueError('Pair driver imports changed')
    return driver


def script(root=ROOT):
    root = Path(root)
    archive, _, _ = payload(root, include_series=False)
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        manifest = json.loads(bundle.read('bundle_manifest.json'))
        files = {name: bundle.read(name) for name in
                 ['src/model/perturb.py', 'tools/perturbation_probe.py', 'tools/colab_small_runtime.py']}
        frozen_runner = bundle.read('tools/train_nested_proxy_control.py').decode()
    files['src/model/silence_mask.py'] = (root / 'src/model/silence_mask.py').read_bytes()
    files['tools/train_silence.py'] = silence_source(frozen_runner).encode()
    manifest.update({name: hashlib.sha256(data).hexdigest() for name, data in files.items()})
    embedded = {name: base64.b64encode(data).decode() for name, data in files.items()}
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    head = pair_body().split('\ndef assemble():', 1)[0]
    helpers = pair_body().split('\ndef assemble():', 1)[1].split('\nGATE = ', 1)[0]
    source = (f'BUNDLE_SHA256 = {digest!r}\nMANIFEST = {manifest!r}\nEMBEDDED = {embedded!r}\n'
              f'LABEL_WEIGHTING = {LABEL_WEIGHTING!r}\n' + head + '\ndef assemble():' + helpers
              + '\nGATE = ' + main_block())
    compile(source, 'silence_screen.py', 'exec')
    return source, digest


def main(root=ROOT, destination=None):
    out = Path(destination) if destination else Path(root) / OUT
    out.mkdir(parents=True, exist_ok=False)
    source, digest = script(root)
    (out / 'silence_screen.py').write_text(source)
    metadata = {'id': KERNEL_ID, 'title': 'Knee Silence Screen V1', 'code_file': 'silence_screen.py',
                'language': 'python', 'kernel_type': 'script', 'is_private': True,
                'enable_gpu': True, 'enable_tpu': False, 'enable_internet': True,
                'dataset_sources': [], 'competition_sources': ['rsna-knee-abnormality-detection'],
                'kernel_sources': SOURCES, 'machine_shape': 'NvidiaTeslaT4'}
    (out / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(out, digest)
    return out


if __name__ == '__main__':
    main()
