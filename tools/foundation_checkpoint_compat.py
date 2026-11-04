"""A narrow loader compatibility shim; the frozen training bundle stays unchanged.

torch.__version__ is a TorchVersion (str subclass) in callback provenance saved
inside our own hash-verified checkpoints. weights_only=True needs that specific
installed PyTorch class allowlisted. No unrestricted pickle loading is enabled.
"""
import hashlib
import json
from pathlib import Path

from src.validation.proxy import sha

CHILD = ("import runpy,sys,torch; "
         "from torch.torch_version import TorchVersion; "
         "torch.serialization.add_safe_globals([TorchVersion]); "
         "script=sys.argv.pop(1); runpy.run_path(script,run_name='__main__')")
SHIM = '''
# CHECKPOINT_TORCHVERSION_COMPAT: preserve the original hash-bound training source.
_original_foundation_run_process = run_process
def run_process(command, code, log_path):
    if len(command) > 2 and command[1] == '-u' and command[2].endswith('/tools/train_foundation_finetune.py'):
        command = [command[0], '-u', '-c', COMPAT_CHILD, *command[2:]]
    return _original_foundation_run_process(command, code, log_path)
'''.replace("COMPAT_CHILD", repr(CHILD))


def apply(staged):
    staged = Path(staged)
    path = staged / "foundation_kernel.py"
    receipt_path = staged / "staging_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if sha(path) != receipt["source_sha256"] or "loader_compatibility" in receipt:
        raise ValueError("Changed or already patched staged source")
    source = path.read_text()
    marker = "\nif __name__ == '__main__':\n    run(BUNDLE_BASE64, BUNDLE_SHA256, EXTERNAL_FILES)\n"
    if not source.endswith(marker):
        raise ValueError("Unexpected Kaggle entry point")
    patched = source[:-len(marker)] + SHIM + marker
    compile(patched, str(path), "exec")
    if len(patched.encode()) >= 1_000_000:
        raise ValueError("Patched source exceeds Kaggle limit")
    original_hash = receipt["source_sha256"]
    path.write_text(patched)
    receipt.update(source_sha256=sha(path), source_bytes=len(patched.encode()),
        loader_compatibility={"allowlisted_class": "torch.torch_version.TorchVersion",
                              "weights_only_remains_true": True, "training_bundle_unchanged": True,
                              "unpatched_source_sha256": original_hash,
                              "shim_sha256": hashlib.sha256(SHIM.encode()).hexdigest()})
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
