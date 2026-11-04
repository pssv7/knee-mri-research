import io
import json
import pickle
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch
from torch.torch_version import TorchVersion

from src.validation.proxy import sha
from tools.foundation_checkpoint_compat import apply, CHILD

ROOT = Path(__file__).resolve().parents[1]


def test_narrow_torchversion_allowlist_keeps_weights_only_loading():
    blob = io.BytesIO()
    torch.save({"binding": {"torch": torch.__version__}, "weight": torch.ones(2)}, blob)
    blob.seek(0)
    with pytest.raises(pickle.UnpicklingError, match="TorchVersion"):
        torch.load(blob, weights_only=True)
    blob.seek(0)
    with torch.serialization.safe_globals([TorchVersion]):
        restored = torch.load(blob, weights_only=True)
    assert restored["binding"]["torch"] == torch.__version__
    assert torch.equal(restored["weight"], torch.ones(2))


def test_child_wrapper_preserves_arguments_and_loads_versioned_checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"version": torch.__version__}, checkpoint)
    script = tmp_path / "child.py"
    script.write_text("import torch,sys; assert sys.argv[1]=='--resume'; "
                      "print(torch.load(sys.argv[2],weights_only=True)['version'])")
    result = subprocess.run([sys.executable, "-u", "-c", CHILD, str(script), "--resume", str(checkpoint)],
                            capture_output=True, text=True, check=True)
    assert result.stdout.strip() == str(torch.__version__)


def test_compatibility_preserves_bundle_and_only_wraps_finetune_child(tmp_path):
    original = ROOT / "outputs/kaggle_foundation_finetune_s01"
    if not original.is_dir():
        pytest.skip("Local staged Kaggle artifact is not available in this checkout")
    shutil.copyfile(original / "foundation_kernel.py", tmp_path / "foundation_kernel.py")
    shutil.copyfile(original / "staging_receipt.json", tmp_path / "staging_receipt.json")
    old = json.loads((tmp_path / "staging_receipt.json").read_text())
    receipt = apply(tmp_path)
    assert receipt["bundle_sha256"] == old["bundle_sha256"]
    assert receipt["source_sha256"] != old["source_sha256"]
    scope = {"__name__": "inspect_only"}
    exec((tmp_path / "foundation_kernel.py").read_text(), scope)
    calls = []
    scope["_original_foundation_run_process"] = lambda *args: calls.append(args)
    scope["run_process"]([sys.executable, "-u", "/source/tools/train_foundation_finetune.py", "--resume", "prior"], "code", "log")
    assert calls[-1][0][2:4] == ["-c", CHILD]
    assert calls[-1][0][-3:] == ["/source/tools/train_foundation_finetune.py", "--resume", "prior"]
    command = [sys.executable, "-u", "/source/tools/evaluate_foundation.py"]
    scope["run_process"](command, "code", "log")
    assert calls[-1][0] == command
    assert receipt["source_sha256"] == sha(tmp_path / "foundation_kernel.py")
    with pytest.raises(ValueError, match="already patched"):
        apply(tmp_path)
