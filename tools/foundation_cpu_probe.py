"""Bounded real-weight CPU verification using synthetic pixel tensors, not patients."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from src.model import foundation as fm, net, dataset
from src.validation.proxy import write_json_new, sha


def probe(repo, weights, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    started = time.monotonic()
    lock = ROOT / "configs/orthofoundation_assets.json"
    model = fm.make_model(repo, weights, "orthofoundation", lock, chunk_size=8).eval()
    volumes = torch.randint(0, 256, (1, 5, 16, 224, 224), dtype=torch.uint8)
    meta = torch.zeros(1, 5, len(dataset.META_COLUMNS))
    mask = torch.ones(1, 5, dtype=torch.bool)
    with torch.no_grad():
        features = model.extract_slices(volumes.flatten(0, 1)).reshape(1, 5, 16, 1024)
        image_logits = model(volumes, meta, mask)
    head = fm.FoundationKneeNet(cached=True).eval()
    head.load_state_dict(fm.head_state(model), strict=True)
    with torch.no_grad():
        cached_logits = head(features, meta, mask)
    torch.testing.assert_close(image_logits, cached_logits, atol=1e-6, rtol=1e-6)
    before_head = fm.state_sha(fm.head_state(head))
    head.train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4)
    targets = torch.linspace(.1, .9, 12).reshape(1, 12)
    loss = net.weighted_bce(head(features, meta, mask), targets, torch.ones_like(targets))
    loss.backward()
    grads = [p.grad for p in head.parameters() if p.grad is not None]
    if not grads or not all(torch.isfinite(g).all() for g in grads):
        raise RuntimeError("Missing or nonfinite head gradients")
    optimizer.step()
    if fm.state_sha(fm.head_state(head)) == before_head:
        raise RuntimeError("Head optimizer did not update weights")
    selected = out / "transient_head.pt"
    torch.save(head.state_dict(), selected)
    restored_head = fm.FoundationKneeNet(cached=True).eval()
    restored_head.load_state_dict(torch.load(selected, weights_only=True, map_location="cpu"), strict=True)
    head.eval()
    with torch.no_grad():
        before = head(features, meta, mask)
        after = restored_head(features, meta, mask)
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    # Reconstruct the encoder from the release and compare an actual image.
    fresh = fm.make_model(repo, weights, "orthofoundation", lock, chunk_size=8).eval()
    fresh.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        fresh_feature = fresh.extract_slices(volumes[:, :1, :1].reshape(1, 1, 224, 224))
    torch.testing.assert_close(fresh_feature, features[:, 0, :1], rtol=1e-5, atol=1e-5)
    receipt = {"status": "passed", "device": "cpu", "torch": torch.__version__,
        "input": "synthetic uint8 pixel tensors; no diagnostic validity test",
        "study_shape": list(volumes.shape), "feature_shape": list(features.shape),
        "logit_shape": list(image_logits.shape), "encoder_sha256": sha(weights),
        "dinov3_commit": fm.DINOV3_COMMIT,
        "encoder_parameters": sum(p.numel() for p in model.backbone.parameters()),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "image_vs_cached_max_abs": float((image_logits - cached_logits).abs().max()),
        "head_roundtrip_max_abs": float((before - after).abs().max()),
        "encoder_reload_max_abs": float((fresh_feature - features[:, 0, :1]).abs().max()),
        "head_optimizer_steps": 1, "finite_loss": float(loss.detach()),
        "elapsed_seconds": time.monotonic() - started, "gpu_tested": False,
        "accuracy_gain_demonstrated": False, "trained_weights_retained": False,
        "source_sha256": {name: sha(ROOT / name) for name in
            ["src/model/foundation.py", "tools/foundation_cpu_probe.py", "configs/orthofoundation_assets.json"]}}
    selected.unlink()
    write_json_new(out / "cpu_probe.json", receipt)
    return receipt


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["repo", "weights", "out"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(probe(a.repo, a.weights, a.out), indent=2))
