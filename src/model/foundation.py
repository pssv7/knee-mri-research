"""Offline DINOv3-L / OrthoFoundation integration, isolated from legacy models.

The official forward_features API retains rotary position embeddings. The
OrthoFoundation release's custom block loop is deliberately not imported.
"""
from contextlib import nullcontext
import hashlib
import importlib
import json
from pathlib import Path
import sys

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from src.model.net import KneeNet, IMAGENET_MEAN, IMAGENET_STD

ORTHO_SHA256 = "385a775822107b68eaa486336feb982e1ce7bd6d4e8c03ceb482a0bf546f2ff9"
DINOV3_COMMIT = "6876159a11b4df116f30f667f8c9888617df0751"
ENCODER_DIM = 1024
HEAD_DIM = 768
IMAGE_SIZE = 224


def file_sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_source(repo, lock):
    repo = Path(repo).resolve()
    if lock.get("dinov3_commit") != DINOV3_COMMIT or not lock.get("source_sha256"):
        raise ValueError("Unrecognized or incomplete DINOv3 source lock")
    for relative, expected in lock["source_sha256"].items():
        path = (repo / relative).resolve()
        if not path.is_relative_to(repo) or not path.is_file() or file_sha(path) != expected:
            raise ValueError(f"DINOv3 source missing or changed: {relative}")
    # Added Python files can change imports even when all original files match.
    actual = {str(p.relative_to(repo)) for p in (repo / "dinov3").rglob("*.py")}
    expected = {p for p in lock["source_sha256"] if p.startswith("dinov3/") and p.endswith(".py")}
    if actual != expected:
        raise ValueError("DINOv3 Python source inventory changed")
    for name, module in tuple(sys.modules.items()):
        if name == "dinov3" or name.startswith("dinov3."):
            origin = getattr(module, "__file__", None)
            if origin and not Path(origin).resolve().is_relative_to(repo):
                raise RuntimeError("Another DINOv3 installation is already imported; use a fresh process")
    return repo


def extract_encoder_state(state, arm):
    """Accept only the two known release formats; never silently drop keys."""
    if arm not in {"orthofoundation", "dinov3"}:
        raise ValueError("Unknown foundation arm")
    if not isinstance(state, dict) or not state or not all(
            isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
        raise ValueError("Expected a tensor-only encoder state dictionary")
    prefix = "backbone." if arm == "orthofoundation" else ""
    if prefix and not all(k.startswith(prefix) for k in state):
        raise ValueError("OrthoFoundation release must contain only backbone.* keys")
    result = {k[len(prefix):]: v for k, v in state.items()}
    if "storage_tokens" not in result or "rope_embed.periods" not in result:
        raise ValueError("Checkpoint is not the expected DINOv3 architecture")
    return result


class FoundationEncoder(nn.Module):
    num_features = ENCODER_DIM

    def __init__(self, encoder, frozen=True, chunk_size=16, checkpoint_blocks=False):
        super().__init__()
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.encoder = encoder
        self.frozen = frozen
        self.chunk_size = chunk_size
        self.checkpoint_blocks = checkpoint_blocks
        self.encoder.requires_grad_(not frozen)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self

    def _features(self, images):
        result = self.encoder.forward_features(images)
        if not isinstance(result, dict) or "x_norm_clstoken" not in result:
            raise ValueError("Official DINOv3 class-token output is required")
        features = result["x_norm_clstoken"]
        if features.shape != (len(images), self.num_features):
            raise ValueError("Unexpected DINOv3 feature shape")
        return features

    def forward(self, images):
        if images.ndim != 4 or images.shape[1:] != (3, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError("Foundation encoder requires N x 3 x 224 x 224 normalized images")
        chunks = []
        with torch.no_grad() if self.frozen else nullcontext():
            for part in images.split(self.chunk_size):
                if self.checkpoint_blocks and self.training and not self.frozen:
                    features = activation_checkpoint(self._features, part, use_reentrant=False)
                else:
                    features = self._features(part)
                chunks.append(features)
        return torch.cat(chunks)


def load_encoder(repo, weights, arm, lock, expected_sha256=None, frozen=True,
                 chunk_size=16, checkpoint_blocks=False):
    """Verify local files, construct official architecture, and strictly load.

    No network fallback, pickle fallback, approximate key matching, or random
    initialization fallback is allowed. The DINOv3 control needs a supplied
    official checkpoint and its full SHA-256 in the experiment specification.
    """
    if isinstance(lock, (str, Path)):
        lock = json.loads(Path(lock).read_text())
    repo = verify_source(repo, lock)
    digest = ORTHO_SHA256 if arm == "orthofoundation" else expected_sha256
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("A full expected checkpoint SHA-256 is required")
    if expected_sha256 is not None and expected_sha256 != digest:
        raise ValueError("Checkpoint identity differs from the released OrthoFoundation asset")
    if not Path(weights).is_file() or file_sha(weights) != digest:
        raise ValueError("Encoder checkpoint missing or SHA-256 mismatch")
    sys.path.insert(0, str(repo))
    try:
        # Avoid hubconf.py, which imports unrelated segmentation dependencies.
        build = importlib.import_module("dinov3.hub.backbones").dinov3_vitl16
        with torch.random.fork_rng(devices=[]):
            encoder = build(pretrained=False)
        state = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)
        encoder.load_state_dict(extract_encoder_state(state, arm), strict=True)
    finally:
        sys.path.remove(str(repo))
    return FoundationEncoder(encoder, frozen, chunk_size, checkpoint_blocks)


class FoundationKneeNet(KneeNet):
    """Same aggregation design as Tiny, with a 1024 -> 768 feature projection.

    Each grayscale slice occupies all three channels, as in domain pretraining.
    A distinct checkpoint signature prevents accidental loading into the legacy
    adjacent-slice path. cached=True consumes unprojected slice embeddings.
    """

    def __init__(self, backbone=None, *, cached=False, head_dim=HEAD_DIM):
        if cached != (backbone is None):
            raise ValueError("Cached heads have no backbone; image models require one")
        super().__init__(backbone, head_dim, input_dim=ENCODER_DIM,
                         input_mean=IMAGENET_MEAN, input_std=IMAGENET_STD)
        self.cached = cached
        self.register_buffer("foundation_signature", torch.tensor(
            [1, IMAGE_SIZE, ENCODER_DIM, head_dim, 1], dtype=torch.int64))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        key = prefix + "foundation_signature"
        if key not in state_dict or not torch.equal(
                state_dict[key].cpu(), self.foundation_signature.cpu()):
            error_msgs.append("Missing or incompatible foundation preprocessing signature")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def extract_slices(self, volumes):
        if self.cached or volumes.ndim != 4 or volumes.shape[-2:] != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError("Slice extraction requires an image encoder and N x D x 224 x 224")
        if volumes.dtype == torch.uint8:
            volumes = volumes.float() / 255.0
        # Float callers follow the existing dataset's [0,1] convention.
        images = volumes.flatten(0, 1).unsqueeze(1).expand(-1, 3, -1, -1)
        images = (images - self.input_mean) / self.input_std
        return self.backbone(images).reshape(*volumes.shape[:2], ENCODER_DIM)

    def encode_series(self, volumes):
        features = volumes if self.cached else self.extract_slices(volumes)
        if features.ndim != 3 or features.shape[-1] != ENCODER_DIM:
            raise ValueError("Expected N x D x 1024 foundation features")
        return self.slice_pool(self.project(features.float()))


def make_model(repo, weights, arm, lock, *, seed=0, frozen=True, chunk_size=16):
    digest = file_sha(weights)
    if arm == "dinov3" and not digest.startswith("8aa4cbdd"):
        raise ValueError("Control must be the official LVD1689M 8aa4cbdd checkpoint")
    encoder = load_encoder(repo, weights, arm, lock, frozen=frozen,
                           expected_sha256=digest if arm == "dinov3" else None,
                           chunk_size=chunk_size, checkpoint_blocks=not frozen)
    # Encoder construction/loading must not change the matched head initialization.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return FoundationKneeNet(encoder)


def head_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            if not k.startswith("backbone.")}


def state_sha(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode())
        h.update(str((value.dtype, tuple(value.shape))).encode())
        h.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
