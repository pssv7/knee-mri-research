"""Tests for the frozen DINOv2 slice encoder.

The encoder runs at both training and submission time, so the properties worth
guarding are the ones whose breakage is silent: a wrong input size that timm
quietly accepts, or a pooling that drops the patch tokens.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")

from src import config  # noqa: E402
from src.model import embed  # noqa: E402


@pytest.fixture(scope="module")
def encoder():
    # pretrained=False: the architecture and pooling are what is under test,
    # and downloading weights in a unit test would be absurd.
    return embed.build_encoder(pretrained=False)


class TestModelChoice:
    def test_name_has_pretrained_weights_in_timm(self):
        assert embed.MODEL_NAME in timm.list_models("*dinov2*", pretrained=True)

    def test_registers_variant_is_used(self):
        # Registers absorb the high-norm artefact tokens that would otherwise
        # contaminate the patch mean we pool.
        assert "reg4" in embed.MODEL_NAME

    def test_input_size_matches_the_cache(self):
        assert embed.INPUT_SIZE == config.CACHE_SIZE

    def test_input_size_divides_by_the_patch_size(self):
        # 224 / 14 = 16. A size that does not divide cleanly would be resized
        # or padded somewhere out of sight.
        assert embed.INPUT_SIZE % 14 == 0


class TestEncoder:
    def test_is_frozen(self, encoder):
        assert not any(p.requires_grad for p in encoder.parameters())

    def test_is_in_eval_mode(self, encoder):
        assert not encoder.training

    def test_prefix_tokens_are_cls_plus_registers(self, encoder):
        assert encoder.num_prefix_tokens == 5

    def test_embedding_width(self, encoder):
        x = torch.randint(0, 255, (2, 3, embed.INPUT_SIZE, embed.INPUT_SIZE),
                          dtype=torch.uint8)
        assert embed.embed_slices(encoder, x).shape == (2, embed.EMBED_DIM)

    def test_patch_tokens_contribute(self, encoder):
        # If the second half were also CLS, the two halves would be identical
        # and the mean-patch hedge would be doing nothing.
        x = torch.randint(0, 255, (1, 3, embed.INPUT_SIZE, embed.INPUT_SIZE),
                          dtype=torch.uint8)
        out = embed.embed_slices(encoder, x)[0]
        assert not torch.allclose(out[:embed.TOKEN_DIM], out[embed.TOKEN_DIM:])

    def test_deterministic(self, encoder):
        x = torch.randint(0, 255, (2, 3, embed.INPUT_SIZE, embed.INPUT_SIZE),
                          dtype=torch.uint8)
        assert torch.allclose(embed.embed_slices(encoder, x),
                              embed.embed_slices(encoder, x))

    def test_float_input_also_accepted(self, encoder):
        x = torch.rand(1, 3, embed.INPUT_SIZE, embed.INPUT_SIZE)
        assert embed.embed_slices(encoder, x).shape == (1, embed.EMBED_DIM)


class TestNormalisation:
    def test_uint8_is_scaled_before_standardising(self, encoder):
        white = torch.full((1, 3, 8, 8), 255, dtype=torch.uint8)
        out = embed.normalize(white)
        # (1.0 - mean) / std, not (255 - mean) / std.
        expected = (1.0 - embed.MEAN[0]) / embed.STD[0]
        assert out[0, 0].mean().item() == pytest.approx(expected, abs=1e-4)

    def test_channels_use_their_own_statistics(self, encoder):
        x = torch.full((1, 3, 4, 4), 128, dtype=torch.uint8)
        out = embed.normalize(x)
        assert not torch.allclose(out[0, 0], out[0, 2])


class TestSpatialPooling:
    """Patch tokens pooled to a grid rather than averaged flat.

    The mean over 256 patches is where a focal finding disappears. What must
    hold is that pooling respects the 2D layout -- averaging a patch with its
    spatial neighbours, not with whatever sat beside it in a flattened
    sequence.
    """

    def test_dimensions_follow_the_grid(self):
        assert embed.embed_dim(1) == embed.TOKEN_DIM * 2
        assert embed.embed_dim(2) == embed.TOKEN_DIM * 5
        assert embed.embed_dim(4) == embed.TOKEN_DIM * 17

    def test_encoder_output_matches_declared_width(self, encoder):
        x = torch.randint(0, 255, (2, 3, embed.INPUT_SIZE, embed.INPUT_SIZE),
                          dtype=torch.uint8)
        for grid in (1, 2, 4):
            out = embed.embed_slices(encoder, x, grid=grid)
            assert out.shape == (2, embed.embed_dim(grid)), grid

    def test_pooling_is_spatial_not_sequential(self):
        # A gradient across the grid must survive pooling as a gradient.
        # Flat-sequence pooling would scramble it.
        side, channels = 4, 2
        values = torch.arange(side * side, dtype=torch.float32)
        patches = values.view(1, side * side, 1).repeat(1, 1, channels)
        pooled = embed.pool_patches(patches, grid=2).view(2, 2, channels)
        # Top-left quadrant averages {0,1,4,5} = 2.5; bottom-right {10,11,14,15} = 12.5
        assert pooled[0, 0, 0].item() == pytest.approx(2.5)
        assert pooled[1, 1, 0].item() == pytest.approx(12.5)

    def test_grid_one_is_the_plain_mean(self, encoder):
        x = torch.rand(1, 3, embed.INPUT_SIZE, embed.INPUT_SIZE)
        out = embed.embed_slices(encoder, x, grid=1)
        assert out.shape[-1] == embed.TOKEN_DIM * 2

    def test_non_square_patch_count_falls_back(self):
        # Rather than reshaping into the wrong grid.
        patches = torch.rand(1, 7, 3)
        assert embed.pool_patches(patches, grid=2).shape == (1, 3)

    def test_spatial_retains_more_than_the_mean(self):
        # Two arrangements of the same patch values: the mean cannot tell them
        # apart, the grid must.
        side = 4
        a = torch.arange(side * side, dtype=torch.float32).view(1, -1, 1)
        b = a.flip(1)
        assert torch.allclose(a.mean(1), b.mean(1))
        assert not torch.allclose(embed.pool_patches(a, 2),
                                  embed.pool_patches(b, 2))
