"""A checkpoint must carry the statistics it was scored under.

The original defect: `KneeNet` registered input_mean/input_std with
persistent=False, so they were absent from the state dict and every
construction site had to supply them again. A site that forgot did not fail --
the strict load restored every weight, normalisation stayed off, and the model
simply predicted worse. ConvNeXt hid it entirely (its defaults are None), so
it only ever showed on the ViT path, where the 24-slice run logged gold 0.864
in memory and 0.823 from the same weights reloaded.

The buffers are persistent now, and `KneeNet._load_from_state_dict` reconciles
both directions. These tests hold that fix in place from three angles: the
statistics travel, a bare model adopts them, and a checkpoint written before
the change still loads strictly. The notebook checks stay too -- passing the
statistics explicitly at each site is still correct, and it is what makes a
freshly initialised model score the same way as a loaded one.
"""

import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import net  # noqa: E402

VIT = "vit_small_patch14_reg4_dinov2.lvd142m"
CONVNEXT = "convnext_tiny.fb_in22k_ft_in1k"
NOTEBOOKS = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "notebooks")


def vit_model():
    defaults = net.backbone_defaults(VIT)
    return net.KneeNet(None, 384, input_dim=384,
                       input_mean=defaults["input_mean"],
                       input_std=defaults["input_std"])


class TestNormalisationTravels:
    def test_the_buffers_are_in_the_state_dict(self):
        state = vit_model().state_dict()
        assert "input_mean" in state and "input_std" in state

    def test_a_bare_model_adopts_them_on_load(self):
        trained = vit_model()
        naive = net.KneeNet(None, 384, input_dim=384)
        assert naive.input_mean is None
        naive.load_state_dict(trained.state_dict(), strict=True)
        assert torch.equal(naive.input_mean, trained.input_mean)

    def test_the_adopted_model_predicts_identically(self):
        # The end the whole thing is for: two constructions of the same
        # checkpoint must score a study the same way.
        torch.manual_seed(0)
        trained = vit_model().eval()
        naive = net.KneeNet(None, 384, input_dim=384).eval()
        naive.load_state_dict(trained.state_dict(), strict=True)

        features = torch.rand(1, 2, 3, 384)
        meta = torch.rand(1, 2, trained.film.to_scale_shift[0].in_features)
        mask = torch.ones(1, 2, dtype=torch.bool)
        with torch.no_grad():
            assert torch.allclose(trained.forward_features(features, meta, mask),
                                  naive.forward_features(features, meta, mask))

    def test_a_checkpoint_from_before_the_change_still_loads(self):
        trained = vit_model()
        old = {k: v for k, v in trained.state_dict().items()
               if not k.startswith("input_")}
        vit_model().load_state_dict(old, strict=True)

    def test_convnext_state_dicts_are_unchanged(self):
        # No new keys, so every ConvNeXt checkpoint already uploaded to Kaggle
        # loads against the current class and every reference number stands.
        plain = net.KneeNet(None, 384, input_dim=384)
        assert not [k for k in plain.state_dict() if k.startswith("input_")]


class TestEveryNotebookSuppliesIt:
    """The generated notebooks are the actual construction sites.

    Still enforced after the buffers were made persistent: a model built for
    training has no checkpoint to adopt from, and the scoring sites should not
    depend on the reconciliation to be correct.
    """

    def sources(self, name):
        with open(os.path.join(NOTEBOOKS, name)) as handle:
            notebook = json.load(handle)
        return ["".join(c["source"]) for c in notebook["cells"]
                if c["cell_type"] == "code"]

    @pytest.mark.parametrize("name", ["08_finetune.ipynb",
                                      "08_finetune_colab.ipynb",
                                      "10_submit_ensemble.ipynb",
                                      "10_submit_ensemble_colab.ipynb"])
    def test_kneenet_is_never_built_bare(self, name):
        for source in self.sources(name):
            for line, text in enumerate(source.splitlines(), 1):
                if "net.KneeNet(" not in text:
                    continue
                # The frozen path passes input_dim and has no trunk to
                # normalise for; every trunk-bearing site must pass the stats.
                if "input_dim=" in text or "input_dim=input_dim" in text:
                    continue
                tail = "\n".join(source.splitlines()[line - 1:line + 4])
                assert "input_mean" in tail, (
                    f"{name}: KneeNet built without input_mean near line "
                    f"{line}:\n{tail}")


class TestGeometryTravelsTogether:
    def test_notebook_10_matches_notebook_08(self):
        def constant(sources, name):
            for source in sources:
                for text in source.splitlines():
                    if text.startswith(f"{name} ="):
                        return text.split("=", 1)[1].strip()
            return None

        eight = self.sources = TestEveryNotebookSuppliesIt().sources
        train = eight("08_finetune.ipynb")
        submit = eight("10_submit_ensemble.ipynb")
        assert constant(train, "BACKBONE") == constant(submit, "TUNED_BACKBONE")
        assert constant(train, "SLICES_PER_SERIES") == constant(
            submit, "TUNED_SLICES")
        assert constant(train, "MAX_SERIES") == constant(
            submit, "TUNED_MAX_SERIES")
