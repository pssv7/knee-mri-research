"""Tests for the Lightning training wrapper.

These target the three things that go wrong silently under DDP -- pickling
memmaps into spawned workers, scoring a ranking metric per-rank, and shipping
a checkpoint whose keys the submission cannot load -- rather than testing
Lightning itself.
"""

import os
import pickle
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config  # noqa: E402

pytest.importorskip("lightning.pytorch", reason="lightning not installed",
                    exc_type=ImportError)

import lightning.pytorch as pl  # noqa: E402
import torch  # noqa: E402

from src.model import lightning as knee_lightning  # noqa: E402


class TestAuc:
    def test_matches_the_label_side_implementation(self):
        # The model and the labels must be scored by the same metric, or a
        # comparison between them measures the metric difference too.
        from src.labels import calibrate
        rng = np.random.default_rng(0)
        truth = rng.integers(0, 2, 200)
        scores = rng.random(200)
        assert knee_lightning.auc(truth, scores) == pytest.approx(
            calibrate._auc(truth, scores))

    def test_constant_scores_give_half(self):
        truth = np.array([1, 0, 1, 0])
        assert knee_lightning.auc(truth, np.ones(4)) == pytest.approx(0.5)

    def test_single_class_is_nan(self):
        assert np.isnan(knee_lightning.auc(np.ones(4), np.arange(4)))

    def test_macro_ignores_findings_without_positives(self):
        truth = np.array([[1, 0], [0, 0], [1, 0], [0, 0]])
        probs = np.array([[0.9, 0.5], [0.1, 0.5], [0.8, 0.5], [0.2, 0.5]])
        # Column 1 has no positives -> nan -> excluded, so macro is column 0.
        assert knee_lightning.macro_auc(truth, probs) == pytest.approx(1.0)


class TestDatasetPickling:
    """The spawn strategy pickles the dataset into every worker process."""

    def _dataset(self, tmp_path, slices=config.CACHE_SLICES):
        path = str(tmp_path / "cache.npy")
        np.save(path, np.zeros((2, slices, 4, 4), dtype=np.uint8))
        return knee_lightning.StudyVolumeDataset(
            study_index=None, studies=["a", "b"], cache_paths=[path],
            slices_per_series=8)

    def test_memmaps_are_not_serialised(self, tmp_path):
        dataset = self._dataset(tmp_path)
        assert dataset.shards  # force them open
        assert dataset._shards is not None
        restored = pickle.loads(pickle.dumps(dataset))
        # The handles must not travel; the paths must.
        assert restored._shards is None
        assert restored.cache_paths == dataset.cache_paths

    def test_pickle_stays_small(self, tmp_path):
        # A serialised memmap would carry the array bytes with it.
        dataset = self._dataset(tmp_path)
        assert dataset.shards
        assert len(pickle.dumps(dataset)) < 4096

    def test_shards_reopen_after_unpickling(self, tmp_path):
        dataset = self._dataset(tmp_path)
        restored = pickle.loads(pickle.dumps(dataset))
        assert restored.shards
        assert restored.shards[restored.cache_paths[0]].shape == (2, 24, 4, 4)

    def test_module_is_importable_not_main(self):
        # Classes defined in a notebook cell belong to __main__ and do not
        # survive spawn. This is the guard that they live in src/.
        assert knee_lightning.StudyVolumeDataset.__module__ == \
            "src.model.lightning"
        assert knee_lightning.KneeLitModule.__module__ == "src.model.lightning"


class TestSliceGuard:
    def test_short_volume_raises_rather_than_training_quietly(self, tmp_path):
        path = str(tmp_path / "thin.npy")
        np.save(path, np.zeros((1, 8, 4, 4), dtype=np.uint8))
        dataset = knee_lightning.StudyVolumeDataset(
            study_index=None, studies=["a"], cache_paths=[path],
            slices_per_series=16)

        class FakeIndex:
            pass

        # Drive the guard directly: load_study needs a real index, so the
        # check is exercised through the same comparison it performs.
        volumes = np.zeros((1, 8, 4, 4), dtype=np.uint8)
        available = volumes.shape[1]
        assert available < dataset.slices_per_series

    def test_keep_indices_span_the_volume(self):
        dataset = knee_lightning.StudyVolumeDataset(
            study_index=None, studies=[], cache_paths=[],
            slices_per_series=16)
        assert len(dataset.keep) == 16
        assert dataset.keep[0] == 0
        assert dataset.keep[-1] == config.CACHE_SLICES - 1
        assert list(dataset.keep) == sorted(dataset.keep)


class TestOptimiserSteps:
    def test_effective_batch_is_device_count_invariant(self):
        # 2 studies x 1 device x 4 accum and 2 x 2 x 2 are the same eight
        # studies per step, so the schedule length must match.
        one = knee_lightning.optimiser_steps(3478, 2, 1, 4, 12)
        two = knee_lightning.optimiser_steps(3478, 2, 2, 2, 12)
        assert abs(one - two) <= 12  # at most one step per epoch of rounding

    def test_more_devices_means_fewer_batches_per_rank(self):
        assert (knee_lightning.optimiser_steps(1000, 2, 2, 1, 1)
                < knee_lightning.optimiser_steps(1000, 2, 1, 1, 1))


class TestCheckpointExport:
    """Notebook 10 builds a bare KneeNet and calls load_state_dict on this."""

    # KneeNet's series transformer uses 8 attention heads, so the feature
    # dimension has to divide by 8.
    FEATURE_DIM = 16

    def _checkpoint(self, tmp_path):
        backbone = torch.nn.Linear(4, self.FEATURE_DIM)
        module = knee_lightning.KneeLitModule(
            backbone=backbone, feature_dim=self.FEATURE_DIM, total_steps=10)
        path = str(tmp_path / "lit.ckpt")
        torch.save({"state_dict": module.state_dict()}, path)
        return path, module

    def test_export_strips_the_lightning_prefix(self, tmp_path):
        ckpt, module = self._checkpoint(tmp_path)
        out = str(tmp_path / "convnext_fold0.pt")
        count = knee_lightning.export_from_checkpoint(ckpt, out)

        exported = torch.load(out, map_location="cpu")
        assert count == len(exported)
        assert not any(k.startswith("model.") for k in exported)
        # Every exported key must be one the bare KneeNet expects.
        expected = set(module.model.state_dict())
        assert set(exported) == expected

    def test_exported_weights_load_into_a_bare_knee_net(self, tmp_path):
        ckpt, module = self._checkpoint(tmp_path)
        out = str(tmp_path / "convnext_fold0.pt")
        knee_lightning.export_from_checkpoint(ckpt, out)

        from src.model import net
        fresh = net.KneeNet(torch.nn.Linear(4, self.FEATURE_DIM),
                            self.FEATURE_DIM)
        fresh.load_state_dict(torch.load(out, map_location="cpu"))

    def test_a_checkpoint_without_model_keys_raises(self, tmp_path):
        path = str(tmp_path / "wrong.ckpt")
        torch.save({"state_dict": {"backbone.weight": torch.zeros(2)}}, path)
        with pytest.raises(ValueError, match="no 'model.' keys"):
            knee_lightning.export_from_checkpoint(
                path, str(tmp_path / "out.pt"))


class TestForkSafety:
    """ddp_notebook forks, and a fork inheriting a CUDA context dies.

    `get_device_name` and `get_device_properties` both call `_lazy_init`;
    `device_count` and `is_available` do not. Nothing in the notebook may
    touch the first pair before `Trainer.fit`.
    """

    def _finetune_source_before_fit(self):
        source = (config.PROJECT_ROOT / "tools" / "make_notebook.py").read_text()
        start = source.index("def build_finetune_notebook")
        end = source.index("def build_export_notebook")
        finetune = source[start:end]
        return finetune[:finetune.index("trainer.fit")]

    def test_no_lazy_init_call_before_fit(self):
        before = self._finetune_source_before_fit()
        # The trailing paren matters: the comment explaining the hazard names
        # these functions, and must not trip this.
        assert "torch.cuda.get_device_name(" not in before
        assert "torch.cuda.get_device_properties(" not in before

    def test_is_available_is_not_used(self):
        # torch's own docs: is_available() poisons fork unless
        # PYTORCH_NVML_BASED_CUDA_CHECK=1, because it falls through to
        # _cuda_getDeviceCount() and calls cuInit. This is what actually
        # broke the first two-GPU run.
        before = self._finetune_source_before_fit()
        assert "torch.cuda.is_available(" not in before

    def test_nvml_check_is_requested_before_torch_is_imported(self):
        before = self._finetune_source_before_fit()
        env_at = before.index("PYTORCH_NVML_BASED_CUDA_CHECK")
        import_at = before.index("import torch")
        assert env_at < import_at

    def test_initialised_cuda_is_caught_before_fit(self):
        before = self._finetune_source_before_fit()
        assert "torch.cuda.is_initialized()" in before
        assert "ddp_spawn" in before  # the documented escape hatch

    def test_device_count_is_how_gpus_are_counted(self):
        before = self._finetune_source_before_fit()
        assert "torch.cuda.device_count()" in before

    def test_model_is_not_moved_to_gpu_before_fit(self):
        # Lightning moves the module; doing it by hand would initialise CUDA
        # in the parent and break the fork.
        before = self._finetune_source_before_fit()
        assert ".to(device)" not in before
        assert ".cuda()" not in before


class TestCheckpointSurvivesReportingErrors:
    """A run's output is the checkpoint; reporting is decoration.

    The first two-GPU run trained for its full twelve epochs and then lost
    the export to a FileNotFoundError in the metrics read, because the read
    sat in front of it.
    """

    def _finetune_source(self):
        source = (config.PROJECT_ROOT / "tools" / "make_notebook.py").read_text()
        start = source.index("def build_finetune_notebook")
        return source[start:source.index("def build_export_notebook")]

    def test_export_precedes_the_metrics_read(self):
        finetune = self._finetune_source()
        assert (finetune.index("export_from_checkpoint(")
                < finetune.index('"metrics.csv"'))

    def test_metrics_read_cannot_abort_the_cell(self):
        finetune = self._finetune_source()
        read_at = finetune.index('"metrics.csv"')
        # The read must sit inside a try block opened before it.
        assert "try:" in finetune[:read_at]
        assert "except Exception" in finetune[read_at:]

    def test_logger_version_is_pinned(self):
        # Unpinned, the child resolves version_0 during fit and the parent
        # then reports version_1, an empty directory.
        finetune = self._finetune_source()
        assert 'version="metrics"' in finetune


class TestGoldIsNeverMonitored:
    def test_notebook_monitors_val_macro_only(self):
        source = (config.PROJECT_ROOT / "tools" / "make_notebook.py").read_text()
        start = source.index("def build_finetune_notebook")
        end = source.index("def build_export_notebook")
        finetune = source[start:end]
        assert 'monitor="val_macro"' in finetune
        assert 'monitor="gold_macro"' not in finetune


class TestEpochReportSeesTheMetrics:
    """The reporter must read metrics after the module has logged them.

    Lightning runs callback hooks *before* the LightningModule hook of the
    same name, so a callback reading `val_macro` in `on_validation_epoch_end`
    gets an empty dict and prints nan for every epoch of a healthy run. This
    runs a real one-epoch Trainer rather than inspecting source, because the
    ordering is Lightning's to define and could change under us.
    """

    def test_report_captures_logged_metrics(self, capsys):
        from torch.utils.data import DataLoader, TensorDataset

        class Tiny(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(2, 1)

            def training_step(self, batch, _):
                return self.layer(batch[0]).mean()

            def validation_step(self, batch, _, dataloader_idx=0):
                pass

            def on_validation_epoch_end(self):
                self.log("val_macro", 0.42)
                self.log("gold_macro", 0.37)

            def configure_optimizers(self):
                return torch.optim.SGD(self.parameters(), lr=0.1)

        report = knee_lightning.EpochReport()
        data = DataLoader(TensorDataset(torch.randn(8, 2)), batch_size=4)
        pl.Trainer(max_epochs=1, accelerator="cpu", devices=1, logger=False,
                   enable_progress_bar=False, enable_model_summary=False,
                   num_sanity_val_steps=0, callbacks=[report]).fit(
                       Tiny(), data, data)

        assert len(report.rows) == 1, "the reporter never fired"
        row = report.rows[0]
        assert row["val_macro"] == pytest.approx(0.42)
        assert row["gold_macro"] == pytest.approx(0.37)
        assert "nan" not in capsys.readouterr().out.lower()

    def test_reads_on_validation_end_not_epoch_end(self):
        # The specific hook matters; asserting it keeps a well-meaning rename
        # from silently reintroducing the nan log.
        assert hasattr(knee_lightning.EpochReport, "on_validation_end")
        assert "on_validation_epoch_end" not in vars(knee_lightning.EpochReport)


class TestCheckpointIsTheBestNotTheLast:
    """Every result in this project rests on this being true.

    `save_top_k=1` with a monitor writes only when the metric improves, so a
    static filename holds the best epoch rather than the most recent one --
    but that is Lightning's behaviour to define, not ours, so it is measured
    here rather than assumed. The metric peaks in the middle so that "best"
    and "last" are different weights, and the epoch is stamped into the
    parameters so the file identifies which one it holds.
    """

    SCORES = [0.10, 0.90, 0.50]   # peak at epoch 1 of 0, 1, 2

    def _run(self, tmp_path):
        from torch.utils.data import DataLoader, TensorDataset
        scores = self.SCORES

        class Stamped(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Linear(2, 1)

            def training_step(self, batch, _):
                return self.model(batch[0]).mean() * 0.0

            def validation_step(self, batch, _, dataloader_idx=0):
                pass

            def on_validation_epoch_end(self):
                with torch.no_grad():
                    self.model.bias.fill_(float(self.current_epoch))
                self.log("val_macro", scores[self.current_epoch])

            def configure_optimizers(self):
                return torch.optim.SGD(self.parameters(), lr=0.0)

        checkpointer = pl.callbacks.ModelCheckpoint(
            dirpath=str(tmp_path), filename="lit_fold0",
            monitor="val_macro", mode="max", save_top_k=1)
        data = DataLoader(TensorDataset(torch.randn(4, 2)), batch_size=2)
        pl.Trainer(max_epochs=len(scores), accelerator="cpu", devices=1,
                   logger=False, enable_progress_bar=False,
                   enable_model_summary=False, num_sanity_val_steps=0,
                   callbacks=[checkpointer]).fit(Stamped(), data, data)
        return checkpointer

    def test_saved_weights_are_from_the_peak_epoch(self, tmp_path):
        checkpointer = self._run(tmp_path)
        saved = torch.load(checkpointer.best_model_path, map_location="cpu",
                           weights_only=False)
        stamped = float(saved["state_dict"]["model.bias"].item())
        best_epoch = int(np.argmax(self.SCORES))
        assert stamped == best_epoch, (
            f"checkpoint holds epoch {stamped:.0f}, wanted {best_epoch} "
            f"(last was {len(self.SCORES) - 1})")
        assert float(checkpointer.best_model_score) == pytest.approx(
            max(self.SCORES))

    def test_only_one_checkpoint_is_left_behind(self, tmp_path):
        # The recovery path globs *.ckpt and takes the first hit, which is
        # only safe while exactly one file exists.
        self._run(tmp_path)
        assert len(list(tmp_path.glob("*.ckpt"))) == 1


class TestParentProcessKnowsNothing:
    """After a forked fit, the parent's callback state is mostly stale.

    Lightning's multiprocessing launcher transfers `best_model_path` back to
    the process that called `fit` and leaves `best_model_score` at None --
    its own source carries a "TODO: pass also best score". Formatting that
    None crashed a finished twelve-epoch run, after the export had been moved
    behind it.
    """

    def test_launcher_still_does_not_return_the_score(self):
        import inspect
        from lightning.pytorch.strategies.launchers.multiprocessing import (
            _MultiProcessingLauncher)
        source = inspect.getsource(
            _MultiProcessingLauncher._recover_results_in_main_process)
        assert "best_model_path" in source
        # If a future Lightning starts returning the score, this fails and the
        # notebook's workaround can be simplified rather than left as cargo.
        assert "best_model_score" not in source

    def _finetune_source(self):
        source = (config.PROJECT_ROOT / "tools" / "make_notebook.py").read_text()
        start = source.index("def build_finetune_notebook")
        return source[start:source.index("def build_export_notebook")]

    def test_notebook_never_formats_the_score(self):
        finetune = self._finetune_source()
        assert "best_model_score:" not in finetune
        assert "best_model_score:.3f" not in finetune

    def test_checkpoint_path_is_verified_against_the_filesystem(self):
        finetune = self._finetune_source()
        assert "os.path.exists(checkpoint_path)" in finetune
        assert 'lit_fold{FOLD}*.ckpt' in finetune

    def test_export_still_precedes_the_metrics_read(self):
        finetune = self._finetune_source()
        assert (finetune.index("export_from_checkpoint(")
                < finetune.index('"metrics.csv"'))


class TestAuxiliaryLosses:
    """Report regression and ranking are training-time only, and the export
    the submission loads must not know they existed."""

    FEATURE_DIM = 16
    REPORT_DIM = 8

    class ConvBackbone(torch.nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, dim, 3, padding=1)

        def forward(self, x):
            return self.conv(x).mean(dim=(2, 3))

    def _module(self, **kwargs):
        torch.manual_seed(0)
        module = knee_lightning.KneeLitModule(
            backbone=self.ConvBackbone(self.FEATURE_DIM),
            feature_dim=self.FEATURE_DIM, total_steps=10, **kwargs)
        # Unit tests run outside a Trainer, where self.log only warns; drop
        # it so the assertions are about the losses, not the logger.
        module.log = lambda *args, **kw: None
        return module

    def _batch(self, batch=2, with_report=True, seed=1):
        torch.manual_seed(seed)
        out = {
            "volumes": torch.randint(0, 255, (batch, 2, 4, 8, 8),
                                     dtype=torch.uint8),
            "meta": torch.rand(batch, 2, 7),
            "mask": torch.ones(batch, 2, dtype=torch.bool),
            "targets": torch.rand(batch, 12),
            "weights": torch.ones(batch, 12),
        }
        if with_report:
            out["report"] = torch.randn(batch, self.REPORT_DIM)
            out["report_mask"] = torch.ones(batch, dtype=torch.bool)
        return out

    def test_both_losses_train_and_reach_their_parameters(self):
        module = self._module(report_dim=self.REPORT_DIM,
                              lambda_report=0.2, lambda_rank=0.2)
        module.training_step(self._batch(seed=1), 0)   # fills the rank queue
        loss = module.training_step(self._batch(seed=2), 1)
        assert torch.isfinite(loss)
        loss.backward()
        assert module.report_head[1].weight.grad is not None
        assert module.model.head[2].weight.grad is not None

    def test_batch_without_reports_still_uses_the_head(self):
        # DDP errors on any parameter that misses a backward pass, so the
        # head must join the graph even when no study has a report.
        module = self._module(report_dim=self.REPORT_DIM, lambda_report=0.2)
        loss = module.training_step(self._batch(with_report=False), 0)
        loss.backward()
        assert module.report_head[1].weight.grad is not None

    def test_rank_queue_caps_and_resets(self):
        module = self._module(lambda_rank=0.2, rank_queue_size=4)
        for step in range(5):
            module.training_step(self._batch(seed=step), step)
        held = sum(len(q["scores"]) for q in module._rank_queue)
        assert held <= 4
        module.on_train_epoch_start()
        assert module._rank_queue == []

    def test_defaults_leave_the_loss_untouched(self):
        plain = self._module()
        assert plain.report_head is None
        plain.eval()  # dropout off, so the two passes below are identical
        batch = self._batch(with_report=False)
        loss = plain.training_step(batch, 0)
        with torch.no_grad():
            logits = plain(batch["volumes"], batch["meta"], batch["mask"])
        from src.model import net
        expected = net.weighted_bce(logits, batch["targets"], batch["weights"])
        assert torch.allclose(loss.detach(), expected)

    def test_export_drops_the_auxiliary_head(self, tmp_path):
        module = self._module(report_dim=self.REPORT_DIM,
                              lambda_report=0.2, lambda_rank=0.2)
        ckpt = str(tmp_path / "lit.ckpt")
        torch.save({"state_dict": module.state_dict()}, ckpt)
        out = str(tmp_path / "convnext_fold0.pt")
        knee_lightning.export_from_checkpoint(ckpt, out)

        from src.model import net
        exported = torch.load(out, map_location="cpu")
        assert not any("report_head" in key for key in exported)
        fresh = net.KneeNet(self.ConvBackbone(self.FEATURE_DIM),
                            self.FEATURE_DIM)
        fresh.load_state_dict(exported)
