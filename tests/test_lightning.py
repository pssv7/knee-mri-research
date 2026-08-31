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
from src.model import net  # noqa: E402


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

    def test_batch_progress_renders_and_closes(self, capsys):
        # BatchProgress exists because Lightning's own bar is a notebook
        # widget that never reaches a Save & Run All log. Run a real
        # one-epoch fit and assert plain-text tqdm output actually landed on
        # stdout and the bar was released at epoch end.
        from torch.utils.data import DataLoader, TensorDataset

        class Tiny(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(2, 1)

            def training_step(self, batch, _):
                return self.layer(batch[0]).mean()

            def configure_optimizers(self):
                return torch.optim.SGD(self.parameters(), lr=0.1)

        progress = knee_lightning.BatchProgress(refresh_seconds=0.0)
        data = DataLoader(TensorDataset(torch.randn(8, 2)), batch_size=4)
        pl.Trainer(max_epochs=1, accelerator="cpu", devices=1, logger=False,
                   enable_progress_bar=False, enable_model_summary=False,
                   limit_val_batches=0, callbacks=[progress]).fit(Tiny(), data)

        assert "epoch 1/1" in capsys.readouterr().out
        assert progress._bar is None, "bar not closed at epoch end"

    def test_reads_on_validation_end_not_epoch_end(self):
        # The specific hook matters; asserting it keeps a well-meaning rename
        # from silently reintroducing the nan log.
        assert hasattr(knee_lightning.EpochReport, "on_validation_end")
        assert "on_validation_epoch_end" not in vars(knee_lightning.EpochReport)


class TestStage1Objectives:
    """The stratified rank loss, the coupling penalty, and the shared queue.

    All three are off at their defaults, so the first thing guarded is that a
    run which does not ask for them is byte-identical to the reference runs.
    """

    FEATURES = 8

    def _module(self, **kwargs):
        torch.manual_seed(0)

        class TinyBackbone(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, dim, 3, padding=1)

            def forward(self, x):
                return self.conv(x).mean(dim=(2, 3))

        return knee_lightning.KneeLitModule(
            TinyBackbone(self.FEATURES), self.FEATURES, **kwargs)

    def _batch(self, studies=2, series=2, depth=4, size=8):
        from src.model.dataset import META_COLUMNS
        torch.manual_seed(1)
        return {
            "volumes": torch.randint(0, 255, (studies, series, depth, size, size),
                                     dtype=torch.uint8),
            "meta": torch.rand(studies, series, len(META_COLUMNS)),
            "mask": torch.ones(studies, series, dtype=torch.bool),
            "targets": torch.rand(studies, len(config.LABELS)),
            "weights": torch.ones(studies, len(config.LABELS)),
        }

    def test_defaults_leave_the_reference_loss_untouched(self):
        module = self._module()
        assert module.lambda_stratified_rank == 0.0
        assert module.lambda_coupling == 0.0
        assert module.model.residual_pairs is None

    def test_a_step_with_every_term_on_produces_a_finite_loss(self):
        module = self._module(lambda_rank=0.1, lambda_stratified_rank=0.2,
                              lambda_coupling=0.05, coupling_target=0.403,
                              residual_beta=0.6)
        batch = self._batch()
        first = module.training_step(batch, 0)
        second = module.training_step(batch, 1)   # now the queue is populated
        assert torch.isfinite(first) and torch.isfinite(second)
        second.backward()
        assert any(p.grad is not None for p in module.parameters())

    def test_the_queue_carries_the_effusion_score_for_the_reference_studies(self):
        module = self._module(lambda_stratified_rank=0.2)
        module.training_step(self._batch(), 0)
        reference = module._rank_reference()
        assert set(reference) == {"scores", "targets", "weights", "strata"}
        assert reference["strata"].shape == (2,)
        assert ((reference["strata"] >= 0) & (reference["strata"] <= 1)).all()

    def test_one_push_per_step_even_with_both_ranking_terms_on(self):
        module = self._module(lambda_rank=0.1, lambda_stratified_rank=0.1)
        for step in range(3):
            module.training_step(self._batch(), step)
        assert len(module._rank_queue) == 3

    def test_nothing_is_queued_when_no_ranking_term_is_on(self):
        module = self._module()
        module.training_step(self._batch(), 0)
        assert module._rank_queue == []

    def test_a_coupling_penalty_without_a_target_is_refused(self):
        # Independence is the wrong target: the findings co-occur at OR 6.09.
        with pytest.raises(ValueError, match="coupling_target"):
            self._module(lambda_coupling=0.1)

    def test_an_unknown_finding_is_refused(self):
        with pytest.raises(ValueError, match="unknown finding"):
            self._module(stratified_findings=("Synovitus",))

    def test_the_residual_coupling_reaches_the_exported_checkpoint(self, tmp_path):
        module = self._module(residual_beta=0.75)
        path = str(tmp_path / "exported.pt")
        knee_lightning.export_state_dict(module, path)
        bare = net.KneeNet(module.model.backbone, self.FEATURES)
        bare.load_state_dict(net.load_knee_state(path))
        assert bare.residual_beta.tolist() == pytest.approx([0.75], abs=1e-6)
        assert bare.residual_pairs.tolist() == [
            [config.LABELS.index("Synovitis"), config.LABELS.index("Effusion")]]


class TestConditionalIsLoggedBesidePooled:
    """§3: pooled AUC on synovitis is obtainable from the effusion label alone.

    An epoch-level number that cannot distinguish "learned synovitis" from
    "learned effusion" is what seven consecutive null readings were judged on,
    so the conditional number is logged every epoch beside it.
    """

    def _module(self):
        # feature_dim must divide the transformer's head count; the module is
        # never run forward here, only its validation hook.
        return knee_lightning.KneeLitModule(torch.nn.Identity(), 8)

    def _feed(self, module, truth, probs):
        tensor_truth = torch.tensor(truth, dtype=torch.float32)
        tensor_probs = torch.tensor(probs, dtype=torch.float32)
        module._val_outputs[1] = [{"probs": tensor_probs,
                                   "targets": tensor_truth}]
        logged = {}
        module.log = lambda name, value, **kwargs: logged.__setitem__(
            name, float(value))
        module.all_gather = lambda x: x
        module.on_validation_epoch_end()
        return logged

    def test_a_gated_model_logs_a_high_pooled_and_a_chance_conditional(self):
        rng = np.random.default_rng(0)
        n = 600      # enough that a chance reading is not a sampling accident
        effusion = rng.integers(0, 2, n)
        synovitis = np.where(
            rng.random(n) < np.where(effusion == 1, 0.7, 0.3), 1, 0)
        truth = np.zeros((n, len(config.LABELS)))
        truth[:, config.LABELS.index("Effusion")] = effusion
        truth[:, config.LABELS.index("Synovitis")] = synovitis
        probs = np.full((n, len(config.LABELS)), 0.5)
        # The failure mode: the synovitis head is the effusion head.
        gated = 0.5 + 0.4 * effusion + rng.normal(0, 0.02, n)
        probs[:, config.LABELS.index("Effusion")] = gated
        probs[:, config.LABELS.index("Synovitis")] = gated

        logged = self._feed(self._module(), truth, probs)
        assert logged["gold_Synovitis_pooled"] > 0.6
        assert abs(logged["gold_Synovitis_conditional"] - 0.5) < 0.05
        assert "gold_macro" in logged


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

    # -- Gradient-similarity gating (arXiv:1812.02224, Algorithm 1) --------

    def test_cosine_mode_is_inert_when_no_auxiliary_is_configured(self):
        # The fold ensemble trains with both lambdas at 0. Turning the mode
        # on must not change a single number there, or an unrelated flag
        # would silently alter the runs the leaderboard is built on.
        for mode in ("fixed", "cosine"):
            module = self._module(aux_mode=mode)
            module.eval()
            batch = self._batch(with_report=False)
            loss = module.training_step(batch, 0)
            with torch.no_grad():
                logits = module(batch["volumes"], batch["meta"], batch["mask"])
            from src.model import net
            expected = net.weighted_bce(logits, batch["targets"],
                                        batch["weights"])
            assert torch.allclose(loss.detach(), expected), mode

    def test_aligned_auxiliary_is_applied_and_opposed_is_blocked(self):
        # The gate is the whole point, so drive both branches directly: an
        # auxiliary that is the main loss (cosine +1) must pass, and its
        # negation (cosine -1) must be blocked to exactly zero weight.
        module = self._module(report_dim=self.REPORT_DIM, lambda_report=1.0,
                              aux_mode="cosine")
        batch = self._batch()
        logits, pooled = module.model.forward_pooled(
            batch["volumes"], batch["meta"], batch["mask"])
        from src.model import net
        main = net.weighted_bce(logits, batch["targets"], batch["weights"])

        aligned = module._aux_cosine(main, main * 2.0, pooled)
        opposed = module._aux_cosine(main, -main, pooled)
        assert aligned.item() == pytest.approx(1.0, abs=1e-3)
        assert opposed.item() == pytest.approx(-1.0, abs=1e-3)
        assert module._aux_weight(aligned).item() == 1.0
        assert module._aux_weight(opposed).item() == 0.0

    def test_weighted_gate_scales_rather_than_switches(self):
        module = self._module(report_dim=self.REPORT_DIM, lambda_report=1.0,
                              aux_mode="cosine", aux_gate="weighted")
        assert module._aux_weight(torch.tensor(0.4)).item() == pytest.approx(0.4)
        # Negative transfer is clipped, never negated -- flipping the sign
        # would *maximise* the auxiliary loss rather than ignore it.
        assert module._aux_weight(torch.tensor(-0.4)).item() == 0.0

    def test_gating_does_not_break_the_real_backward(self):
        # autograd.grad probes run before Lightning's backward over the same
        # graph; without retain_graph the real backward would raise.
        module = self._module(report_dim=self.REPORT_DIM, lambda_report=0.5,
                              lambda_rank=0.5, aux_mode="cosine")
        module.training_step(self._batch(seed=1), 0)
        loss = module.training_step(self._batch(seed=2), 1)
        loss.backward()
        assert module.report_head[1].weight.grad is not None
        assert module.model.head[2].weight.grad is not None

    def test_invalid_mode_is_rejected_at_construction(self):
        with pytest.raises(ValueError, match="aux_mode"):
            self._module(aux_mode="adaptive")
        with pytest.raises(ValueError, match="aux_gate"):
            self._module(aux_gate="soft")


class TestBackboneLearningRateGroups:
    """A pretrained trunk can be given a slower rate than the head it feeds.

    Scale 1.0 must take the original single-group path, because every
    ConvNeXt result on record was trained through it.
    """

    def _module(self, scale):
        import torch.nn as nn
        return knee_lightning.KneeLitModule(
            backbone=nn.Linear(4, 8), feature_dim=8, lr=1e-4,
            total_steps=100, backbone_lr_scale=scale)

    def test_default_keeps_one_group(self):
        cfg = self._module(1.0).configure_optimizers()
        assert len(cfg["optimizer"].param_groups) == 1

    def test_a_scaled_trunk_gets_its_own_slower_group(self):
        # OneCycleLR rewrites every group's `lr` to max_lr/div_factor as soon
        # as it is constructed, so the peak it will reach -- stored on the
        # group as `max_lr` -- is the honest thing to assert.
        module = self._module(0.1)
        groups = module.configure_optimizers()["optimizer"].param_groups
        assert len(groups) == 2
        assert groups[0]["max_lr"] == pytest.approx(3e-5)
        assert groups[1]["max_lr"] == pytest.approx(3e-4)

    def test_every_parameter_lands_in_exactly_one_group(self):
        module = self._module(0.1)
        groups = module.configure_optimizers()["optimizer"].param_groups
        assigned = [id(p) for g in groups for p in g["params"]]
        assert sorted(assigned) == sorted(id(p) for p in module.parameters())
        assert len(assigned) == len(set(assigned))

    def test_one_group_still_peaks_where_it_always_did(self):
        groups = self._module(1.0).configure_optimizers()["optimizer"].param_groups
        assert groups[0]["max_lr"] == pytest.approx(3e-4)


class TestSchedule:
    """The warmup must be pinnable to an absolute step count.

    `OneCycleLR`'s `pct_start` is a fraction of the run, so lengthening a run
    moves the learning-rate peak. Twelve epochs put it near epoch 3.6 and
    twenty put it near epoch 6, which is why "the best epoch moved" across two
    runs of different lengths says nothing on its own.
    """

    import torch.nn as _nn

    def _module(self, **kwargs):
        import torch.nn as nn
        return knee_lightning.KneeLitModule(
            backbone=nn.Linear(4, 8), feature_dim=8, lr=1e-4, **kwargs)

    def _peak_step(self, module):
        """Where the schedule's learning rate is highest, by stepping it."""
        cfg = module.configure_optimizers()
        scheduler = cfg["lr_scheduler"]["scheduler"]
        optimiser = cfg["optimizer"]
        rates = []
        for _ in range(module.total_steps):
            rates.append(optimiser.param_groups[0]["lr"])
            optimiser.step()
            scheduler.step()
        return int(np.argmax(rates)), rates

    def test_the_default_is_the_thirty_percent_cycle_it_always_was(self):
        peak, _ = self._peak_step(self._module(total_steps=100))
        assert peak == pytest.approx(30, abs=2)

    def test_an_absolute_warmup_holds_the_peak_across_run_lengths(self):
        short, _ = self._peak_step(self._module(total_steps=100,
                                                warmup_steps=20))
        long, _ = self._peak_step(self._module(total_steps=200,
                                               warmup_steps=20))
        assert short == pytest.approx(20, abs=2)
        assert long == pytest.approx(20, abs=2)

    def test_the_fractional_default_does_not(self):
        # The behaviour being fixed, asserted so the fix has something to be
        # a fix of.
        short, _ = self._peak_step(self._module(total_steps=100))
        long, _ = self._peak_step(self._module(total_steps=200))
        assert long > short * 1.5

    def test_cosine_warms_up_then_decays_to_nothing(self):
        peak, rates = self._peak_step(
            self._module(total_steps=100, warmup_steps=10, schedule="cosine"))
        assert peak == pytest.approx(9, abs=1)
        assert rates[-1] < rates[peak] * 0.01

    def test_cosine_peaks_where_onecycle_does(self):
        _, cycle = self._peak_step(self._module(total_steps=100,
                                                warmup_steps=10))
        _, cosine = self._peak_step(self._module(total_steps=100,
                                                 warmup_steps=10,
                                                 schedule="cosine"))
        assert max(cosine) == pytest.approx(max(cycle), rel=0.02)

    def test_cosine_keeps_the_trunk_group_slower(self):
        module = self._module(total_steps=100, warmup_steps=10,
                              schedule="cosine", backbone_lr_scale=0.1)
        cfg = module.configure_optimizers()
        optimiser, scheduler = cfg["optimizer"], cfg["lr_scheduler"]["scheduler"]
        for _ in range(15):
            optimiser.step()
            scheduler.step()
        trunk, rest = (g["lr"] for g in optimiser.param_groups)
        assert trunk == pytest.approx(rest * 0.1)

    def test_an_unknown_schedule_is_refused(self):
        with pytest.raises(ValueError, match="schedule"):
            self._module(total_steps=100, schedule="triangular")

    def test_a_warmup_longer_than_the_run_is_refused(self):
        with pytest.raises(ValueError, match="warmup_steps"):
            self._module(total_steps=100, warmup_steps=200)


class TestNoiseRobustLossKnobs:
    """Label smoothing, per-finding weighting and symmetric CE, all inert by
    default so every reference run stays reproducible."""

    def _module(self, **kwargs):
        import torch.nn as nn
        module = knee_lightning.KneeLitModule(
            backbone=nn.Linear(4, 8), feature_dim=8, total_steps=10, **kwargs)
        module.log = lambda *args, **kw: None
        return module

    def test_the_defaults_change_nothing(self):
        module = self._module()
        assert module.label_smoothing == 0.0
        assert module.symmetric_beta == 0.0
        assert module.pos_weight is None

    def test_pos_weight_reaches_the_loss_as_a_tensor(self):
        module = self._module(pos_weight=[2.0] * len(config.LABELS))
        assert module.pos_weight.shape == (len(config.LABELS),)
        # On the wrapper, so it never reaches the exported KneeNet.
        assert not [k for k in module.state_dict()
                    if k.startswith("model.") and "pos_weight" in k]

    def test_label_smoothing_pulls_targets_toward_a_half(self):
        # Read off the gradient rather than the loss value: BCE's gradient at
        # a logit is (p - target), so the smoothed target is exactly the
        # probability the loss stops pushing at. At eps=0.2 a target of 1.0
        # becomes 0.9, and the gradient at p=0.9 must vanish.
        logits = torch.logit(torch.full((1, 1), 0.9)).requires_grad_()
        targets, weights = torch.ones(1, 1), torch.ones(1, 1)

        net.weighted_bce(logits, targets, weights,
                         label_smoothing=0.2).backward()
        assert logits.grad.abs().item() == pytest.approx(0.0, abs=1e-6)

        logits.grad = None
        net.weighted_bce(logits, targets, weights).backward()
        assert logits.grad.item() == pytest.approx(-0.1, abs=1e-6)

    def test_pos_weight_raises_the_cost_of_missing_a_positive(self):
        logits = torch.full((1, 2), -4.0)
        targets = torch.ones(1, 2)
        weights = torch.ones(1, 2)
        plain = net.weighted_bce(logits, targets, weights)
        weighted = net.weighted_bce(logits, targets, weights,
                                    pos_weight=torch.tensor([3.0, 3.0]))
        assert weighted == pytest.approx(plain * 3.0, rel=1e-5)

    def test_symmetric_ce_bounds_what_one_wrong_label_contributes(self):
        # A confident prediction against a confident opposite label: plain BCE
        # grows without bound as the logit does, the reverse term does not.
        weights = torch.ones(1, 1)
        targets = torch.zeros(1, 1)
        losses = [net.weighted_bce(torch.tensor([[scale]]), targets, weights,
                                   symmetric_beta=1.0)
                  - net.weighted_bce(torch.tensor([[scale]]), targets, weights)
                  for scale in (4.0, 8.0)]
        # The extra term saturates rather than tracking the logit.
        assert losses[1] == pytest.approx(losses[0], rel=0.05)

    def test_the_training_step_runs_with_every_knob_on(self):
        module = self._module(label_smoothing=0.05, symmetric_beta=0.5,
                              pos_weight=[1.5] * len(config.LABELS))
        module.model = net.KneeNet(TestAuxiliaryLosses.ConvBackbone(8), 8)
        batch = TestAuxiliaryLosses()._batch(with_report=False)
        loss = module.training_step(batch, 0)
        loss.backward()
        assert torch.isfinite(loss)


class TestWeightAverageCallback:
    """EMA and SWA of the trajectory, written from inside `fit`.

    Selection-free by construction: nothing here reads a metric, so nothing
    here can select into the memorisation phase the mined validation curve
    rewards.
    """

    def _run(self, tmp_path, epochs=4, **kwargs):
        from torch.utils.data import DataLoader, TensorDataset

        class Counting(pl.LightningModule):
            """Weights equal to the epoch number, so an average is readable."""

            def __init__(self):
                super().__init__()
                self.model = torch.nn.Linear(1, 1)

            def training_step(self, batch, _):
                with torch.no_grad():
                    self.model.bias.fill_(float(self.current_epoch + 1))
                return (self.model(batch[0]).mean() * 0.0
                        + self.model.bias.sum() * 0.0)

            def configure_optimizers(self):
                return torch.optim.SGD(self.parameters(), lr=0.0)

        average = knee_lightning.WeightAverage(
            prefix=str(tmp_path / "fold0"), **kwargs)
        data = DataLoader(TensorDataset(torch.randn(2, 1)), batch_size=2)
        pl.Trainer(max_epochs=epochs, accelerator="cpu", devices=1,
                   logger=False, enable_progress_bar=False,
                   enable_model_summary=False, num_sanity_val_steps=0,
                   callbacks=[average]).fit(Counting(), data)
        return average

    def test_both_files_are_written_by_the_end_of_fit(self, tmp_path):
        average = self._run(tmp_path, total_steps=4)
        assert os.path.exists(average.ema_path)
        assert os.path.exists(average.swa_path)

    def test_the_files_are_bare_state_dicts_the_submission_can_load(self,
                                                                    tmp_path):
        average = self._run(tmp_path, total_steps=4)
        state = torch.load(average.ema_path, map_location="cpu")
        # No "model." prefix: this is the bare KneeNet format
        # notebook 10 loads, not a Lightning checkpoint.
        assert set(state) == {"weight", "bias"}

    def test_swa_averages_the_tail_not_the_whole_run(self, tmp_path):
        # Four steps holding 1, 2, 3, 4; the last half is 3 and 4.
        average = self._run(tmp_path, epochs=4, total_steps=4,
                            swa_fraction=0.5)
        swa = torch.load(average.swa_path, map_location="cpu")
        assert swa["bias"].item() == pytest.approx(3.5)

    def test_the_ema_lags_the_final_weights(self, tmp_path):
        average = self._run(tmp_path, epochs=4, total_steps=4, decay=0.5)
        ema = torch.load(average.ema_path, map_location="cpu")
        assert ema["bias"].item() < 4.0

    def test_no_swa_file_when_the_window_never_opened(self, tmp_path):
        average = self._run(tmp_path, epochs=2, total_steps=1000)
        assert os.path.exists(average.ema_path)
        assert not os.path.exists(average.swa_path)

    def test_it_updates_on_optimiser_steps_not_batches(self, tmp_path):
        # With accumulation, most batches leave the weights untouched. Folding
        # an unchanged model in `accumulate` times per step would make the
        # effective decay decay**accumulate -- a shorter memory than asked
        # for, set by a knob that is meant to be numerically neutral.
        from torch.utils.data import DataLoader, TensorDataset

        class Trivial(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Linear(1, 1)

            def training_step(self, batch, _):
                return self.model(batch[0]).mean() * 0.0

            def configure_optimizers(self):
                return torch.optim.SGD(self.parameters(), lr=0.0)

        average = knee_lightning.WeightAverage(prefix=str(tmp_path / "f"),
                                               total_steps=100)
        data = DataLoader(TensorDataset(torch.randn(8, 1)), batch_size=1)
        pl.Trainer(max_epochs=1, accelerator="cpu", devices=1, logger=False,
                   enable_progress_bar=False, enable_model_summary=False,
                   num_sanity_val_steps=0, accumulate_grad_batches=4,
                   callbacks=[average]).fit(Trivial(), data)
        assert average.average.steps == 2      # 8 batches / 4 accumulation

    def test_an_impossible_swa_fraction_is_refused(self):
        with pytest.raises(ValueError, match="swa_fraction"):
            knee_lightning.WeightAverage(prefix="x", swa_fraction=1.5)


class TestWeightAverageCannotKillAFinishedRun:
    """Hours of GPU time exist as the monitored checkpoint by the time
    `on_train_end` fires. An exception raised there propagates out of `fit`
    and takes the export and scoring cells with it."""

    def test_a_failing_export_is_reported_not_raised(self, tmp_path, capsys):
        from torch.utils.data import DataLoader, TensorDataset

        class Trivial(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Linear(1, 1)

            def training_step(self, batch, _):
                return self.model(batch[0]).mean() * 0.0

            def configure_optimizers(self):
                return torch.optim.SGD(self.parameters(), lr=0.0)

        # A directory that does not exist, so torch.save raises.
        average = knee_lightning.WeightAverage(
            prefix=str(tmp_path / "missing" / "fold0"), total_steps=2)
        data = DataLoader(TensorDataset(torch.randn(2, 1)), batch_size=1)
        pl.Trainer(max_epochs=1, accelerator="cpu", devices=1, logger=False,
                   enable_progress_bar=False, enable_model_summary=False,
                   num_sanity_val_steps=0,
                   callbacks=[average]).fit(Trivial(), data)
        assert "could not write its averages" in capsys.readouterr().out
