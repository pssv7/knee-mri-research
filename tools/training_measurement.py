"""Lightning callback for representative train/validation cost receipts."""

import json
import time
from pathlib import Path

import lightning.pytorch as pl
import torch


class TrainingMeasurement(pl.Callback):
    def __init__(self, output="training_measurement.json", memory_limit_gib=36):
        self.output = Path(output)
        self.limit = memory_limit_gib
        self.records = {"train": [], "validation": []}
        self.active = None

    def start(self, phase):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.active = phase
        self.started = time.monotonic()
        self.batches = 0

    def finish(self, trainer, phase):
        if self.active != phase:
            return
        torch.cuda.synchronize()
        record = {"epoch": trainer.current_epoch, "batches": self.batches,
                  "seconds": time.monotonic() - self.started,
                  "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                  "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
        self.records[phase].append(record)
        self.active = None
        report = {"phases": self.records, "gpu": torch.cuda.get_device_name(),
                  "precision": str(trainer.precision),
                  "accumulation": trainer.accumulate_grad_batches,
                  "training_memory_limit_gib": self.limit,
                  "training_memory_passed": bool(self.records["train"]) and all(
                      r["peak_reserved_gib"] <= self.limit for r in self.records["train"])}
        if trainer.is_global_zero:
            self.output.write_text(json.dumps(report, indent=2) + "\n")
        if phase == "train" and record["peak_reserved_gib"] > self.limit:
            raise RuntimeError("Representative training exceeded the 36-GiB campaign gate")

    def on_train_epoch_start(self, trainer, pl_module):
        self.start("train")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.batches += 1

    def on_validation_start(self, trainer, pl_module):
        if not trainer.sanity_checking:
            self.finish(trainer, "train")
            self.start("validation")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not trainer.sanity_checking:
            self.batches += 1

    def on_validation_end(self, trainer, pl_module):
        if not trainer.sanity_checking:
            self.finish(trainer, "validation")

    def on_train_epoch_end(self, trainer, pl_module):
        self.finish(trainer, "train")


def instrument(source):
    """Embed the callback without adding an external file dependency."""
    import ast
    tree = ast.parse(source)
    for pos, node in enumerate(tree.body):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_contract' for t in node.targets):
            tree.body[pos + 1:pos + 1] = ast.parse("_contract.update(microbatch=BATCH_STUDIES, accumulation=ACCUM_STEPS, precision='16-mixed')").body
            break
    callback = ast.parse(Path(__file__).read_text())
    callback.body = [n for n in callback.body if not isinstance(n, ast.FunctionDef)]
    for pos, node in enumerate(tree.body):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "trainer" for t in node.targets):
            keywords = node.value.keywords
            callbacks = next(k.value for k in keywords if k.arg == "callbacks")
            callbacks.elts.append(ast.Call(func=ast.Name(id="TrainingMeasurement", ctx=ast.Load()), args=[], keywords=[]))
            tree.body[pos:pos] = callback.body
            return ast.unparse(ast.fix_missing_locations(tree)) + "\n"
    raise ValueError("Cannot locate training instrumentation boundary")
