"""Training-time whole-knee mirroring with labels kept.

The perturbation probe (docs/perturbation_probe.md) found the Tiny control reads
the medial compartment by anatomy, and scored against written labels a mirror
with labels kept beats the repository's medial/lateral swap by +0.028 macro
[+0.020, +0.037]. Even correctly labelled, mirrored knees cost ACL, the menisci
and MCL, the findings whose loss is model-driven. This wrapper is the one
change in the mirror screen: each training study is mirrored with probability
`probability`, every series along its plane's medial-lateral axis
(`augment.flip_volume`), with targets, weights and metadata untouched --
`is_right` stays as acquired, as in `augment.flip_laterality`. It deliberately
does not use `augment.flip_laterality`, which swaps the labels.

`augment.py` and `lightning.py` are hash-pinned by the frozen design, so the
wrapper lives here and the base dataset, including its legacy intensity
jitter, runs unchanged underneath.
"""
import numpy as np
from torch.utils.data import DataLoader, Dataset

from src.model import augment


class MirrorStudies(Dataset):
    def __init__(self, base, probability: float = 0.5):
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be in [0, 1]")
        self.base, self.probability = base, probability

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = self.base[i]
        # Worker numpy state is seeded by Lightning (seed_everything workers=True),
        # the same source the base dataset's legacy jitter draws from.
        if self.probability and np.random.random() < self.probability:
            volumes, meta = item["volumes"], item["meta"]
            item["volumes"] = np.stack([
                augment.flip_volume(volumes[s], sagittal=bool(meta[s, augment.SAGITTAL_COLUMN] > 0.5))
                for s in range(volumes.shape[0])]) if volumes.shape[0] else volumes
            item["mirrored"] = True
        else:
            item["mirrored"] = False
        return item


def mirror_loader(loader: DataLoader, probability: float = 0.5) -> DataLoader:
    """The same training loader, over mirrored studies. Settings are copied."""
    return DataLoader(MirrorStudies(loader.dataset, probability), batch_size=loader.batch_size,
                      shuffle=True, num_workers=loader.num_workers,
                      persistent_workers=loader.persistent_workers, pin_memory=loader.pin_memory,
                      collate_fn=loader.collate_fn)
