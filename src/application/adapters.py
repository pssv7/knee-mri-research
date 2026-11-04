"""Adapters reuse existing MRI code without changing its preprocessing policy."""


class CachedStudySource:
    def __init__(self, index, shards, *, duplicate_free=False):
        self.index = index
        self.shards = shards
        self.duplicate_free = duplicate_free

    def __call__(self, request):
        from src.model.dataset import load_study

        return load_study(self.index, request.study_id, self.shards,
                          duplicate_free=self.duplicate_free)


class TorchPredictor:
    """Wrap a loaded KneeNet. Caller owns model construction and recipe identity.

    Input is already prepared by the checkpoint's preprocessing recipe, including
    depth sampling. No normalization, resampling or ensemble policy is added here.
    """

    def __init__(self, model, device="cpu"):
        self.model = model.to(device).eval()
        self.device = device

    def __call__(self, prepared):
        import torch
        from src.model.dataset import collate_studies

        # Only image inputs enter inference, even if a training index was passed.
        item = {key: prepared[key] for key in ("StudyInstanceUID", "volumes", "meta")}
        item.update(targets=None, weights=None, report=None)
        batch = collate_studies([item])
        with torch.inference_mode():
            logits = self.model(*(batch[key].to(self.device)
                                  for key in ("volumes", "meta", "mask")))
            if tuple(logits.shape) != (1, 12):
                raise ValueError("model must return logits with shape (1, 12)")
            return logits.sigmoid()[0].cpu().tolist()
