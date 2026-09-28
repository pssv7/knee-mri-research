"""GPU memory/compute probe; synthetic volumes do not measure AUC or data loading."""

import gc
import json
from pathlib import Path
import time

import torch
from src.model import net


def main():
    results = []
    for size in (224, 384):
        torch.manual_seed(0)
        backbone = net.build_backbone('convnext_tiny.fb_in22k_ft_in1k', pretrained=False)
        backbone.set_grad_checkpointing(True)
        model = net.KneeNet(backbone, backbone.num_features).cuda().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler('cuda')
        volumes = torch.randint(0, 256, (1, 5, 16, size, size), device='cuda', dtype=torch.uint8)
        meta = torch.zeros((1, 5, 7), device='cuda')
        mask = torch.ones((1, 5), device='cuda', dtype=torch.bool)
        target = torch.zeros((1, 12), device='cuda')
        elapsed = []
        torch.cuda.reset_peak_memory_stats()
        for step in range(5):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            start = time.monotonic()
            with torch.autocast('cuda', dtype=torch.float16):
                logits = model(volumes, meta, mask)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            if not torch.isfinite(loss):
                raise ValueError('non-finite GPU smoke loss')
            elapsed.append(time.monotonic() - start)
        results.append({'size': size, 'series': 5, 'slices': 16,
                        'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                        'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
                        'seconds_per_microbatch': sum(elapsed[1:]) / 4,
                        'finite_loss': True, 'gradient_checkpointing': True})
        del optimizer, model, backbone, volumes, meta, mask, target, scaler, logits, loss
        gc.collect()
        torch.cuda.empty_cache()
    report = {'gpu': torch.cuda.get_device_name(), 'results': results,
              'limitation': 'Synthetic maximum-series microbatch; excludes DDP, data loading, validation and checkpoint I/O. Not an AUC measurement.'}
    Path('/kaggle/working/resolution_gpu_probe.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
