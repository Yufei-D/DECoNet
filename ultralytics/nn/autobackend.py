# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""PyTorch inference for DECoNet checkpoints."""
from pathlib import Path
import torch
import torch.nn as nn

def check_class_names(names):
    if isinstance(names, list):
        names = dict(enumerate(names))
    names = {int(k): str(v) for k, v in names.items()}
    if set(names) != set(range(len(names))):
        raise ValueError("Class indices must be contiguous and start at zero.")
    return names

class AutoBackend(nn.Module):
    """Load a PyTorch detector and expose its inference metadata."""
    @torch.no_grad()
    def __init__(self, weights, device=torch.device("cpu"), data=None, fp16=False, fuse=True, verbose=True):
        super().__init__()
        if isinstance(weights, nn.Module):
            model = weights.to(device)
            if fuse:
                model = model.fuse(verbose=verbose)
        else:
            if Path(weights).suffix != ".pt":
                raise ValueError("Expected a PyTorch .pt checkpoint.")
            from ultralytics.nn.tasks import attempt_load_one_weight
            model, _ = attempt_load_one_weight(weights, device=device, fuse=fuse)
        self.fp16 = bool(fp16 and device.type == "cuda")
        self.model = (model.half() if self.fp16 else model.float()).eval()
        self.device = device
        self.stride = max(int(model.stride.max()), 32)
        self.names = check_class_names(model.names)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def forward(self, im, augment=False, visualize=False, embed=None):
        return self.model(im.half() if self.fp16 else im.float(), augment=augment, visualize=visualize, embed=embed)

    def warmup(self, imgsz=(1, 3, 640, 640)):
        if self.device.type == "cuda":
            self.forward(torch.empty(*imgsz, device=self.device, dtype=torch.float16 if self.fp16 else torch.float32))
