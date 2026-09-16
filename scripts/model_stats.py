#!/usr/bin/env python3
"""Report DECoNet inference parameters and FLOPs at a given input size."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops


DEFAULT_MODEL = ROOT / "configs" / "models" / "yolo12n-deconet.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--weights", type=Path)
    source.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def active_inference_parameters(model, imgsz):
    active = set()
    hooks = []

    def mark(module, _inputs, _output):
        active.update(id(parameter) for parameter in module.parameters(recurse=False))

    for module in model.modules():
        hooks.append(module.register_forward_hook(mark))
    try:
        parameter = next(model.parameters())
        with torch.inference_mode():
            model(torch.zeros(1, 3, imgsz, imgsz, device=parameter.device, dtype=parameter.dtype))
    finally:
        for hook in hooks:
            hook.remove()
    return sum(parameter.numel() for parameter in model.parameters() if id(parameter) in active)


def main():
    args = parse_args()
    source = args.weights or args.model
    model = YOLO(str(source)).model.eval()
    active = active_inference_parameters(model, args.imgsz)
    total = sum(parameter.numel() for parameter in model.parameters())
    report = {
        "input": [args.imgsz, args.imgsz],
        "inference_parameters": active,
        "inference_parameters_million": active / 1e6,
        "training_graph_parameters": total,
        "gflops": float(get_flops(model, args.imgsz)),
        "flop_convention": "2 FLOPs per multiply-accumulate (Ultralytics THOP convention)",
        "fused": False,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
