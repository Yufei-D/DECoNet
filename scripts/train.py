#!/usr/bin/env python3
"""Train DECoNet with the configuration reported in the ICASSP 2027 manuscript."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO


DEFAULT_MODEL = ROOT / "configs" / "models" / "yolo12n-deconet.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Dataset YAML (the dataset is not bundled).")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--pretrained",
        default="yolov12n.pt",
        help="Official YOLOv12n weights or a local checkpoint; use 'none' for random initialization.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="runs/deconet")
    parser.add_argument("--name", default="paper")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-steps", type=int, default=26800, help="Paper run: 100 epochs x 268 steps.")
    return parser.parse_args()


def set_paper_tmg_defaults(total_steps):
    defaults = {
        "DECONET_TOTAL_STEPS": total_steps,
        "DECONET_BURN_IN": 5000,
        "DECONET_C1_TAU_START": 0.45,
        "DECONET_C1_TAU_END": 0.30,
        "DECONET_C1_TAU_M": 0.30,
        "DECONET_C1_IOU_THRESH": 0.10,
        "DECONET_C2_HIGH_START": 0.50,
        "DECONET_C2_HIGH_END": 0.40,
        "DECONET_C2_LOW_START": 0.35,
        "DECONET_C2_LOW_END": 0.25,
        "DECONET_C2_M_THRESH": 0.25,
        "DECONET_C2_WEIGHT_HIGH": 0.75,
        "DECONET_C2_WEIGHT_LOW": 0.50,
        "DECONET_C3_ALPHA": 0.10,
        "DECONET_AUX_WEIGHT": 0.25,
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, str(value))


def main():
    args = parse_args()
    set_paper_tmg_defaults(args.total_steps)

    model = YOLO(str(args.model))
    if args.pretrained.lower() != "none":
        model.load(args.pretrained)

    model.train(
        data=str(args.data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        optimizer="AdamW",
        lr0=1e-3,
        lrf=0.01,
        momentum=0.937,
        weight_decay=5e-4,
        warmup_epochs=5.0,
        amp=True,
        use_teacher_model=True,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
