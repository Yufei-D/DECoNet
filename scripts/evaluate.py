#!/usr/bin/env python3
"""Evaluate a DECoNet checkpoint and export overall and per-class detection metrics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default="0")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold used by validation.")
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--plots", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(str(args.weights))
    results = model.val(
        data=str(args.data),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        iou=args.iou,
        plots=args.plots,
        verbose=False,
    )
    box = results.box
    overall = {
        "split": args.split,
        "precision": float(box.mp),
        "recall": float(box.mr),
        "map50": float(box.map50),
        "map50_95": float(box.map),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(overall, handle, indent=2)

    class_indices = [int(x) for x in box.ap_class_index]
    names = model.names
    with (args.output / "per_class.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("class_id", "class", "precision", "recall", "map50", "map50_95"))
        writer.writeheader()
        for position, class_id in enumerate(class_indices):
            writer.writerow(
                {
                    "class_id": class_id,
                    "class": names[class_id],
                    "precision": float(box.p[position]),
                    "recall": float(box.r[position]),
                    "map50": float(box.ap50[position]),
                    "map50_95": float(box.ap[position]),
                }
            )

    print(json.dumps(overall, indent=2))
    print(f"Per-class metrics: {args.output / 'per_class.csv'}")


if __name__ == "__main__":
    main()
