#!/usr/bin/env python3
"""Create deterministic class-wise training-box removal splits for Table 3 experiments."""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True, help="Source training-label directory.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rates", type=int, nargs="+", default=(30, 50, 70, 90))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Allow replacement of existing generated splits.")
    return parser.parse_args()


def load_labels(root):
    records = {}
    by_class = defaultdict(list)
    for path in sorted(root.rglob("*.txt")):
        relative = path.relative_to(root)
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        records[relative] = lines
        for line_index, line in enumerate(lines):
            class_id = int(float(line.split(maxsplit=1)[0]))
            by_class[class_id].append((relative, line_index))
    if not records:
        raise FileNotFoundError(f"No YOLO label files found under {root}")
    return records, by_class


def main():
    args = parse_args()
    rates = sorted(set(args.rates))
    if any(rate < 0 or rate > 100 for rate in rates):
        raise ValueError("Removal rates must be between 0 and 100")
    records, by_class = load_labels(args.labels)

    shuffled = {}
    for class_id, boxes in by_class.items():
        boxes = boxes.copy()
        random.Random(args.seed + class_id).shuffle(boxes)
        shuffled[class_id] = boxes

    for rate in rates:
        output_dir = args.output_root / f"labels_missing{rate}"
        if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
            raise FileExistsError(f"{output_dir} is not empty; pass --force to replace generated text files")
        removed = set()
        for boxes in shuffled.values():
            removed.update(boxes[: round(len(boxes) * rate / 100)])
        for relative, lines in records.items():
            destination = output_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            kept = [line for index, line in enumerate(lines) if (relative, index) not in removed]
            destination.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
        print(f"{rate}%: removed {len(removed)} boxes -> {output_dir}")


if __name__ == "__main__":
    main()
