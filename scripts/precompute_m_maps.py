#!/usr/bin/env python3
"""Precompute DECoNet M-maps with h_M(p) = 1 - max_k cos(z_p, c_k)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="Training-image directory.")
    parser.add_argument("--prototype-bank", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", help="Override the encoder recorded in the prototype bank.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    bank = torch.load(args.prototype_bank, map_location="cpu", weights_only=True)
    prototype_tensor = bank.get("prototypes", bank.get("normal_centroids_normalized"))
    if prototype_tensor is None:
        raise KeyError("Prototype bank has neither 'prototypes' nor 'normal_centroids_normalized'")
    prototypes = F.normalize(prototype_tensor.float(), dim=-1)
    model_name = args.model or bank.get("encoder", "facebook/dinov3-vitb16-pretrain-lvd1689m")
    stored_size = bank.get("image_size", bank.get("input_size", 1024))
    image_size = int(stored_size[0] if isinstance(stored_size, (list, tuple)) else stored_size)
    patch_size = int(bank.get("patch_size", 16))
    stored_grid = bank.get("grid_size", bank.get("patch_grid", image_size // patch_size))
    grid_size = int(stored_grid[0] if isinstance(stored_grid, (list, tuple)) else stored_grid)
    prefix_tokens = int(bank.get("prefix_tokens", 5))

    files = sorted(p for p in args.images.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise FileNotFoundError(f"No images found under {args.images}")
    duplicate_stems = len({p.stem for p in files}) != len(files)
    if duplicate_stems:
        raise ValueError("Image stems must be unique because the YOLO loader indexes M-maps by image stem")

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    prototypes = prototypes.to(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        for image_path in tqdm(files, desc="M-maps"):
            output_path = args.output_dir / f"{image_path.stem}.npy"
            if output_path.exists() and not args.overwrite:
                continue
            with Image.open(image_path) as image:
                image = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BICUBIC)
            pixels = processor(images=image, do_resize=False, do_center_crop=False, return_tensors="pt")[
                "pixel_values"
            ].to(device)
            tokens = model(pixel_values=pixels).last_hidden_state[:, prefix_tokens:]
            if tokens.shape[1] != grid_size**2:
                raise RuntimeError(f"Unexpected patch-token count {tokens.shape[1]} for {image_path}")
            tokens = F.normalize(tokens[0].float(), dim=-1)
            anomaly = 1.0 - (tokens @ prototypes.T).amax(dim=1)
            np.save(output_path, anomaly.view(grid_size, grid_size).cpu().numpy().astype(np.float32))

    print(f"M-maps written to {args.output_dir}")


if __name__ == "__main__":
    main()
