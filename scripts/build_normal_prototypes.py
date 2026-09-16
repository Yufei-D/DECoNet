#!/usr/bin/env python3
"""Build the frozen DINOv3 normal-prototype bank used by the DECoNet M-map."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal-images", type=Path, required=True, help="Directory of defect-free training images.")
    parser.add_argument("--output", type=Path, required=True, help="Output prototype-bank .pt file.")
    parser.add_argument(
        "--model",
        default="facebook/dinov3-vitb16-pretrain-lvd1689m",
        help="Hugging Face DINOv3 model name or local directory.",
    )
    parser.add_argument("--clusters", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=5, help="Full passes over the normal-image pool.")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--kmeans-batch-size", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=0, help="Optional smoke-test limit; 0 uses every image.")
    return parser.parse_args()


def image_files(root: Path) -> list[Path]:
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise FileNotFoundError(f"No images found under {root}")
    return files


def load_pixels(paths, processor, image_size):
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB").resize((image_size, image_size), Image.Resampling.BICUBIC))
    return processor(images=images, do_resize=False, do_center_crop=False, return_tensors="pt")["pixel_values"]


@torch.inference_mode()
def patch_features(model, pixels, grid_size):
    tokens = model(pixel_values=pixels).last_hidden_state
    prefix_tokens = tokens.shape[1] - grid_size**2
    if prefix_tokens < 1:
        raise RuntimeError(f"Unexpected DINO token count: {tokens.shape[1]} for a {grid_size}x{grid_size} grid")
    return tokens[:, prefix_tokens:].flatten(0, 1).float().cpu().numpy(), prefix_tokens


def main():
    args = parse_args()
    files = image_files(args.normal_images)
    if args.max_images:
        files = files[: args.max_images]
    if args.image_size % 16:
        raise ValueError("--image-size must be divisible by the ViT-B/16 patch size")

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    grid_size = args.image_size // 16
    kmeans = MiniBatchKMeans(
        n_clusters=args.clusters,
        batch_size=args.kmeans_batch_size,
        max_no_improvement=50,
        n_init="auto",
        random_state=args.seed,
    )

    prefix_tokens = None
    steps_per_epoch = (len(files) + args.batch_size - 1) // args.batch_size
    for epoch in range(args.epochs):
        batches = range(0, len(files), args.batch_size)
        for start in tqdm(batches, total=steps_per_epoch, desc=f"DINOv3 + KMeans {epoch + 1}/{args.epochs}"):
            pixels = load_pixels(files[start : start + args.batch_size], processor, args.image_size).to(device)
            features, prefix_tokens = patch_features(model, pixels, grid_size)
            if features.shape[0] < args.clusters and not hasattr(kmeans, "cluster_centers_"):
                raise ValueError("The first feature batch must contain at least --clusters patch tokens")
            kmeans.partial_fit(features)

    centroids = torch.from_numpy(kmeans.cluster_centers_).float()
    prototypes = F.normalize(centroids, dim=-1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "prototypes": prototypes,
            "normal_centroids": centroids,
            "normal_centroids_normalized": prototypes,
            "encoder": args.model,
            "image_size": args.image_size,
            "patch_size": 16,
            "grid_size": grid_size,
            "prefix_tokens": prefix_tokens,
            "normal_image_count": len(files),
            "epochs": args.epochs,
            "clusters": args.clusters,
            "seed": args.seed,
        },
        args.output,
    )
    print(f"Saved {tuple(prototypes.shape)} prototype bank to {args.output}")


if __name__ == "__main__":
    main()
