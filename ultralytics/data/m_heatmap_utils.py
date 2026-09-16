# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Utilities for defectness M heatmaps aligned with YOLO image augmentations."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_m_npy_to_hw(m_path: Path) -> np.ndarray | None:
    """Load a single-channel M heatmap from .npy; return HxW float32 or None."""
    if not m_path.exists():
        return None
    m = np.load(str(m_path)).astype(np.float32)
    m = np.squeeze(m)
    if m.ndim != 2:
        raise ValueError(f"M heatmap must be 2D after squeeze, got shape {m.shape} from {m_path}")
    return m


def resize_m_to_shape(m: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Resize M to (out_h, out_w). Prefer INTER_AREA when shrinking, else LINEAR."""
    if m.shape[0] == out_h and m.shape[1] == out_w:
        return m.astype(np.float32, copy=False)
    interp = cv2.INTER_AREA if (out_h < m.shape[0] or out_w < m.shape[1]) else cv2.INTER_LINEAR
    return cv2.resize(m, (out_w, out_h), interpolation=interp).astype(np.float32)


def load_m_heatmap_for_image(im_file: str, img_hw: tuple[int, int], heatmap_dir: Path) -> np.ndarray:
    """
    Load M for one image path and resize to match current image (H, W) after load_image.

    Args:
        im_file: Absolute path to the RGB/BGR image file.
        img_hw: (H, W) of label['img'] at load time.
        heatmap_dir: Directory containing {stem}.npy heatmaps.

    Returns:
        HxW float32 array; zeros if file missing.
    """
    stem = Path(im_file).stem
    m_path = heatmap_dir / f"{stem}.npy"
    h, w = int(img_hw[0]), int(img_hw[1])
    m = load_m_npy_to_hw(m_path)
    if m is None:
        return np.zeros((h, w), dtype=np.float32)
    return resize_m_to_shape(m, h, w)


def ensure_m2d(m: np.ndarray) -> np.ndarray:
    """Return HxW float32."""
    x = np.asarray(m, dtype=np.float32)
    x = np.squeeze(x)
    if x.ndim != 2:
        raise ValueError(f"m_heatmap must be 2D, got {x.shape}")
    return x
