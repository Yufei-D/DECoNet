"""
TGFA: Teacher-Guided Feature Augmentation
==========================================

A training-only, target-guided wavelet feature augmentation module for DECoNet.

    y = x + sigma * m * ReLU(psi(x))

Guarantees:
  G1. y >= x pointwise (no suppression path).
  G2. If m_ij = 0 then y_ij = x_ij EXACTLY.
  G3. epsilon-identity at init (sigma ~ 0.018).
  G4. Identity (y=x) reachable via sigma -> 0.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, DWConv


# ===========================================================================
# Haar Wavelet
# ===========================================================================
class HaarWavelet(nn.Module):
    def __init__(self):
        super().__init__()
        ll = torch.tensor([[1., 1.], [1., 1.]])
        lh = torch.tensor([[-1., -1.], [1., 1.]])
        hl = torch.tensor([[-1., 1.], [-1., 1.]])
        hh = torch.tensor([[1., -1.], [-1., 1.]])
        filt = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1) / 2.0
        self.register_buffer("filt", filt, persistent=False)

    def _w(self, c, device, dtype):
        return self.filt.to(device=device, dtype=dtype).repeat(c, 1, 1, 1)

    def dwt(self, x):
        b, c, h, w = x.shape
        if h % 2 or w % 2:
            raise ValueError(f"DWT requires even H,W; got ({h},{w})")
        return F.conv2d(x, self._w(c, x.device, x.dtype), stride=2, groups=c)

    def idwt(self, y):
        b, c4, _, _ = y.shape
        if c4 % 4:
            raise ValueError(f"IDWT expects 4C channels; got {c4}")
        c = c4 // 4
        return F.conv_transpose2d(y, self._w(c, y.device, y.dtype), stride=2, groups=c)


# ===========================================================================
# Utilities
# ===========================================================================
def _check_feature(x, name="x"):
    if not torch.is_tensor(x):
        raise TypeError(f"{name} must be a Tensor, got {type(x)}")
    if x.dim() != 4:
        raise ValueError(f"{name} must be 4-D [B,C,H,W], got {tuple(x.shape)}")


# ===========================================================================
# Backward-compat shim: 旧版 TGFABlock 包含 _SelfMaskHead，pickle 在 ckpt 里。
# 已移除该类，但保留一个 dummy stub 让旧 ckpt 能 unpickle（其权重会被
# load_state_dict 的 strict=False 忽略）。
# ===========================================================================
class _SelfMaskHead(nn.Module):
    """Deprecated. Kept only so old checkpoints can unpickle."""
    def __init__(self, c=None, hidden=None):
        super().__init__()

    def forward(self, x):
        # 不再使用；如被错误调用则返回全 1（与新 _get_mask 行为一致）
        return torch.ones(x.shape[0], 1, *x.shape[-2:], device=x.device, dtype=x.dtype)


# ===========================================================================
# Main block
# ===========================================================================
class TGFABlock(nn.Module):
    """Provably non-degrading neck module.

    y = x + sigma * m * ReLU(psi(x))

    Args:
        c1, c2: input/output channels (must equal).
        stride: FPN stride (8/16/32), for API symmetry.
        reduce_ratio: channel bottleneck inside psi.
        sigma_init_s: initial s value (sigma = sigmoid(s)).
        use_teacher_mask: whether to use the externally supplied target mask in training.
        detach_external_mask: whether to detach that mask from the computation graph.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        stride: int = 8,
        reduce_ratio: int = 4,
        sigma_init_s: float = -4.0,
        use_teacher_mask: bool = True,
        detach_external_mask: bool = True,
    ):
        super().__init__()
        if c1 != c2:
            raise ValueError(f"TGFABlock requires c1 == c2, got {c1}, {c2}")
        if c1 < 8:
            raise ValueError(f"c1 must be >= 8, got {c1}")

        self.c1 = int(c1)
        self.c2 = int(c2)
        self.stride = int(stride)
        self.reduce_ratio = int(reduce_ratio)
        self.use_teacher_mask = bool(use_teacher_mask)
        self.detach_external_mask = bool(detach_external_mask)

        cm = max(8, c1 // reduce_ratio)
        self.cm = cm

        self.wavelet = HaarWavelet()

        self.hf_reduce = Conv(3 * c1, 3 * cm, k=1, s=1, g=3, act=True)
        self.hf_dw = DWConv(3 * cm, 3 * cm, k=3, s=1, act=True)
        self.hf_expand = nn.Conv2d(
            3 * cm, 3 * c1, kernel_size=1, stride=1, padding=0, groups=3, bias=True
        )
        with torch.no_grad():
            self.hf_expand.weight.mul_(0.1)
            self.hf_expand.bias.zero_()

        self.s = nn.Parameter(torch.tensor(float(sigma_init_s)))

        self._ext_mask: Optional[torch.Tensor] = None

    def set_external_mask(self, mask: Optional[torch.Tensor]) -> None:
        if mask is None:
            self._ext_mask = None
            return
        if mask.dim() != 4 or mask.shape[1] != 1:
            raise ValueError(f"mask must be [B,1,H,W], got {tuple(mask.shape)}")
        self._ext_mask = mask.detach() if self.detach_external_mask else mask

    def clear_external_mask(self) -> None:
        self._ext_mask = None

    def _get_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Return the spatial mask for TGFA enhancement.

        The external mask is built from annotated and PLR-recovered target boxes. The
        complete TGFA branch is skipped by DetectAuxTGFA during inference.
        """
        # Training with a target mask: enhance only target regions.
        if self.training and self.use_teacher_mask and self._ext_mask is not None:
            m = self._ext_mask
            B = x.shape[0]
            # Fall back to an all-one mask if the batch sizes do not match.
            if m.shape[0] == B:
                if m.shape[-2:] != x.shape[-2:]:
                    m = F.interpolate(m, size=x.shape[-2:], mode="bilinear", align_corners=False)
                if m.dtype != x.dtype:
                    m = m.to(dtype=x.dtype)
                if m.device != x.device:
                    m = m.to(device=x.device)
                return m.clamp(0.0, 1.0)
        # Training before a target mask is available uses a neutral all-one mask.
        B, _, H, W = x.shape
        return torch.ones(B, 1, H, W, device=x.device, dtype=x.dtype)

    @staticmethod
    def _pad_even(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        H, W = x.shape[-2:]
        ph, pw = H % 2, W % 2
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        return x, (ph, pw)

    def _compute_psi(self, x: torch.Tensor) -> torch.Tensor:
        xp, (ph, pw) = self._pad_even(x)
        B, C = xp.shape[0], xp.shape[1]
        H2, W2 = xp.shape[-2] // 2, xp.shape[-1] // 2

        coeffs = self.wavelet.dwt(xp).view(B, C, 4, H2, W2)
        hf = coeffs[:, :, 1:].permute(0, 2, 1, 3, 4).contiguous().view(B, 3 * C, H2, W2)

        hf_e = self.hf_expand(self.hf_dw(self.hf_reduce(hf)))

        hf_out = hf_e.view(B, 3, C, H2, W2).permute(0, 2, 1, 3, 4).contiguous()
        ll_zero = torch.zeros(
            (B, C, 1, H2, W2), dtype=x.dtype, device=x.device
        )
        combined = torch.cat([ll_zero, hf_out], dim=2).view(B, 4 * C, H2, W2)
        psi = self.wavelet.idwt(combined)
        if ph or pw:
            psi = psi[..., :x.shape[-2], :x.shape[-1]]
        return psi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """y = x + σ · m · ReLU(ψ(x))

        The containing head calls TGFA only while training.
        """
        _check_feature(x, "x")
        if x.shape[1] != self.c1:
            raise ValueError(
                f"TGFABlock built for c1={self.c1}, got C={x.shape[1]}"
            )

        m = self._get_mask(x)
        psi = self._compute_psi(x)
        aug = F.relu(psi)
        sigma = torch.sigmoid(self.s)

        return x + sigma * m * aug

    def extra_repr(self) -> str:
        return (
            f"c={self.c1}, stride={self.stride}, cm={self.cm}, "
            f"sigma={torch.sigmoid(self.s).item():.4f}, "
            f"teacher_mask={self.use_teacher_mask}"
        )
