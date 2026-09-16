"""WSCDownDual: dual-branch wavelet downsampling for YOLO backbone."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, DWConv


class HaarDownsampling(nn.Module):
    """
    Fixed Haar wavelet downsampling.
    Input:  [B, C, H, W]
    Output: [B, 4C, H/2, W/2]
    """

    def __init__(self):
        super().__init__()
        ll = torch.tensor([[1., 1.], [1., 1.]])
        lh = torch.tensor([[-1., -1.], [1., 1.]])
        hl = torch.tensor([[-1., 1.], [-1., 1.]])
        hh = torch.tensor([[1., -1.], [-1., 1.]])

        filt = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1) / 2.0  # [4, 1, 2, 2]
        self.register_buffer("filt", filt)

    def forward(self, x):
        b, c, h, w = x.shape
        weight = self.filt.repeat(c, 1, 1, 1)  # [4C, 1, 2, 2]
        return F.conv2d(x, weight, stride=2, padding=0, groups=c)


class BlurPool(nn.Module):
    """Lightweight anti-aliased downsampling with [1,2,1] binomial kernel."""

    def __init__(self, channels, stride=2):
        super().__init__()
        filt = torch.tensor([1., 2., 1.])
        filt = filt[:, None] * filt[None, :]
        filt = filt / filt.sum()  # [3, 3]
        filt = filt[None, None, :, :].repeat(channels, 1, 1, 1)
        self.register_buffer("filt", filt)
        self.channels = channels
        self.stride = stride

    def forward(self, x):
        return F.conv2d(x, self.filt, stride=self.stride, padding=1, groups=self.channels)


class WSCDownDual(nn.Module):
    """
    Dual-branch wavelet downsampling for YOLO backbone.

    Main branch:  Haar -> 1x1 -> DW 3x3 -> 1x1
    Aux  branch:  BlurPool/AvgPool -> 1x1
    Fusion:       main + alpha * aux

    Args:
        c1: input channels
        c2: output channels
        e:  hidden expansion ratio relative to c2 (default 0.5)
        pool: 'blur' or 'avg'
        act: activation flag
        shortcut_scale: if True, learn a scalar fusion weight for aux branch
    """

    def __init__(self, c1, c2, e=0.5, pool='blur', act=True, shortcut_scale=True):
        super().__init__()
        assert pool in {'blur', 'avg'}

        cm = max(8, int(c2 * e))

        # main branch
        self.haar = HaarDownsampling()
        self.main_pw1 = Conv(4 * c1, cm, k=1, s=1, act=act)
        self.main_dw = DWConv(cm, cm, k=3, s=1, act=act)
        self.main_pw2 = Conv(cm, c2, k=1, s=1, act=act)

        # aux branch
        if pool == 'blur':
            self.pool = BlurPool(c1, stride=2)
        else:
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.aux_pw = Conv(c1, c2, k=1, s=1, act=False)

        # learnable fusion weight
        self.alpha = nn.Parameter(torch.tensor(1.0)) if shortcut_scale else None

    def forward(self, x):
        y_main = self.main_pw2(self.main_dw(self.main_pw1(self.haar(x))))
        y_aux = self.aux_pw(self.pool(x))
        if self.alpha is not None:
            return y_main + self.alpha * y_aux
        return y_main + y_aux
