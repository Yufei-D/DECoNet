# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import math
import os
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors

from .metrics import bbox_iou
from .tal import bbox2dist


class DFLoss(nn.Module):
    """Criterion class for computing DFL losses during training."""

    def __init__(self, reg_max=16) -> None:
        """Initialize the DFL module."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        """
        Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max=16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class _CosineThresholdScheduler:
    """τ(t) = τ_end + (τ_start - τ_end) × (1 + cos(π·t)) / 2"""

    def __init__(self, tau_start: float, tau_end: float, burn_in: int, decay_steps: int):
        self.tau_start, self.tau_end = tau_start, tau_end
        self.burn_in, self.decay_steps = burn_in, decay_steps

    def __call__(self, step: int) -> float:
        if step < self.burn_in:
            return self.tau_start
        if step >= self.decay_steps:
            return self.tau_end
        t = (step - self.burn_in) / (self.decay_steps - self.burn_in)
        return self.tau_end + (self.tau_start - self.tau_end) * (1 + math.cos(math.pi * t)) / 2

    def __repr__(self):
        return f"Cosine({self.tau_start:.2f}→{self.tau_end:.2f}, burn={self.burn_in}, end={self.decay_steps})"


class DECoNetDetectionLoss:
    """
    YOLOv8/v12 detection loss with optional C1+C2+C3 (teacher-guided).

    Pipeline in __call__ (order matters):
      Teacher forward → GT prep → C2 pseudo merge → assigner → C2 pseudo down-weight
      → C3 soft labels → C1 negative ignore → cls/box loss.

    Requires trainer to set ``model.teacher_model`` (EMA copy); see default.yaml ``use_teacher_model``.
    """

    def __init__(self, model, tal_topk=10):  # model must be de-paralleled
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)
        self.model = model  # EMA teacher: getattr(model, "teacher_model")，由 trainer 挂载

        # ----- C1+C2+C3（顺序见 __call__）：C2 伪标签 → assigner → C2 降权 → C3 软标签 → C1 ignore -----
        # 全局参数（基于 100ep 实验数据：step<5000 精度<70%，Teacher 不可靠）
        # Paper defaults can be overridden with DECONET_* environment variables.
        def _env(key, default):
            v = os.environ.get(f"DECONET_{key}")
            return type(default)(v) if v is not None else default

        _total_steps = _env("TOTAL_STEPS", 26800)  # paper run: 100 epochs x 268 batches
        _burn_in     = _env("BURN_IN", 5000)
        _decay_end   = int(_total_steps * 0.7)  # ~ep70

        # ===== C1: Negative Suppression Config =====
        self.c1_enabled = True
        self.c1_burn_in = _burn_in
        self.c1_tau_m = _env("C1_TAU_M", 0.30)
        self.c1_iou_thresh = _env("C1_IOU_THRESH", 0.10)
        self.c1_step_count = 0
        self.c1_tau_scheduler = _CosineThresholdScheduler(
            tau_start=_env("C1_TAU_START", 0.45),
            tau_end=_env("C1_TAU_END", 0.30),
            burn_in=_burn_in, decay_steps=_decay_end)
        self.c1_m_heatmap_dir = os.environ.get("DECONET_M_HEATMAP_DIR", "")
        self.c1_m_cache = {}

        # ===== C2: Pseudo Label Recovery Config =====
        self.c2_enabled = True
        self.c2_burn_in = _burn_in
        self.c2_high_scheduler = _CosineThresholdScheduler(
            tau_start=_env("C2_HIGH_START", 0.50),
            tau_end=_env("C2_HIGH_END", 0.40),
            burn_in=_burn_in, decay_steps=_decay_end)
        self.c2_low_scheduler = _CosineThresholdScheduler(
            tau_start=_env("C2_LOW_START", 0.35),
            tau_end=_env("C2_LOW_END", 0.25),
            burn_in=_burn_in, decay_steps=_decay_end)
        self.c2_m_thresh = _env("C2_M_THRESH", 0.25)
        self.c2_iou_thresh = _env("C2_IOU_THRESH", 0.20)
        self.c2_max_pseudo_per_img = _env("C2_MAX_PSEUDO", 20)
        self.c2_weight_high = _env("C2_WEIGHT_HIGH", 0.75)
        self.c2_weight_low  = _env("C2_WEIGHT_LOW", 0.50)
        self.c2_step_count = 0
        self.current_pseudo_labels = None

        # ===== C3: Soft Label Config =====
        self.c3_enabled = True
        self.c3_burn_in = _burn_in
        self.c3_alpha = _env("C3_ALPHA", 0.10)
        self.c3_step_count = 0
        self.teacher_preds = None  # set each step in __call__

    def _get_im_files(self, batch):
        if batch is None:
            return None
        im_raw = batch.get("im_file")
        if im_raw is None:
            return None
        return list(im_raw) if isinstance(im_raw, (list, tuple)) else [im_raw]

    @torch.no_grad()
    def _build_gt_masks(self, batch, B, fpn_shapes, device):
        """Rasterize current, augmentation-aligned annotations into FPN masks."""
        empty = [torch.zeros(B, 1, H, W, device=device) for H, W in fpn_shapes]
        if batch is None:
            return empty
        batch_idx = batch.get("batch_idx")
        bboxes_xywh = batch.get("bboxes")
        if batch_idx is None or bboxes_xywh is None or bboxes_xywh.numel() == 0:
            return empty
        batch_idx = batch_idx.to(device).long().view(-1)
        b_xywh = bboxes_xywh.to(device)
        cx, cy, bw, bh = b_xywh.unbind(dim=-1)
        x1 = (cx - bw / 2).clamp(0.0, 1.0)
        y1 = (cy - bh / 2).clamp(0.0, 1.0)
        x2 = (cx + bw / 2).clamp(0.0, 1.0)
        y2 = (cy + bh / 2).clamp(0.0, 1.0)
        masks = []
        for H, W in fpn_shapes:
            m = torch.zeros(B, 1, H, W, device=device)
            gx1 = (x1 * W).long().clamp(0, W - 1)
            gy1 = (y1 * H).long().clamp(0, H - 1)
            gx2 = (x2 * W).long().clamp(1, W)
            gy2 = (y2 * H).long().clamp(1, H)
            for k in range(batch_idx.shape[0]):
                b = int(batch_idx[k].item())
                if 0 <= b < B and gy2[k] > gy1[k] and gx2[k] > gx1[k]:
                    m[b, 0, gy1[k]:gy2[k], gx1[k]:gx2[k]] = 1.0
            masks.append(m)
        return masks

    def _build_pseudo_masks(self, pseudo_bboxes, pseudo_mask, image_shape, fpn_shapes):
        """Rasterize current-batch PLR boxes into binary masks for TGFA."""
        B = pseudo_bboxes.shape[0]
        device = pseudo_bboxes.device
        image_h, image_w = image_shape
        masks = []
        for H, W in fpn_shapes:
            m = torch.zeros(B, 1, H, W, device=device)
            for b in range(B):
                valid = pseudo_mask[b].squeeze(-1)
                if not valid.any():
                    continue
                boxes = pseudo_bboxes[b, valid]
                x1 = (boxes[:, 0] / image_w * W).long().clamp(0, W - 1)
                y1 = (boxes[:, 1] / image_h * H).long().clamp(0, H - 1)
                x2 = (boxes[:, 2] / image_w * W).long().clamp(1, W)
                y2 = (boxes[:, 3] / image_h * H).long().clamp(1, H)
                for i in range(boxes.shape[0]):
                    if y2[i] > y1[i] and x2[i] > x1[i]:
                        m[b, 0, y1[i]:y2[i], x1[i]:x2[i]] = 1.0
            masks.append(m)
        return masks

    def load_m_heatmap(self, img_path, device):
        if not self.c1_m_heatmap_dir:
            return None
        if img_path in self.c1_m_cache:
            return self.c1_m_cache[img_path]
        stem = Path(img_path).stem
        m_path = Path(self.c1_m_heatmap_dir) / f"{stem}.npy"
        if m_path.exists():
            m = torch.from_numpy(np.load(str(m_path))).float().to(device)
            self.c1_m_cache[img_path] = m
            return m
        return None

    def get_teacher_preds(self, img):
        teacher = getattr(self.model, "teacher_model", None)
        if teacher is None:
            return None
        teacher.eval()
        with torch.no_grad():
            return teacher(img)

    def pre_forward_teacher_inject(self, img, batch=None):
        """Run the EMA teacher and inject the paper's target foreground masks into TGFA."""
        self.teacher_preds = self.get_teacher_preds(img)
        self.current_pseudo_labels = None
        if self.teacher_preds is None or batch is None:
            return

        batch_size, _, image_h, image_w = img.shape
        image_size = torch.tensor([image_h, image_w], device=self.device, dtype=img.dtype)
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=image_size[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        self.current_pseudo_labels = self.generate_pseudo_labels(batch, gt_bboxes, gt_labels, mask_gt)

        from ultralytics.nn.modules.tgfa import TGFABlock

        blocks = [module for module in self.model.modules() if isinstance(module, TGFABlock)]
        if not blocks:
            return
        fpn_shapes = [(image_h // int(stride), image_w // int(stride)) for stride in self.stride]
        fused = self._build_gt_masks(batch, batch_size, fpn_shapes, img.device)
        pseudo_masks = self._build_pseudo_masks(
            self.current_pseudo_labels[0], self.current_pseudo_labels[2], (image_h, image_w), fpn_shapes
        )
        fused = [torch.maximum(gt_mask, pseudo_mask) for gt_mask, pseudo_mask in zip(fused, pseudo_masks)]
        if len(blocks) != len(fused):
            raise RuntimeError(f"Expected {len(fused)} TGFA blocks, found {len(blocks)}")
        for block, mask in zip(blocks, fused):
            block.set_external_mask(mask)

    def compute_c1_ignore_mask(self, batch, fg_mask, anchor_points, stride_tensor, gt_bboxes, mask_gt):
        """C1: 负样本抑制。ignore = NOT_fg AND NOT_near_gt AND (Teacher_high **OR** M_high)"""
        self.c1_step_count += 1
        B, N = fg_mask.shape
        device = fg_mask.device

        if not self.c1_enabled or self.c1_step_count < self.c1_burn_in or self.teacher_preds is None:
            return torch.zeros(B, N, dtype=torch.bool, device=device)

        tau_t = self.c1_tau_scheduler(self.c1_step_count)

        # ── Teacher 高置信度 ──
        teacher_decoded = self.teacher_preds[0]
        t_condition = teacher_decoded[:, 4:, :].max(dim=1).values > tau_t  # [B, N]

        # ── M 热图高响应（初始化为 False，OR 逻辑）──
        anchor_pixel = anchor_points * stride_tensor
        m_condition = torch.zeros(B, N, dtype=torch.bool, device=device)
        imgsz = batch["img"].shape[2]
        if "m_heatmap" in batch and batch["m_heatmap"] is not None:
            m_batch = batch["m_heatmap"].squeeze(1).to(device)
            Hm, Wm = m_batch.shape[1], m_batch.shape[2]
            ax = (anchor_pixel[:, 0] / imgsz * Wm).long().clamp(0, Wm - 1)
            ay = (anchor_pixel[:, 1] / imgsz * Hm).long().clamp(0, Hm - 1)
            m_condition = m_batch[:, ay, ax] > self.c1_tau_m
        elif "im_file" in batch:
            im_raw = batch["im_file"]
            im_files = list(im_raw) if isinstance(im_raw, (list, tuple)) else [im_raw]
            m_list = [self.load_m_heatmap(im_files[i], device) for i in range(min(B, len(im_files)))]
            valid_ms = [(i, m) for i, m in enumerate(m_list) if m is not None]
            if valid_ms:
                indices, maps = zip(*valid_ms)
                m_stack = torch.stack(maps)
                mh, mw = m_stack.shape[1], m_stack.shape[2]
                ax = (anchor_pixel[:, 0] / imgsz * mw).long().clamp(0, mw - 1)
                ay = (anchor_pixel[:, 1] / imgsz * mh).long().clamp(0, mh - 1)
                idx_b = torch.tensor(indices, dtype=torch.long, device=device)
                m_condition[idx_b] = m_stack[:, ay, ax] > self.c1_tau_m

        # ── 不与 GT 有高 IoU ──
        not_near_gt = torch.ones(B, N, dtype=torch.bool, device=device)
        if mask_gt.any():
            hs = stride_tensor.squeeze(-1) / 2
            ax_px, ay_px = anchor_pixel[:, 0], anchor_pixel[:, 1]
            ab = torch.stack([ax_px - hs, ay_px - hs, ax_px + hs, ay_px + hs], dim=-1)
            a = ab[None, :, None, :]
            g = gt_bboxes[:, None, :, :]
            inter = (torch.min(a[..., 2], g[..., 2]) - torch.max(a[..., 0], g[..., 0])).clamp_(0)
            inter *= (torch.min(a[..., 3], g[..., 3]) - torch.max(a[..., 1], g[..., 1])).clamp_(0)
            area_a = (ab[:, 2] - ab[:, 0]) * (ab[:, 3] - ab[:, 1])
            area_g = (gt_bboxes[..., 2] - gt_bboxes[..., 0]) * (gt_bboxes[..., 3] - gt_bboxes[..., 1])
            denom = area_a[None, :, None] + area_g[:, None, :] - inter + 1e-7
            inter /= denom
            inter.masked_fill_(~mask_gt.bool().squeeze(-1)[:, None, :], 0)
            not_near_gt = inter.max(dim=-1).values < self.c1_iou_thresh

        # ── OR 组合：Teacher 高置信 OR M 热图高响应 ──
        signal = t_condition | m_condition
        ignore_mask = (~fg_mask) & not_near_gt & signal

        if self.c1_step_count % 500 == 0:
            n_ign = ignore_mask.sum().item()
            n_t = (t_condition & ~m_condition & ~fg_mask & not_near_gt).sum().item()
            n_m = (m_condition & ~t_condition & ~fg_mask & not_near_gt).sum().item()
            n_b = (t_condition & m_condition & ~fg_mask & not_near_gt).sum().item()
            print(f"[C1] step={self.c1_step_count}, τ_t={tau_t:.3f}, "
                  f"ignore={n_ign} (T={n_t}, M={n_m}, both={n_b})")

        return ignore_mask

    @staticmethod
    def _xywh_to_xyxy(bboxes):
        xy, wh = bboxes[..., :2], bboxes[..., 2:4]
        return torch.cat([xy - wh / 2, xy + wh / 2], dim=-1)

    def generate_pseudo_labels(self, batch, gt_bboxes, gt_labels, mask_gt):
        """C2: 分层伪标签恢复。高置信直接采纳，中置信需 M 热图验证。"""
        self.c2_step_count += 1
        B = gt_bboxes.shape[0]
        device = gt_bboxes.device
        mp = self.c2_max_pseudo_per_img

        pseudo_bboxes  = torch.zeros(B, mp, 4, device=device)
        pseudo_labels  = torch.zeros(B, mp, 1, device=device)
        pseudo_mask    = torch.zeros(B, mp, 1, dtype=torch.bool, device=device)
        pseudo_weights = torch.zeros(B, mp, 1, device=device)

        if not self.c2_enabled or self.c2_step_count < self.c2_burn_in or self.teacher_preds is None:
            return pseudo_bboxes, pseudo_labels, pseudo_mask, pseudo_weights

        tau_high = self.c2_high_scheduler(self.c2_step_count)
        tau_low  = self.c2_low_scheduler(self.c2_step_count)

        teacher_decoded = self.teacher_preds[0]
        t_bboxes = self._xywh_to_xyxy(teacher_decoded[:, :4, :].permute(0, 2, 1))
        t_cls = teacher_decoded[:, 4:, :].permute(0, 2, 1)
        t_max_score, t_max_cls = t_cls.max(dim=-1)
        NA = t_max_score.shape[1]

        # ── 分层置信度 ──
        high_conf = t_max_score >= tau_high
        mid_conf  = (t_max_score >= tau_low) & (t_max_score < tau_high)

        # ── 不与 GT 重叠 ──
        no_overlap = torch.ones(B, NA, dtype=torch.bool, device=device)
        if mask_gt.any():
            tb = t_bboxes[:, :, None, :]
            gb = gt_bboxes[:, None, :, :]
            inter = (torch.min(tb[..., 2], gb[..., 2]) - torch.max(tb[..., 0], gb[..., 0])).clamp_(0)
            inter *= (torch.min(tb[..., 3], gb[..., 3]) - torch.max(tb[..., 1], gb[..., 1])).clamp_(0)
            area_t = (t_bboxes[..., 2] - t_bboxes[..., 0]) * (t_bboxes[..., 3] - t_bboxes[..., 1])
            area_g = (gt_bboxes[..., 2] - gt_bboxes[..., 0]) * (gt_bboxes[..., 3] - gt_bboxes[..., 1])
            denom = area_t[:, :, None] + area_g[:, None, :] - inter + 1e-7
            inter /= denom
            inter.masked_fill_(~mask_gt.bool().squeeze(-1)[:, None, :], 0)
            no_overlap = inter.max(dim=-1).values < self.c2_iou_thresh

        # ── M 热图验证（仅用于中置信度层）──
        # Medium-confidence boxes require M-map evidence; absent maps must not pass PLR.
        m_ok = torch.zeros(B, NA, dtype=torch.bool, device=device)
        imgsz = batch["img"].shape[2]
        if "m_heatmap" in batch and batch["m_heatmap"] is not None:
            m_batch = batch["m_heatmap"].squeeze(1).to(device)
            Hm, Wm = m_batch.shape[1], m_batch.shape[2]
            cx = ((t_bboxes[..., 0] + t_bboxes[..., 2]) / 2 / imgsz * Wm).long().clamp(0, Wm - 1)
            cy = ((t_bboxes[..., 1] + t_bboxes[..., 3]) / 2 / imgsz * Hm).long().clamp(0, Hm - 1)
            flat_idx = (cy * Wm + cx).clamp(0, Hm * Wm - 1)
            m_scores = torch.gather(m_batch.view(B, -1), 1, flat_idx)  # [B, NA]
            m_ok = m_scores > self.c2_m_thresh
        elif "im_file" in batch:
            im_raw = batch["im_file"]
            im_files = list(im_raw) if isinstance(im_raw, (list, tuple)) else [im_raw]
            m_list = [self.load_m_heatmap(im_files[i], device) for i in range(min(B, len(im_files)))]
            valid_ms = [(i, m) for i, m in enumerate(m_list) if m is not None]
            if valid_ms:
                indices, maps = zip(*valid_ms)
                m_stack = torch.stack(maps)
                mh, mw = m_stack.shape[1], m_stack.shape[2]
                idx_b = torch.tensor(indices, dtype=torch.long, device=device)
                t_sel = t_bboxes[idx_b]
                cx = ((t_sel[..., 0] + t_sel[..., 2]) / 2 / imgsz * mw).long().clamp(0, mw - 1)
                cy = ((t_sel[..., 1] + t_sel[..., 3]) / 2 / imgsz * mh).long().clamp(0, mh - 1)
                m_scores = torch.gather(m_stack.view(len(indices), -1), 1, cy * mw + cx)
                m_ok[idx_b] = m_scores > self.c2_m_thresh

        # ── 组合条件 ──
        high_valid = high_conf & no_overlap                # 高置信：不需要 M 验证
        mid_valid  = mid_conf  & no_overlap & m_ok         # 中置信：需要 M 验证
        combined   = high_valid | mid_valid

        # ── 分层权重 ──
        weights = torch.zeros_like(t_max_score)
        weights[high_valid]                = self.c2_weight_high
        weights[mid_valid & ~high_valid]   = self.c2_weight_low
        weights = weights * t_max_score    # 乘以置信度

        # ── Top-K 选择 ──
        masked_scores = t_max_score.clone()
        masked_scores[~combined] = -1.0
        topk_scores, topk_idx = masked_scores.topk(min(mp, NA), dim=-1)
        valid = topk_scores > 0

        if valid.any():
            idx4 = topk_idx.unsqueeze(-1).expand(-1, -1, 4)
            pseudo_bboxes = torch.gather(t_bboxes, 1, idx4)
            pseudo_bboxes[~valid.unsqueeze(-1).expand_as(pseudo_bboxes)] = 0.0
            pseudo_labels = torch.gather(t_max_cls, 1, topk_idx).float().unsqueeze(-1)
            pseudo_labels[~valid.unsqueeze(-1)] = 0.0
            pseudo_mask = valid.unsqueeze(-1)
            pseudo_weights = torch.gather(weights, 1, topk_idx).unsqueeze(-1)
            pseudo_weights[~valid.unsqueeze(-1)] = 0.0

        if self.c2_step_count % 200 == 0 and self.c2_step_count >= self.c2_burn_in:
            total = pseudo_mask.sum().item()
            n_h = high_valid.sum().item()
            n_m = (mid_valid & ~high_valid).sum().item()
            if total > 0:
                avg_w = pseudo_weights[pseudo_mask.expand_as(pseudo_weights)].mean().item()
                print(f"[C2] step={self.c2_step_count}, τ_h={tau_high:.3f}, τ_l={tau_low:.3f}, "
                      f"pseudo={total} (high={n_h}, mid={n_m}), avg_w={avg_w:.3f}")
            else:
                print(f"[C2] step={self.c2_step_count}, τ_h={tau_high:.3f}, τ_l={tau_low:.3f}, pseudo=0")

        return pseudo_bboxes, pseudo_labels, pseudo_mask, pseudo_weights


    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)


    def __call__(self, preds, batch):
        if self.teacher_preds is None:
            self.pre_forward_teacher_inject(batch["img"], batch)

        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds

        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        # GT 预处理
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # ===== C2: Pseudo Label Recovery =====
        if self.current_pseudo_labels is None:
            pseudo_data = self.generate_pseudo_labels(batch, gt_bboxes, gt_labels, mask_gt)
        else:
            pseudo_data = self.current_pseudo_labels
        pseudo_bboxes, pseudo_labels, pseudo_mask, pseudo_weights = pseudo_data
        n_real = gt_bboxes.shape[1]
        is_pseudo = None
        if pseudo_mask.any():
            gt_bboxes = torch.cat([gt_bboxes, pseudo_bboxes], dim=1)
            gt_labels = torch.cat([gt_labels, pseudo_labels], dim=1)
            mask_gt = torch.cat([mask_gt, pseudo_mask], dim=1)
            n_pseudo = pseudo_bboxes.shape[1]
            is_pseudo = torch.zeros(batch_size, n_real + n_pseudo, dtype=torch.bool, device=self.device)
            is_pseudo[:, n_real:] = pseudo_mask.squeeze(-1)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # ===== C2: 伪标签 anchor 降权 (向量化) =====
        if is_pseudo is not None and is_pseudo.any():
            gt_idx = target_gt_idx.long()
            gt_idx_clamped = gt_idx.clamp(0, is_pseudo.shape[1] - 1)
            anchor_matched_pseudo = torch.gather(is_pseudo, 1, gt_idx_clamped) & fg_mask
            if anchor_matched_pseudo.any():
                pseudo_gt_local = (gt_idx_clamped - n_real).clamp(0, self.c2_max_pseudo_per_img - 1)
                weight_per_anchor = torch.gather(pseudo_weights.squeeze(-1), 1, pseudo_gt_local)
                scale = torch.ones_like(target_scores[:, :, 0])
                scale[anchor_matched_pseudo] = weight_per_anchor[anchor_matched_pseudo].to(scale.dtype)
                target_scores = target_scores * scale.unsqueeze(-1)
            target_scores_sum = max(target_scores.sum(), 1)

        # ===== C3: Soft Label Mixing =====
        self.c3_step_count += 1
        if (self.c3_enabled
                and self.c3_step_count >= self.c3_burn_in
                and self.teacher_preds is not None
                and fg_mask.any()):
            teacher_cls = self.teacher_preds[0][:, 4:, :].permute(0, 2, 1)
            alpha = self.c3_alpha
            target_scores[fg_mask] = (
                (1 - alpha) * target_scores[fg_mask]
                + alpha * teacher_cls[fg_mask].to(target_scores.dtype)
            )
            target_scores_sum = max(target_scores.sum(), 1)

        # ===== C1: Negative Suppression Mask =====
        ignore_mask = self.compute_c1_ignore_mask(
            batch, fg_mask, anchor_points, stride_tensor, gt_bboxes, mask_gt
        )

        # Cls loss (带 C1 ignore)
        cls_loss_per_anchor = self.bce(pred_scores, target_scores.to(dtype))
        if ignore_mask.any():
            cls_loss_per_anchor[ignore_mask] = 0.0
        loss[1] = cls_loss_per_anchor.sum() / target_scores_sum

        # Bbox loss (不变)
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask
            )
        # 加权
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl

        return loss.sum() * batch_size, loss.detach()  # 标准返回格式!


class DetectAuxTGFALoss(DECoNetDetectionLoss):
    """DetectAuxTGFA 头的 loss: 主路径 + TGFA-增强辅助路径。

    主路径: 跟 v8DetectionLoss 完全一致 (含 C1+C2+C3 监督管线)。
    辅助路径: 在 TGFA 增强后的特征上, 用同一套 C1+C2+C3 监督。
    Total loss = L_main + λ_aux · L_aux,  λ_aux 默认 0.25 (来自 YOLOv7 lead/aux)。

    DECONET_AUX_WEIGHT can override the paper default for ablations.

    推理时 head 直接走标准 Detect 路径, 此 loss 类不被调用。
    """

    def __init__(self, model):
        super().__init__(model)
        self.aux_weight = float(os.environ.get("DECONET_AUX_WEIGHT", 0.25))

    def __call__(self, preds, batch):
        # 解包 head 输出
        if isinstance(preds, dict) and "main" in preds:
            main_feats = preds["main"]
            aux_feats = preds.get("aux")
        elif isinstance(preds, tuple) and isinstance(preds[1], dict) and "main" in preds[1]:
            main_feats = preds[1]["main"]
            aux_feats = preds[1].get("aux")
        else:
            # 推理 mode 或非 dict 输出, 走标准 v8 loss
            return super().__call__(preds, batch)

        # 主 loss: 走标准 v8DetectionLoss (含 C1+C2+C3)
        main_loss, main_items = super().__call__(main_feats, batch)

        # 辅助 loss: 同样调用 super, 但在 TGFA 增强特征上;
        # 保护 c1/c2/c3 步进计数器, 避免辅助路径推进步数让 cosine schedule 走快一步
        if aux_feats is not None and self.aux_weight > 0:
            saved = (
                getattr(self, "c1_step_count", 0),
                getattr(self, "c2_step_count", 0),
                getattr(self, "c3_step_count", 0),
            )
            try:
                # The auxiliary head uses the same schedule position as the main head.
                self.c1_step_count = max(0, saved[0] - 1)
                self.c3_step_count = max(0, saved[2] - 1)
                aux_loss, aux_items = super().__call__(aux_feats, batch)
            finally:
                if hasattr(self, "c1_step_count"):
                    self.c1_step_count = saved[0]
                if hasattr(self, "c2_step_count"):
                    self.c2_step_count = saved[1]
                if hasattr(self, "c3_step_count"):
                    self.c3_step_count = saved[2]
            return main_loss + self.aux_weight * aux_loss, main_items + self.aux_weight * aux_items

        return main_loss, main_items
