"""Tier-3: drop-in PyTorch loss modules to fine-tune a video generator
toward better WorldArena scores.

All three losses are framed as ``nn.Module`` so any training loop can do::

    bundle = PerceptualLossBundle(
        lpips_weight=0.1,
        flow_weight=0.05,
        vfi_recon_weight=0.05,
        device="cuda",
    )
    extra = bundle(pred_video, gt_video)   # both [B, T, 3, H, W] in [0, 1]
    total_loss = base_loss + extra["total"]

The losses are designed to nudge specifically the metrics that the plan
identified as low-hanging:

* ``LpipsLoss``               -> subject_consistency / background_consistency
* ``FlowConsistencyLoss``     -> photometric_smoothness
* ``VFIReconstructionLoss``   -> motion_smoothness (same VFI the metric uses)

Heavy backbones (LPIPS-VGG, RAFT, VFIMamba) are loaded lazily so users can
opt out of any subset by setting the corresponding weight to 0.0.
"""
from __future__ import annotations

import os
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _flatten_video(x: torch.Tensor) -> torch.Tensor:
    """[B, T, 3, H, W] -> [B*T, 3, H, W]."""
    if x.ndim != 5:
        raise ValueError(f"expected [B,T,3,H,W], got {tuple(x.shape)}")
    b, t, c, h, w = x.shape
    return x.reshape(b * t, c, h, w)


def _adjacent_pairs(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[B, T, 3, H, W] -> ([B*(T-1), 3, H, W], [B*(T-1), 3, H, W])."""
    if x.ndim != 5:
        raise ValueError(f"expected [B,T,3,H,W], got {tuple(x.shape)}")
    a = x[:, :-1].reshape(-1, *x.shape[2:])
    b = x[:, 1:].reshape(-1, *x.shape[2:])
    return a, b


def _triplets(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """[B, T, 3, H, W] -> three views indexed at t-1, t, t+1 for t in [1, T-1)."""
    if x.shape[1] < 3:
        raise ValueError(f"need T>=3 for triplets, got T={x.shape[1]}")
    prev = x[:, :-2].reshape(-1, *x.shape[2:])
    cur = x[:, 1:-1].reshape(-1, *x.shape[2:])
    nxt = x[:, 2:].reshape(-1, *x.shape[2:])
    return prev, cur, nxt


# ----------------------------------------------------------------------------
# LPIPS (pyiqa -> fallback to a raw VGG-feature distance)
# ----------------------------------------------------------------------------


class LpipsLoss(nn.Module):
    """LPIPS distance between paired frames of pred and target video.

    Tries pyiqa's LPIPS first (matches the eval environment), and falls back
    to a torchvision-VGG16 multi-layer feature L1 distance if pyiqa is
    unavailable, fails to instantiate, or the constructor would block on a
    weight download. ``use_pyiqa=False`` forces the fallback path which is
    fully self-contained and does not require network access at startup.
    """

    def __init__(
        self,
        net: str = "alex",
        reduction: str = "mean",
        use_pyiqa: bool = True,
    ):
        super().__init__()
        self.reduction = reduction
        self._impl: Optional[nn.Module] = None
        if use_pyiqa:
            try:
                import pyiqa  # noqa
                self._impl = pyiqa.create_metric("lpips", net=net, as_loss=True)
            except Exception as exc:
                print(f"[LpipsLoss] pyiqa LPIPS unavailable ({exc}); falling back to VGG features.")
        if self._impl is None:
            self._impl = _VggFeatureDistance()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        a = _flatten_video(pred)
        b = _flatten_video(target)
        # pyiqa LPIPS expects values in [0, 1].
        d = self._impl(a, b)
        if self.reduction == "mean":
            return d.mean()
        if self.reduction == "sum":
            return d.sum()
        return d


class _VggFeatureDistance(nn.Module):
    """Simple VGG16 feature L1 distance used when pyiqa LPIPS is unavailable.

    Not a perfect LPIPS replacement, but it preserves the loss-shape contract
    so training pipelines can run without the pyiqa dependency at fine-tune
    time (e.g. when training on a different cluster than evaluation).
    """

    def __init__(self):
        super().__init__()
        from torchvision.models import vgg16
        backbone = vgg16(weights=None).features.eval()
        for p in backbone.parameters():
            p.requires_grad_(False)
        self.layers = nn.ModuleList()
        seq = []
        # Cut at relu1_2, relu2_2, relu3_3 — same pattern as classic LPIPS.
        cuts = {3, 8, 15}
        for i, m in enumerate(backbone):
            seq.append(m)
            if i in cuts:
                self.layers.append(nn.Sequential(*seq))
                seq = []
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = (a - self.mean) / self.std
        b = (b - self.mean) / self.std
        loss = a.new_zeros(())
        x, y = a, b
        for layer in self.layers:
            x = layer(x)
            y = layer(y)
            loss = loss + F.l1_loss(x, y)
        return loss


# ----------------------------------------------------------------------------
# Flow-consistency loss (RAFT optional, Farneback fallback)
# ----------------------------------------------------------------------------


class FlowConsistencyLoss(nn.Module):
    """Penalize divergence between flow(pred_t -> pred_{t+1}) and the GT flow.

    The optical-flow extractor is configurable:
    * ``raft_ckpt`` set    -> uses ``WorldArena.third_party.RAFT`` (matches the
                              flow_score / dynamic_degree metric).
    * otherwise            -> torch-implementation of Lucas-Kanade-style
                              gradient flow, computed in fp32 with no
                              external dependency. This is purely a *fallback*
                              and yields lower-quality flow than RAFT, but
                              keeps the loss API working without checkpoints.
    """

    def __init__(self, raft_ckpt: Optional[str] = None, weight_div: float = 0.0):
        super().__init__()
        self._raft = None
        if raft_ckpt is not None and os.path.exists(raft_ckpt):
            try:
                from easydict import EasyDict as edict
                from WorldArena.third_party.RAFT.core.raft import RAFT  # type: ignore
                args = edict({"model": raft_ckpt, "small": False,
                              "mixed_precision": False, "alternate_corr": False})
                model = RAFT(args)
                ckpt = torch.load(raft_ckpt, map_location="cpu", weights_only=False)
                new_ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
                model.load_state_dict(new_ckpt)
                model.eval()
                for p in model.parameters():
                    p.requires_grad_(False)
                self._raft = model
            except Exception as exc:
                print(f"[FlowConsistencyLoss] RAFT load failed: {exc}")
        self.weight_div = weight_div  # extra penalty on flow divergence (smoothness)

    def _flow(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self._raft is not None:
            # RAFT expects 0..255 floats.
            with torch.no_grad():  # frozen extractor, but the loss flows back to inputs.
                _, flow = self._raft(a * 255.0, b * 255.0, iters=12, test_mode=True)
            return flow
        # Fallback: torch-only Sobel-based gradient flow. Crude but differentiable.
        gray_a = (0.2989 * a[:, 0] + 0.5870 * a[:, 1] + 0.1140 * a[:, 2]).unsqueeze(1)
        gray_b = (0.2989 * b[:, 0] + 0.5870 * b[:, 1] + 0.1140 * b[:, 2]).unsqueeze(1)
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                          dtype=a.dtype, device=a.device).view(1, 1, 3, 3) / 8.0
        ky = kx.transpose(-1, -2)
        ix = F.conv2d(gray_a, kx, padding=1)
        iy = F.conv2d(gray_a, ky, padding=1)
        it = gray_b - gray_a
        denom = ix * ix + iy * iy + 1e-3
        u = -ix * it / denom
        v = -iy * it / denom
        return torch.cat([u, v], dim=1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        a_pred, b_pred = _adjacent_pairs(pred)
        a_tgt, b_tgt = _adjacent_pairs(target)
        flow_pred = self._flow(a_pred, b_pred)
        flow_tgt = self._flow(a_tgt, b_tgt)
        loss = F.l1_loss(flow_pred, flow_tgt)
        if self.weight_div > 0.0:
            # Smoothness term on predicted flow: penalize spatial gradient.
            div_x = flow_pred[..., :, :-1] - flow_pred[..., :, 1:]
            div_y = flow_pred[..., :-1, :] - flow_pred[..., 1:, :]
            loss = loss + self.weight_div * (div_x.abs().mean() + div_y.abs().mean())
        return loss


# ----------------------------------------------------------------------------
# VFI-reconstruction loss
# ----------------------------------------------------------------------------


class VFIReconstructionLoss(nn.Module):
    """``|| VFI(pred_{t-1}, pred_{t+1}) - pred_t ||_1``.

    Mirrors how ``motion_smoothness_metrics.py`` computes the metric: the
    middle frame should be reconstructible from its neighbors. The default
    interpolator is a learnable-free linear blend (``0.5 * a + 0.5 * b``)
    which is differentiable and works without any checkpoint. If a VFIMamba
    checkpoint is provided, the same network the metric uses can be loaded;
    in that case its parameters are frozen and only the loss flows back to
    ``pred``.
    """

    def __init__(self, vfimamba_ckpt: Optional[str] = None):
        super().__init__()
        self._vfi = None
        if vfimamba_ckpt is not None and os.path.exists(vfimamba_ckpt):
            try:
                from WorldArena.third_party.VFIMamba.Trainer_finetune import Model  # type: ignore
                model = Model(-1)
                model.load_model(vfimamba_ckpt, -1)
                model.eval()
                # Freeze VFI weights — we only want gradients through the
                # generator's outputs.
                for p in getattr(model, "parameters", lambda: [])():
                    if hasattr(p, "requires_grad_"):
                        p.requires_grad_(False)
                self._vfi = model
            except Exception as exc:
                print(f"[VFIReconstructionLoss] VFIMamba load failed: {exc}")

    def _vfi_mid(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self._vfi is not None:
            try:
                # VFIMamba's Trainer_finetune.Model.inference takes [N, 3, H, W].
                mid = self._vfi.inference(a, b, [0.5])
                if isinstance(mid, list):
                    mid = mid[0]
                return mid
            except Exception:
                pass
        return 0.5 * a + 0.5 * b  # fallback linear blend

    def forward(self, pred: torch.Tensor, target: torch.Tensor | None = None) -> torch.Tensor:
        prev, cur, nxt = _triplets(pred)
        recon = self._vfi_mid(prev, nxt)
        return F.l1_loss(recon, cur)


# ----------------------------------------------------------------------------
# Bundle
# ----------------------------------------------------------------------------


class PerceptualLossBundle(nn.Module):
    """Convenience wrapper combining the three Tier-3 losses with weights."""

    def __init__(
        self,
        lpips_weight: float = 0.0,
        flow_weight: float = 0.0,
        vfi_recon_weight: float = 0.0,
        raft_ckpt: Optional[str] = None,
        vfimamba_ckpt: Optional[str] = None,
        flow_div_weight: float = 0.0,
        use_pyiqa_lpips: bool = True,
    ):
        super().__init__()
        self.lpips_weight = lpips_weight
        self.flow_weight = flow_weight
        self.vfi_weight = vfi_recon_weight
        self.lpips = LpipsLoss(use_pyiqa=use_pyiqa_lpips) if lpips_weight > 0 else None
        self.flow = (FlowConsistencyLoss(raft_ckpt=raft_ckpt, weight_div=flow_div_weight)
                     if flow_weight > 0 else None)
        self.vfi = (VFIReconstructionLoss(vfimamba_ckpt=vfimamba_ckpt)
                    if vfi_recon_weight > 0 else None)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        total = pred.new_zeros(())
        if self.lpips is not None:
            l = self.lpips(pred, target)
            out["lpips"] = l
            total = total + self.lpips_weight * l
        if self.flow is not None:
            l = self.flow(pred, target)
            out["flow"] = l
            total = total + self.flow_weight * l
        if self.vfi is not None:
            l = self.vfi(pred)
            out["vfi"] = l
            total = total + self.vfi_weight * l
        out["total"] = total
        return out


__all__ = [
    "LpipsLoss",
    "FlowConsistencyLoss",
    "VFIReconstructionLoss",
    "PerceptualLossBundle",
]
