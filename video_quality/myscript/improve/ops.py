"""Pure-numpy / cv2 operations used by the Tier-1 post-processing pipeline.

Every function takes & returns ``uint8 H x W x 3`` BGR frames so that the
ordering of operations in ``postprocess.py`` is irrelevant. Functions are
deterministic and side-effect free, which is required for the multi-gid
consistency goal: applying the same params to the same input MUST yield the
same output bytes regardless of how many times it is called.
"""
from __future__ import annotations

from typing import List, Sequence

import cv2
import numpy as np


def _ensure_size(frame: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    h, w = frame.shape[:2]
    th, tw = target_hw
    if (h, w) != (th, tw):
        return cv2.resize(frame, (tw, th), interpolation=cv2.INTER_CUBIC)
    return frame


# ----------------------------------------------------------------------------
# Step 1: frame-count alignment to GT
# ----------------------------------------------------------------------------


def align_frame_count(frames: Sequence[np.ndarray], target: int) -> List[np.ndarray]:
    """Resample ``frames`` so its length equals ``target``.

    - When source has more frames: even ``np.linspace`` indexing (the same
      sampling depth_accuracy uses internally).
    - When source has fewer frames: linear blend between adjacent source
      frames at the requested time positions. This is a deliberately cheap
      fallback so the dependency surface stays minimal; users who care about
      motion_smoothness should use ``vfi_interp.py`` instead.
    """
    n = len(frames)
    if n == target:
        return list(frames)
    if n < 2:
        return [frames[0]] * target

    # Map output index t in [0, target-1] to source position s in [0, n-1].
    src_pos = np.linspace(0, n - 1, target)
    out: List[np.ndarray] = []
    for s in src_pos:
        i0 = int(np.floor(s))
        i1 = min(i0 + 1, n - 1)
        a = float(s - i0)
        if a == 0.0 or i0 == i1:
            out.append(frames[i0].copy())
        else:
            blended = ((1.0 - a) * frames[i0].astype(np.float32)
                       + a * frames[i1].astype(np.float32))
            out.append(np.clip(blended, 0, 255).astype(np.uint8))
    return out


# ----------------------------------------------------------------------------
# Step 2: first-frame replacement with a short cross-fade
# ----------------------------------------------------------------------------


def replace_first_frame(
    frames: Sequence[np.ndarray],
    gt_first: np.ndarray,
    crossfade_len: int = 3,
) -> List[np.ndarray]:
    """Replace ``frames[0]`` with ``gt_first`` and cross-fade the next frames.

    With ``crossfade_len=3`` the schedule is:
        frame[0] = gt_first
        frame[1] = 0.30 * gt_first + 0.70 * orig[1]
        frame[2] = 0.15 * gt_first + 0.85 * orig[2]
        frame[3+] = orig[3+]

    The fade weights decay linearly so the transition stays visually monotone
    without introducing per-channel hue shifts that CLIP / DINO would notice.
    """
    if not frames:
        return []
    out = [f.copy() for f in frames]
    th, tw = out[0].shape[:2]
    gt_first = _ensure_size(gt_first, (th, tw))
    out[0] = gt_first.copy()

    for i in range(1, min(crossfade_len, len(out))):
        # Linear decay: weight on gt at index i is (crossfade_len - i) / crossfade_len.
        w_gt = (crossfade_len - i) / float(crossfade_len)
        w_pred = 1.0 - w_gt
        blended = (w_gt * gt_first.astype(np.float32)
                   + w_pred * out[i].astype(np.float32))
        out[i] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


# ----------------------------------------------------------------------------
# Step 3: Reinhard color transfer in LAB, computed once against GT first frame
# ----------------------------------------------------------------------------


def _lab_stats(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    mean = lab.reshape(-1, 3).mean(axis=0)
    std = lab.reshape(-1, 3).std(axis=0) + 1e-6
    return mean, std


def reinhard_color_match(
    frames: Sequence[np.ndarray],
    target_frame: np.ndarray,
    strength: float = 1.0,
) -> List[np.ndarray]:
    """Reinhard color transfer in LAB.

    The target statistics are computed once from ``target_frame`` and applied
    uniformly to every frame so multi-gid consistency is preserved (different
    gids only need to receive the same target_frame to come out identical).

    ``strength`` in [0,1] linearly blends original and matched frames; values
    < 1 are useful when GT has a very different exposure that would otherwise
    push the prediction off-distribution for CLIP / DINO.
    """
    target_mean, target_std = _lab_stats(target_frame)
    out: List[np.ndarray] = []
    for f in frames:
        lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32)
        flat = lab.reshape(-1, 3)
        src_mean = flat.mean(axis=0)
        src_std = flat.std(axis=0) + 1e-6
        # (x - src_mean) / src_std * target_std + target_mean
        matched = (lab - src_mean) / src_std * target_std + target_mean
        if strength < 1.0:
            matched = strength * matched + (1.0 - strength) * lab
        matched = np.clip(matched, 0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(matched, cv2.COLOR_LAB2BGR)
        out.append(bgr)
    return out


# ----------------------------------------------------------------------------
# Step 4: light unsharp + contrast / saturation tweak
# ----------------------------------------------------------------------------


def unsharp_and_eq(
    frames: Sequence[np.ndarray],
    unsharp_amount: float = 0.5,
    unsharp_radius: int = 5,
    contrast: float = 1.05,
    saturation: float = 1.05,
    gamma: float = 1.0,
) -> List[np.ndarray]:
    """Apply unsharp masking and a mild contrast / saturation lift.

    Defaults match ``unsharp=5:5:0.5,eq=contrast=1.05:saturation=1.05`` from the
    plan. ``unsharp_amount`` capped at 0.5 because the motion_smoothness metric
    uses VFIMamba+SSIM and over-sharp frames degrade SSIM reconstruction.
    """
    out: List[np.ndarray] = []
    sigma = max(unsharp_radius / 3.0, 0.8)
    for f in frames:
        if unsharp_amount > 0.0:
            blurred = cv2.GaussianBlur(f, (0, 0), sigmaX=sigma, sigmaY=sigma)
            sharp = cv2.addWeighted(f, 1.0 + unsharp_amount, blurred, -unsharp_amount, 0)
        else:
            sharp = f

        if contrast != 1.0 or gamma != 1.0:
            mean = 128.0
            arr = sharp.astype(np.float32)
            arr = (arr - mean) * contrast + mean
            if gamma != 1.0:
                # Apply gamma in [0,1] domain.
                arr = np.clip(arr, 0, 255) / 255.0
                arr = np.power(arr, gamma) * 255.0
            sharp = np.clip(arr, 0, 255).astype(np.uint8)

        if saturation != 1.0:
            hsv = cv2.cvtColor(sharp, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0, 255)
            sharp = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        out.append(sharp)
    return out


# ----------------------------------------------------------------------------
# Step 5: tail EMA stabilization
# ----------------------------------------------------------------------------


def tail_ema_stabilize(
    frames: Sequence[np.ndarray],
    n_tail: int = 5,
    alpha: float = 0.7,
) -> List[np.ndarray]:
    """Apply an exponential moving average to the last ``n_tail`` frames.

    ``alpha`` is the weight on the previous (smoothed) frame, so larger alpha
    means stronger anchoring to the last "trusted" frame. The trusted frame is
    ``frames[-(n_tail+1)]``; if the video has at most ``n_tail`` frames we
    fall back to the first frame as the anchor.
    """
    n = len(frames)
    if n_tail <= 0 or n == 0 or alpha <= 0.0:
        return [f.copy() for f in frames]
    out = [f.copy() for f in frames]
    anchor_idx = max(0, n - n_tail - 1)
    smoothed = out[anchor_idx].astype(np.float32)
    start = max(1, n - n_tail)
    for i in range(start, n):
        cur = out[i].astype(np.float32)
        smoothed = alpha * smoothed + (1.0 - alpha) * cur
        out[i] = np.clip(smoothed, 0, 255).astype(np.uint8)
    return out
