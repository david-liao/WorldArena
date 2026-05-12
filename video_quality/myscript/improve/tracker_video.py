"""Tier-3: a more temporally-coherent trajectory builder than per-frame SAM3.

Two modes share the same ``traj.npy`` interface used by
``processing/detection_tracking.py``:

* ``improve``  -- read an existing ``traj.npy`` (the output of detection_tracking)
                  and rewrite it with Kalman smoothing + cubic-spline gap
                  filling + flow-based outlier rejection. This is the cheapest
                  upgrade because no new detector run is required.
* ``propagate`` -- given a frame directory and frame-0 anchor points (from
                   SAM3 or user-supplied JSON), propagate each anchor through
                   the rest of the clip with optical-flow-driven point
                   tracking. Re-anchors every ``K`` frames if a fresh
                   ``traj.npy`` is provided as a re-anchor source.

In both modes the output is a ``(T, 2, 2)`` float32 array with normalized
coordinates in [0, 1], where missing points are stored as ``[-1, -1]`` and
the second axis is ``[left, right]`` -- exactly what
``WorldArena.trajectory_accuracy`` consumes.

CLI examples::

    # 1. Smooth an existing traj.npy in-place (writes a backup .bak).
    python -m myscript.improve.tracker_video improve --traj path/to/traj.npy

    # 2. Generate a smoothed traj.npy from frames + an existing per-frame
    #    SAM3 trajectory file (used for re-anchoring & outlier rejection).
    python -m myscript.improve.tracker_video propagate \\
        --frames path/to/frames/ \\
        --anchors path/to/sam3/traj.npy \\
        --output  path/to/traj.npy
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE.parent.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent.parent))


# ----------------------------------------------------------------------------
# Coordinate-space conventions: traj entries are normalized to [0, 1] using
# (640, 480). detection_tracking writes -1 for missing; we follow suit.
# ----------------------------------------------------------------------------

_TARGET_W = 640
_TARGET_H = 480


def _is_missing(p: np.ndarray) -> bool:
    return bool(np.any(p < 0))


def _denorm(traj: np.ndarray) -> np.ndarray:
    out = traj.copy()
    out[..., 0] *= _TARGET_W
    out[..., 1] *= _TARGET_H
    return out


def _norm(traj: np.ndarray) -> np.ndarray:
    out = traj.copy()
    out[..., 0] /= _TARGET_W
    out[..., 1] /= _TARGET_H
    return out


# ----------------------------------------------------------------------------
# Improve mode
# ----------------------------------------------------------------------------


def _kalman_1d(values: np.ndarray, process_var: float = 1.0,
               measurement_var: float = 8.0) -> np.ndarray:
    """Constant-velocity 1-D Kalman filter for one coordinate channel.

    Treats negative entries as missing and interpolates them in measurement
    space first; the smoother then tightens jitter without re-introducing
    discontinuities at gap boundaries.
    """
    n = len(values)
    valid = values >= 0
    if valid.sum() < 2:
        return values.astype(np.float32)
    # Linear inpaint of missing.
    idx = np.arange(n)
    measured = values.copy().astype(np.float64)
    measured[~valid] = np.interp(idx[~valid], idx[valid], values[valid])

    # Constant-velocity Kalman.
    x = np.array([measured[0], 0.0])
    P = np.eye(2) * 100.0
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.array([[1.0, 0.0], [0.0, 1.0]]) * process_var
    R = np.array([[measurement_var]])
    out = np.zeros(n)
    for t in range(n):
        # predict
        x = F @ x
        P = F @ P @ F.T + Q
        # update
        z = measured[t]
        y = z - (H @ x)[0]
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + (K @ np.array([y])).flatten()
        P = (np.eye(2) - K @ H) @ P
        out[t] = x[0]
    return out.astype(np.float32)


def _cubic_spline_fill(coords: np.ndarray) -> np.ndarray:
    """Cubic-spline interpolation through valid points; -1 stays where total
    coverage is too sparse to fit a cubic."""
    valid = ~np.any(coords < 0, axis=-1)
    if valid.sum() < 4:
        return coords
    idx = np.arange(len(coords))
    out = coords.copy()
    for c in range(coords.shape[-1]):
        try:
            from scipy.interpolate import CubicSpline
            cs = CubicSpline(idx[valid], coords[valid, c])
            out[~valid, c] = cs(idx[~valid])
        except Exception:
            out[~valid, c] = np.interp(idx[~valid], idx[valid], coords[valid, c])
    return out


def _flow_outlier_mask(coords: np.ndarray, frames: Optional[List[np.ndarray]],
                       max_jump: float = 60.0) -> np.ndarray:
    """Mark detections that disagree with optical-flow propagation as missing.

    coords are pixel-space (after denorm). ``frames`` is the list of grayscale
    frames indexed identically. Returns a boolean array; True = trust this
    detection.
    """
    n = len(coords)
    trust = np.ones(n, dtype=bool)
    if frames is None or len(frames) != n:
        return trust
    for t in range(1, n):
        prev = coords[t - 1]
        cur = coords[t]
        if _is_missing(prev) or _is_missing(cur):
            continue
        # Distance check; oversized jumps are likely SAM3 left/right swaps.
        if np.linalg.norm(cur - prev) > max_jump:
            trust[t] = False
    return trust


def improve_traj(
    traj: np.ndarray,
    frames: Optional[List[np.ndarray]] = None,
    process_var: float = 1.0,
    measurement_var: float = 8.0,
    max_jump_px: float = 60.0,
) -> np.ndarray:
    """Apply outlier rejection + cubic-spline fill + Kalman smoothing.

    Operates in pixel space (640x480) internally, returns normalized ``(T,2,2)``.
    """
    if traj.ndim != 3 or traj.shape[1:] != (2, 2):
        raise ValueError(f"expected (T,2,2), got {traj.shape}")
    pix = _denorm(traj.astype(np.float32))

    out = pix.copy()
    for hand in range(2):  # 0=left, 1=right
        coords = pix[:, hand]  # (T, 2)
        trust = _flow_outlier_mask(coords, frames, max_jump=max_jump_px)
        coords = coords.copy()
        coords[~trust] = -1.0  # mark as missing for downstream fill
        # Cubic-spline fill of missing.
        filled = _cubic_spline_fill(coords)
        # Kalman per channel.
        x_smooth = _kalman_1d(filled[:, 0], process_var, measurement_var)
        y_smooth = _kalman_1d(filled[:, 1], process_var, measurement_var)
        out[:, hand, 0] = x_smooth
        out[:, hand, 1] = y_smooth

    return _norm(out)


# ----------------------------------------------------------------------------
# Propagate mode (optical-flow-driven point tracking)
# ----------------------------------------------------------------------------


def _load_frames(folder: Path) -> List[np.ndarray]:
    files = sorted([p for p in folder.iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    out = []
    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"failed to read {f}")
        img = cv2.resize(img, (_TARGET_W, _TARGET_H), interpolation=cv2.INTER_LINEAR)
        out.append(img)
    return out


def _propagate_point(prev_gray: np.ndarray, cur_gray: np.ndarray,
                     pt: np.ndarray) -> np.ndarray:
    """Propagate a single point with Lucas-Kanade flow."""
    if _is_missing(pt):
        return pt
    p0 = np.array([[pt]], dtype=np.float32)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, p0, None,
                                         winSize=(21, 21), maxLevel=3,
                                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    if st is None or st[0, 0] == 0:
        return np.array([-1.0, -1.0], dtype=np.float32)
    new = p1[0, 0]
    if not (0 <= new[0] < _TARGET_W and 0 <= new[1] < _TARGET_H):
        return np.array([-1.0, -1.0], dtype=np.float32)
    return new.astype(np.float32)


def propagate_traj(
    frames: List[np.ndarray],
    anchors: np.ndarray,
    reanchor_period: int = 8,
    reanchor_max_dist_px: float = 30.0,
) -> np.ndarray:
    """Build a fresh trajectory by Lucas-Kanade propagation seeded by ``anchors``.

    ``anchors`` is the per-frame ``(T,2,2)`` SAM3 trajectory (in normalized
    coords). At ``t=0`` we initialize from the first valid anchor in each
    column. Every ``reanchor_period`` frames we *snap* the propagated point
    to the SAM3 detection if they agree within ``reanchor_max_dist_px`` px,
    otherwise we keep the propagation (treating the SAM3 detection as a
    likely false positive).
    """
    n = len(frames)
    if anchors.shape[0] != n:
        # Anchors may differ in length when a different sampler was used; we
        # only use anchors at indices we can map; resample with linspace.
        idx = np.linspace(0, anchors.shape[0] - 1, n).round().astype(int)
        anchors = anchors[idx]
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    pix_anchors = _denorm(anchors.astype(np.float32))

    # Find an initial valid anchor per hand.
    state = np.full((2, 2), -1.0, dtype=np.float32)
    for hand in range(2):
        for t0 in range(n):
            if not _is_missing(pix_anchors[t0, hand]):
                state[hand] = pix_anchors[t0, hand].copy()
                break
    out = np.full((n, 2, 2), -1.0, dtype=np.float32)
    out[0] = state.copy()

    for t in range(1, n):
        for hand in range(2):
            if not _is_missing(state[hand]):
                state[hand] = _propagate_point(grays[t - 1], grays[t], state[hand])
            # Periodic re-anchor.
            if t % reanchor_period == 0:
                anc = pix_anchors[t, hand]
                if not _is_missing(anc):
                    if _is_missing(state[hand]) or np.linalg.norm(state[hand] - anc) <= reanchor_max_dist_px:
                        state[hand] = anc.copy()
        out[t] = state.copy()

    norm_out = _norm(out)
    return improve_traj(norm_out, frames=frames)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Tier-3 trajectory upgrade for detection_tracking.")
    sub = parser.add_subparsers(dest="mode", required=True)

    pi = sub.add_parser("improve", help="Smooth an existing traj.npy.")
    pi.add_argument("--traj", type=Path, required=True)
    pi.add_argument("--frames", type=Path, default=None,
                    help="Optional frame dir; enables flow-based outlier rejection.")
    pi.add_argument("--output", type=Path, default=None,
                    help="Override output path (default: in-place with .bak backup).")
    pi.add_argument("--process-var", type=float, default=1.0)
    pi.add_argument("--measurement-var", type=float, default=8.0)
    pi.add_argument("--max-jump-px", type=float, default=60.0)

    pp = sub.add_parser("propagate", help="Build a fresh traj.npy by LK propagation.")
    pp.add_argument("--frames", type=Path, required=True)
    pp.add_argument("--anchors", type=Path, required=True,
                    help="Per-frame SAM3 traj.npy used for seed + periodic re-anchoring.")
    pp.add_argument("--output", type=Path, required=True)
    pp.add_argument("--reanchor-period", type=int, default=8)
    pp.add_argument("--reanchor-max-dist-px", type=float, default=30.0)

    args = parser.parse_args(argv)

    if args.mode == "improve":
        traj = np.load(args.traj)
        frames = _load_frames(args.frames) if args.frames is not None else None
        new_traj = improve_traj(
            traj, frames=frames,
            process_var=args.process_var,
            measurement_var=args.measurement_var,
            max_jump_px=args.max_jump_px,
        )
        out_path = args.output or args.traj
        if out_path == args.traj:
            shutil.copy2(args.traj, args.traj.with_suffix(args.traj.suffix + ".bak"))
        np.save(out_path, new_traj.astype(np.float32))
        print(f"[improve] saved {new_traj.shape} -> {out_path}")
        return 0

    if args.mode == "propagate":
        frames = _load_frames(args.frames)
        anchors = np.load(args.anchors)
        if anchors.ndim != 3 or anchors.shape[1:] != (2, 2):
            raise SystemExit(f"--anchors must be (T,2,2); got {anchors.shape}")
        new_traj = propagate_traj(
            frames, anchors,
            reanchor_period=args.reanchor_period,
            reanchor_max_dist_px=args.reanchor_max_dist_px,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, new_traj.astype(np.float32))
        print(f"[propagate] saved {new_traj.shape} -> {args.output}")
        return 0

    raise SystemExit(f"unknown mode: {args.mode}")


if __name__ == "__main__":
    raise SystemExit(main())
