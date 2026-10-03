"""
Sunglasses / lens-occlusion heuristic (pure numpy, no OpenCV or mediapipe).

Split out of eye_detection.py so it can be unit-tested anywhere, including
CI machines without a webcam or OpenCV installed. See the long comment on
EyeConfig in config.py for the reasoning, calibration data and caveats.

`points` are landmark-like objects with normalised `.x` / `.y` attributes
(mediapipe NormalizedLandmark, or any stand-in with the same fields).
"""

from __future__ import annotations

from typing import Optional

from config import EyeConfig


def _eye_patch_stats(gray_frame, points, w, h, pad_ratio: float = 0.4):
    """Mean and std-dev of pixel intensity in the small box around an eye's
    landmarks (padded a bit to include the surrounding lid/lens area).
    Returns None if the box would be degenerate (off-frame, zero-size)."""
    xs = [lm.x * w for lm in points]
    ys = [lm.y * h for lm in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    pad_x = max(2.0, (x_max - x_min) * pad_ratio)
    pad_y = max(2.0, (y_max - y_min) * pad_ratio)
    x0, x1 = max(0, int(x_min - pad_x)), min(w, int(x_max + pad_x))
    y0, y1 = max(0, int(y_min - pad_y)), min(h, int(y_max + pad_y))
    if x1 <= x0 or y1 <= y0:
        return None
    patch = gray_frame[y0:y1, x0:x1]
    if patch.size == 0:
        return None
    return float(patch.mean()), float(patch.std())


def _eye_region_stats(gray_frame, left_pts, right_pts, w, h) -> Optional[tuple]:
    """Average (mean, std) of pixel intensity across both eye patches.
    Returns None if neither patch could be sampled."""
    stats = []
    for pts in (left_pts, right_pts):
        s = _eye_patch_stats(gray_frame, pts, w, h)
        if s is not None:
            stats.append(s)
    if not stats:
        return None
    mean_avg = sum(s[0] for s in stats) / len(stats)
    std_avg = sum(s[1] for s in stats) / len(stats)
    return mean_avg, std_avg


def _occlusion_thresholds_fire(mean_val: float, std_val: float, cfg: EyeConfig) -> bool:
    """Given already-computed (mean, std) for the eye region, decide if the
    occlusion heuristic fires.

    Originally this required BOTH "unusually uniform" (low std) AND
    "unusually dark/bright" (mean) to fire, on the theory that a sunglasses
    lens is flatter than a real eyelid. In practice (see calibrate_occlusion.py
    output), many webcams pick up enough sensor noise/reflection that a lens
    doesn't read as meaningfully flatter than a real eyelid -- std alone
    often can't separate them -- while brightness alone usually can (tinted
    lenses read noticeably darker than skin/sclera). So this fires on
    brightness OR extreme uniformity, whichever the calibration data
    actually supports for your setup. Trade-off: this makes the heuristic
    more sensitive to poor lighting alone (a genuinely dim room could read
    as "dark enough" even with real eyes) -- if that turns out to be a
    problem after calibrating, raise occlusion_dark_mean_threshold rather
    than reverting to the AND logic, which effectively disabled the
    heuristic entirely once brightness was the only reliable signal.
    """
    too_uniform = std_val < cfg.occlusion_std_threshold
    too_dark_or_bright = (
        mean_val < cfg.occlusion_dark_mean_threshold
        or mean_val > cfg.occlusion_bright_mean_threshold
    )
    return too_uniform or too_dark_or_bright


def _looks_like_occluded_lens(gray_frame, left_pts, right_pts, w, h, cfg: EyeConfig) -> bool:
    """Rough sunglasses/occlusion heuristic (see config.py for the reasoning
    and caveats), computed directly from a frame + landmarks."""
    stats = _eye_region_stats(gray_frame, left_pts, right_pts, w, h)
    if stats is None:
        return False
    mean_avg, std_avg = stats
    return _occlusion_thresholds_fire(mean_avg, std_avg, cfg)
