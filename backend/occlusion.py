"""
Sunglasses / lens-occlusion heuristic (pure numpy, no OpenCV or mediapipe).

Split out of eye_detection.py so it can be unit-tested anywhere, including
CI machines without a webcam or OpenCV installed. See the comment on
EyeConfig in config.py for the reasoning, calibration data and caveats.

Why a RATIO and not raw brightness
----------------------------------
The first version flagged "sunglasses" whenever the eye patch was darker
than a fixed value (80/255). That threshold was calibrated in one lighting
condition, so in a dimmer room bare eyes also fell below it: every frame
was flagged, every eye reading was discarded, PERCLOS hit ~100% and the
alert locked on RED. In a helmet that would happen at dusk or in a tunnel.

Dim light darkens the whole face roughly equally; a tinted lens darkens
only the eyes. So we compare the eye patch with a reference patch of bare
skin on the cheekbone just below each eye, and decide on the ratio
eye_mean / cheek_mean, which stays roughly constant as the room gets darker
or brighter. Absolute brightness is only used as a fallback when the cheek
patch can't be sampled, and as a "too dark for the camera to see anything"
floor.

`points` are landmark-like objects with normalised `.x` / `.y` attributes
(mediapipe NormalizedLandmark, or any stand-in with the same fields), in
the 6-point EAR order: p1 and p4 are the eye corners.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from config import EyeConfig

# Reasons the heuristic can distrust an eye reading (shown on the dashboard).
REASON_LENS_DARK = "lens_dark"        # eyes much darker than cheeks: tinted lens
REASON_LENS_BRIGHT = "lens_bright"    # eyes much brighter than cheeks: mirrored lens / glare
REASON_FLAT = "flat"                  # eye patch has almost no texture at all
REASON_TOO_DARK = "too_dark"          # whole face too dark for the camera to see
REASON_ABSOLUTE = "absolute_fallback" # cheek not sampled; old fixed thresholds fired


@dataclass
class OcclusionStats:
    eye_mean: float
    eye_std: float
    ref_mean: Optional[float] = None   # cheek brightness, None if not sampled
    ref_std: Optional[float] = None

    @property
    def ratio(self) -> Optional[float]:
        if self.ref_mean is None or self.ref_mean <= 1e-6:
            return None
        return self.eye_mean / self.ref_mean


def _box_stats(gray_frame, x0: float, x1: float, y0: float, y1: float, w: int, h: int):
    x0, x1 = max(0, int(x0)), min(w, int(x1))
    y0, y1 = max(0, int(y0)), min(h, int(y1))
    if x1 <= x0 or y1 <= y0:
        return None
    patch = gray_frame[y0:y1, x0:x1]
    if patch.size == 0:
        return None
    return float(patch.mean()), float(patch.std())


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
    return _box_stats(gray_frame, x_min - pad_x, x_max + pad_x, y_min - pad_y, y_max + pad_y, w, h)


def _cheek_patch_stats(gray_frame, points, w, h, cfg: EyeConfig):
    """Mean and std of a bare-skin patch on the cheekbone below one eye.

    Placed relative to the eye's own width so it scales with distance from
    the camera: centred `cheek_offset_eye_widths` eye-widths below the eye
    centre, far enough down to clear typical sunglasses lenses."""
    xs = [lm.x * w for lm in points]
    ys = [lm.y * h for lm in points]
    eye_w = max(xs) - min(xs)
    if eye_w < 2.0:
        return None
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys) + cfg.cheek_offset_eye_widths * eye_w
    half_w = 0.4 * eye_w
    half_h = 0.25 * eye_w
    return _box_stats(gray_frame, cx - half_w, cx + half_w, cy - half_h, cy + half_h, w, h)


def _avg(stats_list):
    stats_list = [s for s in stats_list if s is not None]
    if not stats_list:
        return None
    n = len(stats_list)
    return sum(s[0] for s in stats_list) / n, sum(s[1] for s in stats_list) / n


def _eye_region_stats(gray_frame, left_pts, right_pts, w, h) -> Optional[tuple]:
    """Average (mean, std) of pixel intensity across both eye patches.
    Returns None if neither patch could be sampled."""
    return _avg([_eye_patch_stats(gray_frame, pts, w, h) for pts in (left_pts, right_pts)])


def _occlusion_stats(gray_frame, left_pts, right_pts, w, h, cfg: EyeConfig) -> Optional[OcclusionStats]:
    """Eye-patch stats plus cheek reference stats, averaged over both sides.
    Returns None if no eye patch could be sampled."""
    eye = _eye_region_stats(gray_frame, left_pts, right_pts, w, h)
    if eye is None:
        return None
    ref = _avg([_cheek_patch_stats(gray_frame, pts, w, h, cfg) for pts in (left_pts, right_pts)])
    return OcclusionStats(
        eye_mean=eye[0], eye_std=eye[1],
        ref_mean=ref[0] if ref else None, ref_std=ref[1] if ref else None,
    )


def _occlusion_thresholds_fire(mean_val: float, std_val: float, cfg: EyeConfig) -> bool:
    """Original absolute-brightness rule. Now ONLY used as a fallback when
    the cheek reference patch can't be sampled (e.g. lower face off-frame).
    Fires on extreme uniformity OR an absolute too-dark / too-bright patch.
    Known weakness: sensitive to overall lighting -- see module docstring."""
    too_uniform = std_val < cfg.occlusion_std_threshold
    too_dark_or_bright = (
        mean_val < cfg.occlusion_dark_mean_threshold
        or mean_val > cfg.occlusion_bright_mean_threshold
    )
    return too_uniform or too_dark_or_bright


def _occlusion_decision(stats: OcclusionStats, cfg: EyeConfig) -> Tuple[bool, Optional[str]]:
    """Decide whether to distrust this frame's EAR. Returns (fires, reason)."""
    if stats.ratio is None:
        if _occlusion_thresholds_fire(stats.eye_mean, stats.eye_std, cfg):
            return True, REASON_ABSOLUTE
        return False, None

    # Whole face too dark: the camera genuinely can't see. Not sunglasses,
    # but the EAR is still meaningless, so it's distrusted with its own
    # reason (a real helmet needs IR illumination for night riding).
    if stats.ref_mean < cfg.occlusion_too_dark_floor:
        return True, REASON_TOO_DARK

    if stats.ratio < cfg.occlusion_dark_ratio_threshold:
        return True, REASON_LENS_DARK
    if stats.ratio > cfg.occlusion_bright_ratio_threshold:
        return True, REASON_LENS_BRIGHT

    # Texture relative to brightness (coefficient of variation), so a dim
    # but real eye isn't mistaken for a flat surface just because every
    # pixel value is smaller.
    cv = stats.eye_std / max(stats.eye_mean, 1e-6)
    if cv < cfg.occlusion_min_texture_cv:
        return True, REASON_FLAT

    return False, None


def _looks_like_occluded_lens(gray_frame, left_pts, right_pts, w, h, cfg: EyeConfig) -> bool:
    """Convenience wrapper: frame + landmarks -> should this EAR be distrusted?"""
    stats = _occlusion_stats(gray_frame, left_pts, right_pts, w, h, cfg)
    if stats is None:
        return False
    return _occlusion_decision(stats, cfg)[0]
