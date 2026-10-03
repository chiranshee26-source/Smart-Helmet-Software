"""
Unit tests for the sunglasses / lens-occlusion heuristic (backend/occlusion.py).
No camera, OpenCV or mediapipe needed -- synthetic grayscale frames and
stand-in landmark objects only.

Run with: python -m pytest tests/ -v   (from the project root)
"""

import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from config import EyeConfig  # noqa: E402
from occlusion import (  # noqa: E402
    _eye_patch_stats,
    _looks_like_occluded_lens,
    _occlusion_thresholds_fire,
)

W, H = 640, 480
CFG = EyeConfig()


def _eye_landmarks(cx, cy, half_w=0.03, half_h=0.012):
    """Six EAR-style landmarks around a normalised eye centre."""
    pts = [
        (cx - half_w, cy),
        (cx - half_w / 2, cy - half_h),
        (cx + half_w / 2, cy - half_h),
        (cx + half_w, cy),
        (cx + half_w / 2, cy + half_h),
        (cx - half_w / 2, cy + half_h),
    ]
    return [SimpleNamespace(x=x, y=y) for x, y in pts]


LEFT = _eye_landmarks(0.40, 0.45)
RIGHT = _eye_landmarks(0.60, 0.45)


def _frame(mean, std, seed=0):
    """Grayscale frame with the given brightness and texture."""
    rng = np.random.default_rng(seed)
    img = rng.normal(mean, std, size=(H, W)) if std > 0 else np.full((H, W), float(mean))
    return np.clip(img, 0, 255).astype(np.uint8)


# --- Threshold logic, using the calibration readings recorded in config.py ---

def test_calibrated_open_eyes_not_flagged():
    assert not _occlusion_thresholds_fire(106.1, 29.2, CFG)


def test_calibrated_closed_eyes_not_flagged():
    # Closed eyes must NOT read as occlusion, or a real eye closure would be
    # reported as "can't see" instead of as a closed-eye EAR reading.
    assert not _occlusion_thresholds_fire(96.9, 30.9, CFG)


def test_calibrated_sunglasses_flagged():
    assert _occlusion_thresholds_fire(63.4, 31.4, CFG)


def test_bright_reflective_lens_flagged():
    assert _occlusion_thresholds_fire(CFG.occlusion_bright_mean_threshold + 10, 30.0, CFG)


def test_flat_uniform_patch_flagged_even_at_normal_brightness():
    assert _occlusion_thresholds_fire(106.0, CFG.occlusion_std_threshold - 5, CFG)


def test_threshold_boundaries():
    just_ok = CFG.occlusion_dark_mean_threshold + 0.5
    assert not _occlusion_thresholds_fire(just_ok, 30.0, CFG)
    assert _occlusion_thresholds_fire(CFG.occlusion_dark_mean_threshold - 0.5, 30.0, CFG)


# --- Full pixel path: frame + landmarks -> decision ---

def test_textured_normal_eye_region_not_flagged():
    frame = _frame(mean=106, std=29)
    assert not _looks_like_occluded_lens(frame, LEFT, RIGHT, W, H, CFG)


def test_dark_tinted_lens_region_flagged():
    frame = _frame(mean=106, std=29)
    # Paint dark, low-texture "lenses" over both eye regions.
    for cx in (0.40, 0.60):
        x0, x1 = int((cx - 0.06) * W), int((cx + 0.06) * W)
        y0, y1 = int(0.40 * H), int(0.50 * H)
        frame[y0:y1, x0:x1] = _frame(mean=55, std=8, seed=1)[y0:y1, x0:x1]
    assert _looks_like_occluded_lens(frame, LEFT, RIGHT, W, H, CFG)


def test_patch_stats_match_known_region():
    frame = np.full((H, W), 100, dtype=np.uint8)
    mean, std = _eye_patch_stats(frame, LEFT, W, H)
    assert abs(mean - 100) < 1e-6 and std < 1e-6


def test_off_frame_landmarks_do_not_crash_or_flag():
    off = _eye_landmarks(1.5, 1.5)  # both eyes entirely outside the frame
    frame = _frame(mean=106, std=29)
    assert _eye_patch_stats(frame, off, W, H) is None
    assert _looks_like_occluded_lens(frame, off, off, W, H, CFG) is False


def test_one_eye_off_frame_uses_the_other():
    off = _eye_landmarks(1.5, 1.5)
    dark = _frame(mean=50, std=29)
    assert _looks_like_occluded_lens(dark, LEFT, off, W, H, CFG)


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"OK: {test.__name__}")
    print(f"\n{len(tests)} tests passed.")
