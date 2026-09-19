"""
Webcam-based Eye Aspect Ratio (EAR) tracker.

Stands in for the ESP32-CAM eye/face channel described in the IDP document
until real camera hardware is mounted in the helmet. Uses mediapipe's face
landmark model (478 points, same topology as the old "refine_landmarks"
FaceMesh) to locate the eyes and computes EAR per frame:

    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)

Lower EAR ~= more closed eyes. This module only produces (ear, timestamp)
samples -- all "is this drowsy" logic lives in decision_engine.py, so
swapping this out for a real ESP32-CAM stream later just means writing a
different producer of the same (ear, timestamp) samples.

mediapipe has two ways to get face landmarks depending on version:
  - Legacy `mediapipe.solutions.face_mesh` (older installs, <=0.10.2x)
  - Modern Tasks API `mediapipe.tasks.python.vision.FaceLandmarker`
    (current installs, 0.10.30+ / 1.0.x) -- this needs a small model file
    (face_landmarker.task, ~4-9 MB) which is downloaded automatically on
    first run and cached in ~/.cache/smart_helmet/.

This module tries the legacy API first and falls back to the Tasks API
automatically, so it works either way without you needing to know which
mediapipe version you have.

Requires: opencv-python, mediapipe
"""

from __future__ import annotations

import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Iterator, Optional

import cv2
import numpy as np

from config import EyeConfig

# Landmark indices for the left/right eye (6 points each, standard EAR
# formula). Same indices work for both the legacy FaceMesh (with
# refine_landmarks=True) and the modern FaceLandmarker -- they share the
# same 478-point topology.
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
_MODEL_DIR = os.path.join(os.path.expanduser("~"), ".cache", "smart_helmet")
_MODEL_PATH = os.path.join(_MODEL_DIR, "face_landmarker.task")


def _dist(a, b) -> float:
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def _eye_aspect_ratio(points, w, h) -> float:
    pts = [(lm.x * w, lm.y * h) for lm in points]
    p1, p2, p3, p4, p5, p6 = pts
    vertical = _dist(p2, p6) + _dist(p3, p5)
    horizontal = 2.0 * _dist(p1, p4)
    if horizontal == 0:
        return 0.3  # neutral fallback, avoids div-by-zero on a bad frame
    return vertical / horizontal


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


def _ensure_model_downloaded() -> str:
    os.makedirs(_MODEL_DIR, exist_ok=True)
    if not os.path.exists(_MODEL_PATH):
        print(f"[eye_detection] Downloading face landmark model (one-time, ~4MB) to {_MODEL_PATH} ...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("[eye_detection] Model downloaded.")
    return _MODEL_PATH


@dataclass
class EyeFrameResult:
    timestamp: float
    ear: Optional[float]      # None if no face detected, or occlusion suspected
    face_found: bool
    occlusion_suspected: bool = False  # True if sunglasses/lens occlusion heuristic fired
    eye_patch_mean: Optional[float] = None  # avg pixel brightness under the eyes (0-255)
    eye_patch_std: Optional[float] = None   # avg pixel texture/uniformity under the eyes
    frame: Optional["np.ndarray"] = None  # BGR frame, for optional preview windows


class _LegacyBackend:
    """mediapipe.solutions.face_mesh -- older mediapipe installs."""

    def __init__(self):
        import mediapipe as mp

        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process(self, rgb_frame, timestamp_ms: int):
        result = self._face_mesh.process(rgb_frame)
        if not result.multi_face_landmarks:
            return None
        return result.multi_face_landmarks[0].landmark

    def close(self):
        self._face_mesh.close()


class _TasksBackend:
    """mediapipe.tasks.python.vision.FaceLandmarker -- current mediapipe installs."""

    def __init__(self):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        model_path = _ensure_model_downloaded()
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self._last_ts_ms = -1

    def process(self, rgb_frame, timestamp_ms: int):
        import mediapipe as mp

        # The Tasks API requires strictly increasing timestamps per stream.
        if timestamp_ms <= self._last_ts_ms:
            timestamp_ms = self._last_ts_ms + 1
        self._last_ts_ms = timestamp_ms

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.face_landmarks:
            return None
        return result.face_landmarks[0]

    def close(self):
        self._landmarker.close()


def _make_backend():
    try:
        import mediapipe as mp

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            return _LegacyBackend()
    except Exception:
        pass
    return _TasksBackend()


class WebcamEyeTracker:
    """Opens the default (or specified) webcam and yields EAR samples.

    Usage:
        tracker = WebcamEyeTracker()
        for result in tracker.stream():
            ear = result.ear
            ...
        tracker.release()

    Or pull single frames with `tracker.read_once()` if you're driving your
    own loop (e.g. from the FastAPI background task).
    """

    def __init__(self, camera_index: int = 0, eye_config: Optional[EyeConfig] = None):
        self.camera_index = camera_index
        self.eye_cfg = eye_config or EyeConfig()
        self._cap: Optional[cv2.VideoCapture] = None
        self._backend = _make_backend()
        self._start_time = time.time()

    def open(self) -> None:
        if self._cap is None:
            self._cap = cv2.VideoCapture(self.camera_index)
            if not self._cap.isOpened():
                raise RuntimeError(
                    f"Could not open webcam at index {self.camera_index}. "
                    "Check that a camera is connected and not in use by another app."
                )

    def read_once(self) -> EyeFrameResult:
        self.open()
        assert self._cap is not None
        ok, frame = self._cap.read()
        ts = time.time()
        if not ok:
            return EyeFrameResult(timestamp=ts, ear=None, face_found=False)

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp_ms = int((ts - self._start_time) * 1000)

        landmarks = self._backend.process(rgb, timestamp_ms)
        if landmarks is None:
            return EyeFrameResult(timestamp=ts, ear=None, face_found=False, frame=frame)

        left_pts = [landmarks[i] for i in LEFT_EYE]
        right_pts = [landmarks[i] for i in RIGHT_EYE]
        ear = (_eye_aspect_ratio(left_pts, w, h) + _eye_aspect_ratio(right_pts, w, h)) / 2.0

        # Always compute the raw eye-patch pixel stats (cheap) so they're
        # available for calibration/debugging even if the occlusion
        # heuristic itself is disabled.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        stats = _eye_region_stats(gray, left_pts, right_pts, w, h)
        patch_mean, patch_std = stats if stats is not None else (None, None)

        if self.eye_cfg.enable_occlusion_heuristic and stats is not None:
            if _occlusion_thresholds_fire(patch_mean, patch_std, self.eye_cfg):
                # Landmarks were found, but the eye region itself looks like
                # a flat lens rather than a real eyelid -- the EAR computed
                # above is not trustworthy, so treat this frame the same as
                # "no face detected" rather than reporting a confident but
                # meaningless number.
                return EyeFrameResult(
                    timestamp=ts, ear=None, face_found=True,
                    occlusion_suspected=True,
                    eye_patch_mean=patch_mean, eye_patch_std=patch_std,
                    frame=frame,
                )

        return EyeFrameResult(
            timestamp=ts, ear=ear, face_found=True,
            eye_patch_mean=patch_mean, eye_patch_std=patch_std,
            frame=frame,
        )

    def stream(self) -> Iterator[EyeFrameResult]:
        self.open()
        while True:
            yield self.read_once()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._backend.close()


if __name__ == "__main__":
    # Quick manual test: run `python eye_detection.py` on a machine with a
    # webcam to see live EAR printed to the console, and a preview window.
    # Press 'q' to quit.
    tracker = WebcamEyeTracker()
    try:
        for res in tracker.stream():
            if res.frame is not None:
                label = f"EAR: {res.ear:.3f}" if res.ear is not None else "No face"
                cv2.putText(
                    res.frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0) if res.ear else (0, 0, 255), 2,
                )
                cv2.imshow("Eye tracker (q to quit)", res.frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        tracker.release()
        cv2.destroyAllWindows()
