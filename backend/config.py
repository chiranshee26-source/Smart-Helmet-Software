"""
Calibratable thresholds for the drowsiness decision engine.

The IDP document is explicit that timings/thresholds must be calibrated and
validated experimentally, not treated as universal safety limits. Every
number that should eventually be tuned against real test data lives in this
one file so nothing is buried inside logic code.
"""

from dataclasses import dataclass, field


@dataclass
class EyeConfig:
    # EAR below this is considered "closed". Typical open-eye EAR ~0.25-0.35,
    # closed-eye EAR ~0.1-0.2. Calibrate per-camera/per-person if possible.
    ear_closed_threshold: float = 0.21

    # Rolling window (seconds) over which PERCLOS (percentage of time eyes
    # are closed) is computed.
    perclos_window_seconds: float = 15.0

    # PERCLOS thresholds that map to a "moderate" / "severe" eye-drowsiness
    # score. These feed the fusion step, not the final alert level directly.
    perclos_moderate: float = 0.15   # 15% of the window with eyes closed
    perclos_severe: float = 0.40     # 40% of the window with eyes closed

    # --- Sunglasses / occlusion heuristic ---
    # mediapipe's landmark model still predicts eye-region points even when
    # sunglasses cover the eyes (it infers the full face shape, it doesn't
    # know it can't actually see through the lenses), so a "no face
    # detected" check does NOT catch this case -- the EAR just becomes a
    # confident-looking but meaningless number traced off the lens/frame
    # shape. As a rough mitigation, we look at the actual pixels in the
    # small patch under each eye: a real open or closed eyelid has some
    # texture (skin, lashes, a highlight on the sclera); a sunglasses lens
    # is usually a much flatter, more uniform patch -- either uniformly
    # dark (tinted) or uniformly bright (reflective/mirrored). When a patch
    # looks that uniform OR unusually dark/bright, we treat that frame's EAR
    # as unreliable (same as "no face detected") rather than trusting it.
    # (Originally this required uniform AND dark/bright together, but
    # calibration showed texture/std often can't separate real eyes from a
    # lens on typical webcams -- brightness alone is the reliable signal --
    # so it's OR'd instead. See _occlusion_thresholds_fire in
    # eye_detection.py.) This is a coarse heuristic, not real occlusion
    # detection -- it is not guaranteed to catch every case (documented as
    # a known limitation either way).
    #
    # Values below calibrated via calibrate_occlusion.py against this
    # user's actual camera/lighting/sunglasses on 2026-09-19:
    #   eyes open (no glasses):   brightness mean=106.1, texture std=29.2
    #   eyes closed (no glasses): brightness mean=96.9,  texture std=30.9
    #   sunglasses on:            brightness mean=63.4,  texture std=31.4
    # Texture (std) didn't separate cleanly (all three ~29-31), so this is
    # effectively driven by the brightness thresholds below.
    enable_occlusion_heuristic: bool = True
    occlusion_std_threshold: float = 17.5         # below this = "too uniform" (0-255 grayscale)
    occlusion_dark_mean_threshold: float = 80.1   # below this = "too dark"
    occlusion_bright_mean_threshold: float = 180.6  # above this = "too bright/reflective"


@dataclass
class HeadConfig:
    # A "nod event" is registered when head pitch deviates from the rider's
    # recent resting pitch (a slowly-adapting baseline) by at least this many
    # degrees. Measured against the baseline -- NOT sample-to-sample -- so a
    # slow droop is caught the same as a fast one, and the result doesn't
    # depend on the IMU sample rate (5 Hz simulator vs 50-100 Hz MPU6050).
    nod_pitch_delta_threshold: float = 18.0

    # Hysteresis: once a nod has been registered, the head must come back to
    # within (threshold * this ratio) of the baseline before another nod can
    # be counted. Stops the dip AND the recovery of one nod both counting.
    nod_release_ratio: float = 0.5

    # Time constant (seconds) of the resting-pitch baseline. It only adapts
    # while the head is NOT mid-nod, so a nod can't drag the baseline with it.
    baseline_tau_seconds: float = 5.0

    # If the head stays deviated longer than this, treat it as a posture
    # change (helmet re-seated, rider leaning) and re-baseline instead of
    # staying "mid-nod" forever. Sustained eye closure is still caught by
    # the eye channel.
    nod_max_hold_seconds: float = 10.0

    # Rolling window (seconds) over which nod-event frequency is computed.
    motion_window_seconds: float = 15.0

    # Nod-event counts in the window that map to moderate / severe head score.
    nod_count_moderate: int = 2
    nod_count_severe: int = 4


@dataclass
class FusionConfig:
    # Composite drowsiness score (0-1) thresholds for escalating into
    # YELLOW / RED.
    #
    # Fusion formula: composite = max(eye_score, head_score)
    #                            + multi_signal_bonus * min(eye_score, head_score)
    # (capped at 1.0). This means one channel alone hitting "severe" (1.0)
    # is enough to reach RED on its own -- a fully-closed-eyes-for-15s
    # reading shouldn't need head-motion corroboration to be treated as
    # critical -- while agreement between BOTH channels pushes the score up
    # faster than either alone, matching the doc's "multi-signal decision"
    # rationale (reduce reliance on a single measurement) without requiring
    # both signals to fire before anything critical is ever raised.
    multi_signal_bonus: float = 0.3

    yellow_threshold: float = 0.35
    red_threshold: float = 0.70

    # Minimum time (seconds) a lower-severity state must persist before the
    # engine will step back down a level. Prevents flicker between levels
    # on borderline/noisy readings. Escalation is intentionally NOT delayed
    # this way -- the doc calls for prompt warnings, slow, deliberate
    # de-escalation.
    min_dwell_before_deescalate: float = 5.0


@dataclass
class AppConfig:
    eye: EyeConfig = field(default_factory=EyeConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)

    # Target loop / broadcast rate for the live dashboard feed.
    tick_hz: float = 5.0
