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
    # occlusion.py.) This is a coarse heuristic, not real occlusion
    # detection -- it is not guaranteed to catch every case (documented as
    # a known limitation either way).
    #
    # Absolute values below calibrated via calibrate_occlusion.py against this
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
    # ^ These three absolute values are now only a FALLBACK, used when the
    #   cheek reference patch can't be sampled. See below.

    # --- Lighting-invariant check (primary, added 2026-10-04) ---
    # The absolute 80.1 dark threshold above flagged bare eyes as
    # "sunglasses" in a dimmer room (bare-eye patch fell below 80), which
    # discarded every eye reading and locked the alert on RED. The primary
    # check now compares the eye patch with a bare-skin patch on the
    # cheekbone below each eye: ratio = eye_brightness / cheek_brightness.
    # Dim light darkens both, so the ratio holds; a tinted lens darkens only
    # the eyes, so the ratio drops. See occlusion.py.
    #
    # Calibrated with calibrate_occlusion.py on 2026-10-04 (laptop webcam,
    # night, room light + screen):
    #   eyes open, normal light : ratio 0.44 (range 0.32-0.49), cheek 192
    #   eyes closed             : ratio 0.51 (range 0.48-0.54)
    #   eyes open, dim light    : ratio 0.64 (range 0.62-0.65), eye 40, cheek 62
    #   sunglasses on           : ratio 0.83 (range 0.44-1.25), cheek 112
    # Findings:
    #   - Dim light: bare-eye brightness fell 85 -> 40 (the old fixed 80.1
    #     threshold would have flagged it), ratio stayed in 0.44-0.64. The
    #     ratio fixes the stuck-on-RED-in-dim-light bug.
    #   - Sunglasses read HIGHER than bare eyes and overlap them (webcam
    #     auto-exposure brightens the frame; the frames likely cover part of
    #     the cheek patch). On this camera, these sunglasses CANNOT be
    #     detected by brightness. Documented as a known limitation.
    # So thresholds are set to never flag bare eyes (lowest bare 0.32,
    # highest 0.65) with margin; they only catch extreme cases (opaque
    # patch, strong glare).
    occlusion_dark_ratio_threshold: float = 0.25    # eye/cheek below this = tinted lens
    occlusion_bright_ratio_threshold: float = 1.50  # eye/cheek above this = mirrored lens / glare
    occlusion_min_texture_cv: float = 0.05          # eye std/mean below this = featureless flat patch
    # Cheek brightness below this (0-255) = face too dark for the camera to
    # see at all. EAR is distrusted with reason "too_dark", not "sunglasses".
    occlusion_too_dark_floor: float = 25.0
    # Where the cheek patch sits: this many eye-widths below the eye centre
    # (far enough down to clear typical sunglasses lenses).
    cheek_offset_eye_widths: float = 1.2

    # --- No-blink check (see blink_monitor.py) ---
    # Behind sunglasses the landmark model guesses an EAR that tends to stay
    # flat. Real eyes blink every few seconds. If no blink (EAR dip) is seen
    # for this long, the EAR is treated as unverified ("can't see the eyes").
    # 30 s is deliberately generous; tune after testing with real riders.
    enable_no_blink_check: bool = True
    no_blink_seconds: float = 30.0
    # A "blink" = EAR below this fraction of the rider's recent open-eye EAR
    # (or below ear_closed_threshold).
    blink_dip_ratio: float = 0.75
    blink_baseline_tau_seconds: float = 3.0

    # --- Camera exposure lock (experimental) ---
    # Webcam auto-exposure brightens the image when sunglasses go on, which
    # hides the darker lenses from the brightness-ratio check. When True,
    # the camera's exposure is frozen at its settled value right after it
    # opens. Many webcams/drivers ignore this request; the startup log says
    # whether it appeared to work. After enabling, re-run
    # calibrate_occlusion.py to see whether sunglasses now separate.
    lock_camera_exposure: bool = False


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
