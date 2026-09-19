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


@dataclass
class HeadConfig:
    # A single head-motion sample is flagged as a "nod event" when the pitch
    # change between consecutive samples exceeds this (degrees).
    nod_pitch_delta_threshold: float = 18.0

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
