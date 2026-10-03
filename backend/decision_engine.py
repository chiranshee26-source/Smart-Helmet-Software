"""
Multi-signal drowsiness decision engine.

Implements the "Multi-Signal Decision" + "Severity / Persistence" +
"Progressive Alert System" ideas from the IDP document:

  - Eye/face indicator (EAR) and head-motion indicator (IMU pitch) are each
    turned into a rolling-window "severity" score, not judged frame-by-frame.
  - The two scores are fused into one composite drowsiness score.
  - The composite score is mapped to a 3-level alert (GREEN/YELLOW/RED) with
    a state machine that escalates immediately but de-escalates only after
    sustained improvement, so the alert doesn't flicker on noisy input.

This module has NO dependency on OpenCV, mediapipe, or any specific sensor.
It only consumes numeric samples (ear, head_pitch_deg) with timestamps, so
the exact same engine can run against a laptop webcam + simulated IMU today
and against a real ESP32-CAM + MPU6050 stream later without any changes.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Deque, List, Optional, Tuple

from config import AppConfig


class AlertLevel(IntEnum):
    GREEN = 0
    YELLOW = 1
    RED = 2

    @property
    def label(self) -> str:
        return {
            AlertLevel.GREEN: "GREEN",
            AlertLevel.YELLOW: "YELLOW",
            AlertLevel.RED: "RED",
        }[self]

    @property
    def action(self) -> str:
        return {
            AlertLevel.GREEN: "No warning",
            AlertLevel.YELLOW: "Vibration + display warning",
            AlertLevel.RED: "Buzzer + strong vibration (optional GPS/IoT alert)",
        }[self]


@dataclass
class EngineState:
    timestamp: float
    ear: Optional[float]
    perclos: float
    eye_score: float
    head_pitch_deg: Optional[float]
    nod_count: int
    head_score: float
    composite_score: float
    alert_level: AlertLevel
    action: str
    just_changed: bool


@dataclass
class _EventLogEntry:
    timestamp: float
    message: str


class DecisionEngine:
    def __init__(self, config: Optional[AppConfig] = None):
        self.cfg = config or AppConfig()

        self._ear_samples: Deque[Tuple[float, Optional[float]]] = deque()  # (t, ear-or-None)
        self._nod_events: Deque[float] = deque()                     # timestamps
        self._last_head_pitch: Optional[float] = None
        self._last_head_t: Optional[float] = None
        self._head_baseline: Optional[float] = None
        self._in_nod: bool = False
        self._nod_started_at: Optional[float] = None

        self._level: AlertLevel = AlertLevel.GREEN
        self._pending_lower_level: Optional[AlertLevel] = None
        self._pending_lower_since: Optional[float] = None

        self.event_log: List[_EventLogEntry] = []

    # ------------------------------------------------------------------ #
    # Ingest
    # ------------------------------------------------------------------ #
    def update_eye(self, ear: Optional[float], timestamp: float) -> None:
        """ear=None means the face/eyes were not detected this frame at all
        (occlusion, head turned/tilted away, poor lighting, etc). This is
        deliberately NOT treated as "no evidence of drowsiness" -- a camera
        that can no longer see the rider's eyes at all is itself a warning
        condition (the project doc calls out occlusion as a known camera
        limitation), so sustained no-face frames count the same as sustained
        closed-eye frames in the PERCLOS calculation below."""
        self._ear_samples.append((timestamp, ear))
        cutoff = timestamp - self.cfg.eye.perclos_window_seconds
        while self._ear_samples and self._ear_samples[0][0] < cutoff:
            self._ear_samples.popleft()

    def update_head(self, pitch_deg: float, timestamp: float) -> None:
        """Edge-triggered nod detection against a slowly-adapting baseline.

        One nod = one event: the event fires when |pitch - baseline| first
        crosses the threshold, and no further event can fire until the head
        returns close to baseline (hysteresis). Comparing against a baseline
        rather than the previous sample makes detection independent of the
        IMU sample rate and catches slow droops, not just sharp jerks."""
        hc = self.cfg.head
        self._last_head_pitch = pitch_deg

        if self._head_baseline is None:
            self._head_baseline = pitch_deg
            self._last_head_t = timestamp
            return

        dt = max(0.0, timestamp - (self._last_head_t or timestamp))
        self._last_head_t = timestamp
        deviation = abs(pitch_deg - self._head_baseline)

        if not self._in_nod:
            if deviation >= hc.nod_pitch_delta_threshold:
                self._in_nod = True
                self._nod_started_at = timestamp
                self._nod_events.append(timestamp)
            else:
                alpha = 1.0 - math.exp(-dt / hc.baseline_tau_seconds) if dt > 0 else 0.0
                self._head_baseline += alpha * (pitch_deg - self._head_baseline)
        else:
            released = deviation <= hc.nod_pitch_delta_threshold * hc.nod_release_ratio
            held_too_long = (
                timestamp - (self._nod_started_at or timestamp) >= hc.nod_max_hold_seconds
            )
            if released or held_too_long:
                self._in_nod = False
                self._nod_started_at = None
                if held_too_long:
                    self._head_baseline = pitch_deg  # posture change: re-baseline

        cutoff = timestamp - hc.motion_window_seconds
        while self._nod_events and self._nod_events[0] < cutoff:
            self._nod_events.popleft()

    def inject_nod_event(self, timestamp: float) -> None:
        """Force-register a nod event now (used by the simulator/demo controls)."""
        self._nod_events.append(timestamp)
        cutoff = timestamp - self.cfg.head.motion_window_seconds
        while self._nod_events and self._nod_events[0] < cutoff:
            self._nod_events.popleft()

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #
    def _perclos(self) -> float:
        if not self._ear_samples:
            return 0.0
        closed = sum(
            1 for _, ear in self._ear_samples
            if ear is None or ear < self.cfg.eye.ear_closed_threshold
        )
        raw = closed / len(self._ear_samples)

        # Down-weight the ratio while the rolling window is still filling
        # up (e.g. right after startup, or shortly after any gap). Without
        # this, a couple of bad samples out of a nearly-empty window looks
        # like a huge fraction and can trigger a false alarm; a mature,
        # full 15s window gets full weight as intended.
        span = self._ear_samples[-1][0] - self._ear_samples[0][0]
        fill_weight = min(1.0, span / self.cfg.eye.perclos_window_seconds)
        return raw * fill_weight

    def _eye_score(self, perclos: float) -> float:
        lo, hi = self.cfg.eye.perclos_moderate, self.cfg.eye.perclos_severe
        if perclos <= lo:
            return 0.0 if perclos <= 0 else 0.5 * (perclos / lo)
        if perclos >= hi:
            return 1.0
        return 0.5 + 0.5 * (perclos - lo) / (hi - lo)

    def _head_score(self, nod_count: int) -> float:
        lo, hi = self.cfg.head.nod_count_moderate, self.cfg.head.nod_count_severe
        if nod_count <= 0:
            return 0.0
        if nod_count <= lo:
            return 0.5 * (nod_count / lo)
        if nod_count >= hi:
            return 1.0
        return 0.5 + 0.5 * (nod_count - lo) / (hi - lo)

    def _target_level(self, composite: float) -> AlertLevel:
        if composite >= self.cfg.fusion.red_threshold:
            return AlertLevel.RED
        if composite >= self.cfg.fusion.yellow_threshold:
            return AlertLevel.YELLOW
        return AlertLevel.GREEN

    # ------------------------------------------------------------------ #
    # State machine
    # ------------------------------------------------------------------ #
    def _step_level(self, target: AlertLevel, timestamp: float) -> bool:
        """Update self._level given the instantaneous target level.
        Returns True if the level just changed."""
        if target > self._level:
            # Escalate immediately, no debounce -- safety-critical direction.
            changed = target != self._level
            self._level = target
            self._pending_lower_level = None
            self._pending_lower_since = None
            return changed

        if target < self._level:
            if self._pending_lower_level != target:
                self._pending_lower_level = target
                self._pending_lower_since = timestamp
                return False
            elapsed = timestamp - (self._pending_lower_since or timestamp)
            if elapsed >= self.cfg.fusion.min_dwell_before_deescalate:
                self._level = target
                self._pending_lower_level = None
                self._pending_lower_since = None
                return True
            return False

        # target == current level
        self._pending_lower_level = None
        self._pending_lower_since = None
        return False

    # ------------------------------------------------------------------ #
    # Public tick
    # ------------------------------------------------------------------ #
    def tick(self, timestamp: float) -> EngineState:
        perclos = self._perclos()
        eye_score = self._eye_score(perclos)
        nod_count = len(self._nod_events)
        head_score = self._head_score(nod_count)

        composite = min(
            1.0,
            max(eye_score, head_score)
            + self.cfg.fusion.multi_signal_bonus * min(eye_score, head_score),
        )
        target = self._target_level(composite)
        changed = self._step_level(target, timestamp)

        if changed:
            self.event_log.append(
                _EventLogEntry(
                    timestamp,
                    f"Alert level -> {self._level.label} "
                    f"(score={composite:.2f}, perclos={perclos:.2f}, nods={nod_count})",
                )
            )
            self.event_log = self.event_log[-100:]

        last_ear = self._ear_samples[-1][1] if self._ear_samples else None

        return EngineState(
            timestamp=timestamp,
            ear=last_ear,
            perclos=perclos,
            eye_score=eye_score,
            head_pitch_deg=self._last_head_pitch,
            nod_count=nod_count,
            head_score=head_score,
            composite_score=composite,
            alert_level=self._level,
            action=self._level.action,
            just_changed=changed,
        )

    def recent_events(self, limit: int = 20) -> List[Tuple[float, str]]:
        return [(e.timestamp, e.message) for e in self.event_log[-limit:]]
