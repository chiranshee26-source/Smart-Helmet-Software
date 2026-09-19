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

        self._ear_samples: Deque[Tuple[float, float]] = deque()      # (t, ear)
        self._nod_events: Deque[float] = deque()                     # timestamps
        self._last_head_pitch: Optional[float] = None

        self._level: AlertLevel = AlertLevel.GREEN
        self._pending_lower_level: Optional[AlertLevel] = None
        self._pending_lower_since: Optional[float] = None

        self.event_log: List[_EventLogEntry] = []

    # ------------------------------------------------------------------ #
    # Ingest
    # ------------------------------------------------------------------ #
    def update_eye(self, ear: float, timestamp: float) -> None:
        self._ear_samples.append((timestamp, ear))
        cutoff = timestamp - self.cfg.eye.perclos_window_seconds
        while self._ear_samples and self._ear_samples[0][0] < cutoff:
            self._ear_samples.popleft()

    def update_head(self, pitch_deg: float, timestamp: float) -> None:
        if self._last_head_pitch is not None:
            delta = abs(pitch_deg - self._last_head_pitch)
            if delta >= self.cfg.head.nod_pitch_delta_threshold:
                self._nod_events.append(timestamp)
        self._last_head_pitch = pitch_deg

        cutoff = timestamp - self.cfg.head.motion_window_seconds
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
            1 for _, ear in self._ear_samples if ear < self.cfg.eye.ear_closed_threshold
        )
        return closed / len(self._ear_samples)

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
