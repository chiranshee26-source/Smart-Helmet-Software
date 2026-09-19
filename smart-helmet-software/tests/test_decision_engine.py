"""
Unit tests for the decision engine's escalation/de-escalation behaviour.
No camera/hardware needed -- pure synthetic (ear, head_pitch) sequences.

Run with: python -m pytest tests/ -v   (from the project root)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from config import AppConfig  # noqa: E402
from decision_engine import AlertLevel, DecisionEngine  # noqa: E402


def make_engine():
    return DecisionEngine(AppConfig())


def test_stays_green_on_normal_input():
    engine = make_engine()
    t = 0.0
    for _ in range(60):
        engine.update_eye(0.30, t)   # eyes clearly open
        engine.update_head(0.0, t)   # head steady
        state = engine.tick(t)
        t += 0.2
    assert state.alert_level == AlertLevel.GREEN


def test_escalates_to_red_on_sustained_eye_closure():
    engine = make_engine()
    t = 0.0
    state = None
    # Feed 20 seconds of near-closed eyes (well past the 15s PERCLOS window).
    for _ in range(100):
        engine.update_eye(0.10, t)
        engine.update_head(0.0, t)
        state = engine.tick(t)
        t += 0.2
    assert state.alert_level == AlertLevel.RED, f"expected RED, got {state.alert_level}"


def test_escalates_on_repeated_head_nods_alone():
    engine = make_engine()
    t = 0.0
    state = None
    # Eyes stay open the whole time; only head-nod events drive the score.
    for i in range(150):
        engine.update_eye(0.30, t)
        # Inject a nod roughly every second.
        pitch = 30.0 if i % 5 == 0 else 0.0
        engine.update_head(pitch, t)
        state = engine.tick(t)
        t += 0.2
    assert state.alert_level >= AlertLevel.YELLOW, (
        f"expected at least YELLOW from head motion alone, got {state.alert_level}"
    )


def test_escalation_is_immediate_not_debounced():
    engine = make_engine()
    t = 0.0
    # Warm up in GREEN.
    for _ in range(20):
        engine.update_eye(0.30, t)
        engine.update_head(0.0, t)
        engine.tick(t)
        t += 0.2
    # Now slam in enough closed-eye samples to fill the whole PERCLOS window
    # at once (simulates a sudden, sustained eye closure).
    window = AppConfig().eye.perclos_window_seconds
    steps = int(window / 0.2) + 5
    state = None
    for _ in range(steps):
        engine.update_eye(0.05, t)
        engine.update_head(0.0, t)
        state = engine.tick(t)
        t += 0.2
    assert state.alert_level == AlertLevel.RED


def test_deescalation_requires_dwell_time():
    engine = make_engine()
    t = 0.0
    # Drive to RED.
    for _ in range(100):
        engine.update_eye(0.10, t)
        engine.update_head(0.0, t)
        engine.tick(t)
        t += 0.2
    assert engine.tick(t).alert_level == AlertLevel.RED

    # Eyes open again -- PERCLOS will fall as the window slides, but the
    # engine should not instantly snap back to GREEN.
    immediate_state = None
    for _ in range(3):
        engine.update_eye(0.30, t)
        engine.update_head(0.0, t)
        immediate_state = engine.tick(t)
        t += 0.2
    assert immediate_state.alert_level == AlertLevel.RED, (
        "should not de-escalate within a couple of ticks"
    )

    # Keep feeding good data past the window + dwell time -- should recover.
    state = None
    for _ in range(200):
        engine.update_eye(0.30, t)
        engine.update_head(0.0, t)
        state = engine.tick(t)
        t += 0.2
    assert state.alert_level == AlertLevel.GREEN


def test_manual_nod_injection_contributes_to_score():
    engine = make_engine()
    t = 0.0
    for _ in range(10):
        engine.update_eye(0.30, t)
        engine.update_head(0.0, t)
        engine.tick(t)
        t += 0.2

    for _ in range(6):
        engine.inject_nod_event(t)
        t += 0.5
    state = engine.tick(t)
    assert state.nod_count >= 4
    assert state.head_score > 0.0


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"OK: {test.__name__}")
    print(f"\n{len(tests)} tests passed.")
