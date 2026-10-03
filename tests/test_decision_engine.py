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


def _feed_pitch(engine, pitches, dt=0.2, t0=0.0):
    t = t0
    for p in pitches:
        engine.update_eye(0.30, t)
        engine.update_head(p, t)
        engine.tick(t)
        t += dt
    return t


def test_single_nod_counts_once_not_twice():
    # A one-sample spike used to count twice (the dip AND the recovery both
    # exceeded the sample-to-sample delta threshold).
    engine = make_engine()
    _feed_pitch(engine, [0.0] * 10 + [-25.0] + [0.0] * 10)
    assert engine.tick(5.0).nod_count == 1


def test_slow_droop_is_detected():
    # A real drowsy nod is a slow droop: 30 deg over ~1 s is only ~6 deg per
    # sample at 5 Hz, which a sample-to-sample check never catches.
    engine = make_engine()
    # Recovery is gradual too, so a "snap back up" can't be what triggers it.
    droop = (
        [0.0] * 10
        + [-6.0 * k for k in range(1, 6)]
        + [-30.0] * 3
        + [-6.0 * k for k in range(4, -1, -1)]
        + [0.0] * 5
    )
    _feed_pitch(engine, droop)
    assert engine.tick(5.0).nod_count == 1


def test_nod_count_independent_of_sample_rate():
    # Same physical nod (dip to -30 deg over 1 s, recover over 1 s), sampled
    # at 5 Hz (simulator) and 100 Hz (realistic MPU6050 rate).
    def nod_shape(t):
        if t < 2.0:
            return 0.0
        if t < 3.0:
            return -30.0 * (t - 2.0)
        if t < 4.0:
            return -30.0 * (4.0 - t)
        return 0.0

    counts = []
    for hz in (5, 100):
        engine = make_engine()
        dt = 1.0 / hz
        _feed_pitch(engine, [nod_shape(k * dt) for k in range(int(6 * hz))], dt=dt)
        counts.append(engine.tick(6.0).nod_count)
    assert counts == [1, 1], counts


def test_baseline_tracks_slow_posture_drift():
    # Rider slowly settles into a 20-deg-forward riding posture over 60 s.
    # That is a posture change, not a nod.
    engine = make_engine()
    _feed_pitch(engine, [-20.0 * min(1.0, k / 300) for k in range(400)])
    assert engine.tick(80.0).nod_count == 0


def test_idle_simulator_does_not_false_alarm():
    # Eyes open + the simulator at the rate main.py uses: the idle dashboard
    # must not raise alerts on its own.
    from imu_simulator import ImuSimulator

    for seed in range(3):
        engine = make_engine()
        sim = ImuSimulator(random_nod_probability_per_tick=0.002, seed=seed)
        t = 0.0
        for _ in range(5 * 60 * 5):  # 5 minutes at 5 Hz
            engine.update_eye(0.32, t)
            engine.update_head(sim.tick(t).pitch_deg, t)
            state = engine.tick(t)
            assert state.alert_level == AlertLevel.GREEN, f"seed {seed} t={t:.1f}"
            t += 0.2


def test_demo_clicks_escalate_progressively():
    # Dashboard "Inject head-nod" button: 1 click stays GREEN, 2 -> YELLOW,
    # 3 -> RED (previously 1 click jumped straight to YELLOW).
    from imu_simulator import ImuSimulator

    expected = {1: AlertLevel.GREEN, 2: AlertLevel.YELLOW, 3: AlertLevel.RED}
    for clicks, want in expected.items():
        engine = make_engine()
        sim = ImuSimulator(seed=1)
        t, peak = 0.0, AlertLevel.GREEN
        click_ticks = {10 + 8 * k for k in range(clicks)}
        for i in range(60):
            if i in click_ticks:
                sim.trigger_nod()
            engine.update_eye(0.32, t)
            engine.update_head(sim.tick(t).pitch_deg, t)
            peak = max(peak, engine.tick(t).alert_level)
            t += 0.2
        assert peak == want, f"{clicks} clicks -> {peak.name}, expected {want.name}"


# --- Blink- and squint-aware PERCLOS ---------------------------------------

def _eye_stream(engine, seconds, ear_at, hz=15, t0=0.0):
    """Feed EAR samples from ear_at(t) at `hz`; return peak level and last state."""
    dt, t = 1.0 / hz, t0
    peak, state = AlertLevel.GREEN, None
    while t < t0 + seconds:
        engine.update_eye(ear_at(t), t)
        engine.update_head(0.0, t)
        state = engine.tick(t)
        peak = max(peak, state.alert_level)
        t += dt
    return peak, state


def _blinking(open_ear, blinks_per_min, blink_s, closed_ear=0.10):
    period = 60.0 / blinks_per_min
    return lambda t: closed_ear if (t % period) < blink_s else open_ear


def test_frequent_normal_blinks_in_wind_stay_green():
    # Wind / irritated eyes: 30 blinks a minute, 250 ms each. Previously
    # this reached YELLOW (and RED when squinting); an alert rider blinking
    # a lot is not drowsy.
    engine = make_engine()
    peak, state = _eye_stream(engine, 90, _blinking(0.33, 30, 0.25))
    assert peak == AlertLevel.GREEN, f"peak {peak.name}, perclos {state.perclos:.2f}"
    assert state.perclos == 0.0


def test_squinting_in_wind_stays_green():
    # 20 s of normal open eyes to learn the baseline, then a sustained squint
    # at EAR 0.20 (below the old fixed 0.21 threshold) with frequent blinks.
    engine = make_engine()
    _eye_stream(engine, 20, _blinking(0.33, 15, 0.15))
    peak, state = _eye_stream(engine, 60, _blinking(0.20, 30, 0.25), t0=20)
    assert peak == AlertLevel.GREEN, f"peak {peak.name}, thr {state.ear_closed_threshold:.3f}"


def test_slow_drowsy_blinks_escalate():
    # Drowsy pattern: long 0.9 s closures every 3 s (30% of the time closed).
    engine = make_engine()
    _eye_stream(engine, 20, _blinking(0.33, 15, 0.15))
    peak, _ = _eye_stream(engine, 40, _blinking(0.33, 20, 0.9), t0=20)
    assert peak >= AlertLevel.YELLOW


def test_sustained_closure_still_reaches_red_at_15hz():
    engine = make_engine()
    _eye_stream(engine, 20, _blinking(0.33, 15, 0.15))
    peak, _ = _eye_stream(engine, 20, lambda t: 0.10, t0=20)
    assert peak == AlertLevel.RED


def test_eyes_closed_from_startup_cannot_become_the_baseline():
    # Safety guard: if the rider's eyes are closed from the start, the
    # "personal baseline" must not learn closed as normal.
    engine = make_engine()
    peak, state = _eye_stream(engine, 30, lambda t: 0.10)
    assert peak == AlertLevel.RED
    assert state.ear_closed_threshold == AppConfig().eye.ear_closed_threshold


def test_personal_threshold_never_stricter_than_fixed():
    engine = make_engine()
    _, state = _eye_stream(engine, 30, _blinking(0.45, 15, 0.15))  # very wide eyes
    assert state.ear_closed_threshold <= AppConfig().eye.ear_closed_threshold


def test_brief_face_dropouts_do_not_count():
    # 200 ms "no face" dropout every 2 s (tracking glitches) is not closure.
    engine = make_engine()
    peak, state = _eye_stream(engine, 60, lambda t: None if (t % 2.0) < 0.2 else 0.33)
    assert peak == AlertLevel.GREEN and state.perclos == 0.0


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"OK: {test.__name__}")
    print(f"\n{len(tests)} tests passed.")
