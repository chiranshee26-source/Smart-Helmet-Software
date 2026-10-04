"""
Tests for the evaluation pipeline: protocol recording (with a fake camera
and fake clock) and replay scoring (evaluate.py). No camera needed.

Run with: python -m pytest tests/ -v   (from the project root)
"""

import csv
import os
import random
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from config import AppConfig  # noqa: E402
from evaluate import (  # noqa: E402
    Sample, evaluate, evaluate_session, legacy_config, load_session, render_report,
)
from protocol import ALERT, DEFAULT_PROTOCOL, DROWSY, Phase, run_protocol  # noqa: E402

HZ = 15.0


# --- synthetic eye behaviour for each protocol phase -------------------------

def _ear_for(phase: str, t_in_phase: float, rng: random.Random):
    """Plausible EAR for a volunteer following each instruction."""
    def blink(period, length, open_ear):
        return 0.10 if (t_in_phase % period) < length else open_ear + rng.uniform(-0.015, 0.015)
    if phase == "alert_baseline" or phase.startswith("alert_recover"):
        return blink(4.0, 0.15, 0.33)
    if phase == "alert_frequent_blinks":
        return blink(2.0, 0.25, 0.33)          # 30 blinks/min
    if phase == "alert_squint":
        return blink(2.0, 0.25, 0.20)          # squint below the old 0.21 line
    if phase == "alert_glance_around":
        return None if (t_in_phase % 6.0) < 0.3 else blink(4.0, 0.15, 0.33)
    if phase == "drowsy_slow_blinks":
        return blink(3.0, 1.5, 0.30)           # 1.5 s closures
    if phase == "drowsy_microsleep":
        return 0.09
    raise ValueError(phase)


def _synthetic_session(path, seed=0, protocol=DEFAULT_PROTOCOL):
    """Write a session CSV exactly as record_session.py would."""
    clock = [1000.0]
    rng = random.Random(seed)
    state = {"phase": None, "phase_start": 0.0}

    class FakeTracker:
        def read_once(self):
            t = clock[0]
            ear = _ear_for(state["phase"].name, t - state["phase_start"], rng)
            return SimpleNamespace(timestamp=t, ear=ear, face_found=ear is not None,
                                   occlusion_reason=None)

    def on_sample(phase, left, r):
        return True

    def fake_sleep(s):
        clock[0] += s

    # Phase boundaries on the fake clock, so the fake tracker knows which
    # instruction the "volunteer" is following.
    boundaries, t = [], clock[0]
    for p in protocol:
        boundaries.append((t, t + p.seconds, p))
        t += p.seconds

    def now():
        for start, end, p in boundaries:
            if start <= clock[0] < end and state["phase"] is not p:
                state["phase"], state["phase_start"] = p, start
        return clock[0]

    return run_protocol(FakeTracker(), path, protocol, sample_hz=HZ,
                        clock=now, sleep=fake_sleep, on_sample=on_sample)


def _tmp_csv():
    fd, path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    return path


# --- protocol runner ---------------------------------------------------------

def test_protocol_writes_labelled_rows_for_every_phase():
    path = _tmp_csv()
    rows = _synthetic_session(path)
    with open(path, newline="") as fh:
        data = list(csv.DictReader(fh))
    assert rows == len(data)
    assert [p.name for p in DEFAULT_PROTOCOL] == list(dict.fromkeys(r["phase"] for r in data))
    for p in DEFAULT_PROTOCOL:
        n = sum(r["phase"] == p.name for r in data)
        assert abs(n - p.seconds * HZ) <= 2, (p.name, n)
        assert all(r["label"] == p.label for r in data if r["phase"] == p.name)


def test_protocol_abort_stops_early():
    path = _tmp_csv()
    clock = [0.0]
    tracker = SimpleNamespace(read_once=lambda: SimpleNamespace(
        timestamp=clock[0], ear=0.3, face_found=True, occlusion_reason=None))
    calls = []

    def on_sample(phase, left, r):
        calls.append(1)
        return len(calls) < 5

    def sleep(s):
        clock[0] += s

    rows = run_protocol(tracker, path, [Phase("x", ALERT, 60, "")], sample_hz=10,
                        clock=lambda: clock[0], sleep=sleep, on_sample=on_sample)
    assert rows == 5


# --- scoring -----------------------------------------------------------------

def test_current_engine_detects_all_drowsy_phases_without_false_alarms():
    path = _tmp_csv()
    _synthetic_session(path)
    s = evaluate([path], AppConfig(), "Current")
    assert s.drowsy_phases == 2 and s.detected == 2
    assert s.false_alarms == 0, [(p.phase, p.false_alarms) for p in s.phases]
    assert s.mean_time_to_yellow is not None and s.mean_time_to_yellow < 15


def test_legacy_engine_false_alarms_on_blinks_and_squint():
    # Same recording, old logic: the frequent-blink / squint phases that an
    # alert rider produces in wind should show up as false alarms.
    path = _tmp_csv()
    _synthetic_session(path)
    legacy = evaluate([path], legacy_config(), "Legacy")
    assert legacy.false_alarms >= 1
    blinks = [p for p in legacy.phases if p.phase == "alert_frequent_blinks"][0]
    squint = [p for p in legacy.phases if p.phase == "alert_squint"][0]
    assert blinks.false_alarms >= 1
    # The warning carries on through the squint (no new alarm to count, but
    # the rider spends the phase in warning).
    assert squint.warning_seconds >= 0.5 * squint.counted_seconds


def test_missed_detection_is_reported():
    # A "drowsy" phase where the volunteer actually kept eyes open.
    t, samples = 0.0, []
    for phase, label, secs in (("a", ALERT, 30), ("d", DROWSY, 30)):
        for _ in range(int(secs * HZ)):
            ear = 0.10 if (t % 4.0) < 0.15 else 0.33
            samples.append(Sample(t, phase, label, ear))
            t += 1 / HZ
    res = evaluate_session(samples, AppConfig())
    drowsy = [p for p in res if p.label == DROWSY][0]
    assert drowsy.detected is False and drowsy.time_to_yellow is None


def test_recovery_after_drowsy_phase_not_counted_as_false_alarm():
    # The alert legitimately holds after drowsiness ends; that tail must be
    # excluded from the following alert phase's false-alarm count.
    t, samples = 0.0, []
    for phase, label, secs, ear in (("a1", ALERT, 30, 0.33), ("d", DROWSY, 20, 0.09),
                                    ("a2", ALERT, 40, 0.33)):
        for _ in range(int(secs * HZ)):
            samples.append(Sample(t, phase, label, ear))
            t += 1 / HZ
    res = {p.phase: p for p in evaluate_session(samples, AppConfig())}
    assert res["d"].detected
    assert res["a2"].false_alarms == 0
    rec = AppConfig().eye.perclos_window_seconds + AppConfig().fusion.min_dwell_before_deescalate
    assert abs(res["a2"].counted_seconds - (40 - rec)) < 1.0


def test_csv_round_trip_and_report():
    path = _tmp_csv()
    _synthetic_session(path, seed=3)
    samples = load_session(path)
    assert any(s.ear is None for s in samples)            # glance-away frames kept as None
    report = render_report([evaluate([path], AppConfig(), "Current"),
                            evaluate([path], legacy_config(), "Legacy")], 1)
    assert "Drowsy phases detected" in report and "| Current | Legacy |" in report


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"OK: {test.__name__}")
    print(f"\n{len(tests)} tests passed.")
