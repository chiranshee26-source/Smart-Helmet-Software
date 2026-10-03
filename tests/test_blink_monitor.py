"""
Unit tests for the no-blink check (backend/blink_monitor.py).
Synthetic EAR sequences only -- no camera needed.

Run with: python -m pytest tests/ -v   (from the project root)
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from blink_monitor import BlinkMonitor  # noqa: E402
from config import EyeConfig  # noqa: E402

DT = 0.2  # 5 Hz, same as the dashboard loop


def _run(monitor, ears, t0=0.0):
    t, flags = t0, []
    for ear in ears:
        flags.append(monitor.update(ear, t))
        t += DT
    return flags, t


def _real_eyes(seconds, blink_every=4.0, seed=0):
    """Open-eye EAR ~0.33 with noise, and a 1-sample blink every few seconds."""
    rng = random.Random(seed)
    out, n = [], int(seconds / DT)
    period = int(blink_every / DT)
    for i in range(n):
        out.append(0.12 if i % period == period - 1 else 0.33 + rng.uniform(-0.02, 0.02))
    return out


def _flat_guess(seconds, seed=1):
    """What a guessed EAR behind sunglasses tends to look like: steady."""
    rng = random.Random(seed)
    return [0.31 + rng.uniform(-0.015, 0.015) for _ in range(int(seconds / DT))]


def test_real_blinking_eyes_never_flagged():
    flags, _ = _run(BlinkMonitor(), _real_eyes(120))
    assert not any(flags)


def test_flat_ear_flagged_after_timeout():
    cfg = EyeConfig()
    flags, _ = _run(BlinkMonitor(cfg), _flat_guess(cfg.no_blink_seconds + 5))
    first = flags.index(True) * DT
    assert abs(first - cfg.no_blink_seconds) <= DT + 1e-9


def test_not_flagged_before_timeout():
    cfg = EyeConfig()
    flags, _ = _run(BlinkMonitor(cfg), _flat_guess(cfg.no_blink_seconds - 2))
    assert not any(flags)


def test_a_blink_clears_the_flag():
    m = BlinkMonitor()
    flags, t = _run(m, _flat_guess(40))
    assert flags[-1] is True
    assert m.update(0.10, t) is False          # glasses off, rider blinks
    flags2, _ = _run(m, _real_eyes(20), t0=t + DT)
    assert not any(flags2)


def test_long_eye_closure_does_not_trigger_no_blink():
    # Sustained closure is caught by PERCLOS as eyes-closed; it must not be
    # reclassified as "can't see the eyes".
    m = BlinkMonitor()
    _run(m, _real_eyes(10))
    flags, _ = _run(m, [0.10] * int(60 / DT), t0=10.0)
    assert not any(flags)


def test_face_lost_restarts_the_clock():
    cfg = EyeConfig()
    m = BlinkMonitor(cfg)
    _, t = _run(m, _flat_guess(cfg.no_blink_seconds - 5))
    _, t = _run(m, [None] * 10, t0=t)          # face lost briefly
    flags, _ = _run(m, _flat_guess(cfg.no_blink_seconds - 5), t0=t)
    assert not any(flags)


def test_slowly_drifting_open_eyes_not_counted_as_blinks():
    # Open-eye EAR drifting 0.36 -> 0.30 over 30 s (posture, lighting) is
    # not a blink; the baseline adapts. With no real blinks it should flag.
    m = BlinkMonitor()
    n = int(40 / DT)
    flags, _ = _run(m, [0.36 - 0.06 * i / n for i in range(n)])
    assert flags[-1] is True


def test_check_can_be_disabled():
    cfg = EyeConfig(enable_no_blink_check=False)
    flags, _ = _run(BlinkMonitor(cfg), _flat_guess(60))
    assert not any(flags)


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"OK: {test.__name__}")
    print(f"\n{len(tests)} tests passed.")
