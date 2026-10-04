"""
Guided test protocol for evaluating the eye channel with real people.

A volunteer sits in front of the webcam and follows timed on-screen
instructions. Each phase is labelled ALERT or DROWSY ahead of time, so the
recording comes out already labelled -- no manual annotation needed.

SAFETY: this is a seated, controlled protocol. Drowsiness is *acted*
(slow blinks, eyes held closed). Never evaluate the system by riding while
tired. (This is also stated in the project's IDP document.)

Only numbers are recorded (timestamp, EAR, face found, distrust reason) --
no images or video -- so volunteers' faces are never stored.

This module has no OpenCV/mediapipe dependency: the runner takes any object
with a `read_once()` method, so it is unit-tested with a fake tracker.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

ALERT = "alert"
DROWSY = "drowsy"

CSV_FIELDS = ["t", "phase", "label", "ear", "face_found", "occlusion_reason"]


@dataclass(frozen=True)
class Phase:
    name: str
    label: str           # ALERT or DROWSY: what the system SHOULD conclude
    seconds: float
    instruction: str


# ~5 minutes total. Alert phases deliberately include the things that
# caused false alarms during development (frequent blinking, squinting,
# glancing away), so the false-alarm number is honest.
DEFAULT_PROTOCOL: List[Phase] = [
    Phase("alert_baseline", ALERT, 60,
          "Look at the screen normally. Blink as you usually would."),
    Phase("alert_frequent_blinks", ALERT, 30,
          "Stay alert but blink quickly and often, as if wind is hitting your eyes."),
    Phase("alert_squint", ALERT, 30,
          "Squint slightly, as if riding into wind or sun. Keep blinking normally."),
    Phase("drowsy_slow_blinks", DROWSY, 45,
          "Act drowsy: slow, heavy blinks, keeping your eyes closed 1-2 seconds each time."),
    Phase("alert_recover_1", ALERT, 30,
          "Wake up: eyes open, look at the screen, blink normally."),
    Phase("drowsy_microsleep", DROWSY, 20,
          "Close your eyes and keep them closed until told to open them."),
    Phase("alert_recover_2", ALERT, 30,
          "Eyes open, look at the screen, blink normally."),
    Phase("alert_glance_around", ALERT, 45,
          "Stay alert, but every few seconds glance briefly left or right, like checking mirrors."),
]


def protocol_seconds(phases: List[Phase]) -> float:
    return sum(p.seconds for p in phases)


def run_protocol(
    tracker,
    out_path: str,
    phases: List[Phase] = DEFAULT_PROTOCOL,
    sample_hz: float = 15.0,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
    on_sample: Optional[Callable[[Phase, float, object], bool]] = None,
) -> int:
    """Run the timed phases, sampling tracker.read_once() at `sample_hz`,
    and write one CSV row per sample. Returns the number of rows written.

    on_sample(phase, seconds_left_in_phase, result) is called after each
    sample (e.g. to draw a preview window); returning False aborts early.
    """
    period = 1.0 / sample_hz
    rows = 0
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for phase in phases:
            phase_end = clock() + phase.seconds
            while True:
                started = clock()
                if started >= phase_end:
                    break
                r = tracker.read_once()
                writer.writerow({
                    "t": f"{r.timestamp:.3f}",
                    "phase": phase.name,
                    "label": phase.label,
                    "ear": "" if r.ear is None else f"{r.ear:.4f}",
                    "face_found": int(bool(r.face_found)),
                    "occlusion_reason": r.occlusion_reason or "",
                })
                rows += 1
                if on_sample is not None and on_sample(phase, phase_end - clock(), r) is False:
                    return rows
                sleep(max(0.0, period - (clock() - started)))
    return rows
