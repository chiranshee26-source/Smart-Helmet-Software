"""
Record one labelled evaluation session from the webcam.

The volunteer follows ~5 minutes of timed on-screen instructions (see
protocol.py). Each phase is pre-labelled ALERT or DROWSY, so the output CSV
is ready for evaluate.py. Only numbers are saved -- no images or video.

Run from the backend/ folder, with the dashboard (uvicorn) STOPPED so the
camera is free:

    python record_session.py --participant P01

Output: ../recordings/P01_<date>_<time>.csv

Keys in the preview window: q = abort.
SAFETY: seated and controlled only. Drowsiness is acted, never real riding.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2

from config import AppConfig
from eye_detection import WebcamEyeTracker
from protocol import ALERT, DEFAULT_PROTOCOL, protocol_seconds, run_protocol

RECORDINGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "recordings")


def _draw(frame, phase, seconds_left, result):
    h, w = frame.shape[:2]
    colour = (80, 200, 80) if phase.label == ALERT else (60, 160, 255)
    cv2.rectangle(frame, (0, 0), (w, 92), (20, 20, 20), -1)
    cv2.putText(frame, f"{phase.label.upper()}: {phase.name}   {seconds_left:4.0f}s",
                (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2)
    # Wrap the instruction onto two lines if needed.
    words, line, lines = phase.instruction.split(), "", []
    for word in words:
        if len(line) + len(word) > 70:
            lines.append(line)
            line = ""
        line += word + " "
    lines.append(line)
    for i, text in enumerate(lines[:2]):
        cv2.putText(frame, text, (12, 58 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1)
    ear = "--" if result.ear is None else f"{result.ear:.2f}"
    note = result.occlusion_reason or ("" if result.face_found else "no face")
    cv2.putText(frame, f"EAR {ear} {note}", (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--participant", required=True, help="Short anonymous ID, e.g. P01 (don't use real names)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--no-preview", action="store_true", help="Don't open a preview window")
    args = ap.parse_args()

    cfg = AppConfig()
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.abspath(os.path.join(RECORDINGS_DIR, f"{args.participant}_{stamp}.csv"))

    print("=" * 70)
    print(f"Evaluation recording -- participant {args.participant}")
    print("=" * 70)
    print(f"About {protocol_seconds(DEFAULT_PROTOCOL) / 60:.0f} minutes. Phases:")
    for p in DEFAULT_PROTOCOL:
        print(f"  [{p.label.upper():6}] {p.seconds:3.0f}s  {p.instruction}")
    print("\nSit as you normally would, face fully in view, normal room lighting.")
    print("Only numbers are recorded (no video). Press q in the preview to abort.")
    input("\nPress Enter to start...")

    tracker = WebcamEyeTracker(camera_index=args.camera, eye_config=cfg.eye)
    tracker.open()
    last_phase = [None]

    def on_sample(phase, seconds_left, result):
        if phase is not last_phase[0]:
            last_phase[0] = phase
            print(f"\n>>> {phase.label.upper()}: {phase.instruction}\a")
        if args.no_preview or result.frame is None:
            return True
        frame = result.frame.copy()
        _draw(frame, phase, seconds_left, result)
        cv2.imshow("Smart helmet - evaluation recording", frame)
        return (cv2.waitKey(1) & 0xFF) != ord("q")

    try:
        rows = run_protocol(tracker, out_path, phases=DEFAULT_PROTOCOL,
                            sample_hz=cfg.eye_sample_hz, on_sample=on_sample)
    finally:
        tracker.release()
        cv2.destroyAllWindows()

    complete = rows > 0 and last_phase[0] is DEFAULT_PROTOCOL[-1]
    print(f"\nSaved {rows} samples to {out_path}")
    if not complete:
        print("Session was aborted early -- you can still evaluate it, but it's incomplete.")
    print(f"Next: python evaluate.py {os.path.relpath(out_path)}")


if __name__ == "__main__":
    sys.exit(main())
