"""
Interactive EAR calibration for your own face, camera and lighting.

The default ear_closed_threshold (0.21) in config.py is a reasonable
generic starting point, but EAR values shift with your eye shape, camera
angle, resolution and lighting. This script has you hold your eyes open,
then closed, measures your actual EAR in both states, and tells you
exactly what to put in config.py.

Run from the backend/ folder:
    python calibrate.py

Requires a working webcam (same requirement as the main app).
"""

from __future__ import annotations

import statistics
import time

from eye_detection import WebcamEyeTracker


def collect_samples(tracker: WebcamEyeTracker, seconds: float, label: str) -> list[float]:
    print(f"\n{label}")
    print("Collecting for {:.0f} seconds...".format(seconds), end="", flush=True)
    samples = []
    start = time.time()
    while time.time() - start < seconds:
        result = tracker.read_once()
        if result.ear is not None:
            samples.append(result.ear)
            print(".", end="", flush=True)
        else:
            print("x", end="", flush=True)  # no face detected this frame
    print()
    return samples


def summarize(name: str, samples: list[float]) -> float:
    if not samples:
        print(f"  No face detected during '{name}' -- check lighting/camera and try again.")
        raise SystemExit(1)
    mean = statistics.mean(samples)
    print(f"  {name}: {len(samples)} samples, mean EAR = {mean:.3f} "
          f"(min {min(samples):.3f}, max {max(samples):.3f})")
    return mean


def main():
    print("=" * 60)
    print("EAR Calibration")
    print("=" * 60)
    print("This will look at your face through the webcam for a few short")
    print("intervals. Sit as you normally would while wearing the helmet")
    print("(similar distance/angle to camera), in your usual lighting.")
    input("\nPress Enter when ready to start...")

    tracker = WebcamEyeTracker()
    try:
        tracker.open()

        open_samples = collect_samples(
            tracker, seconds=5,
            label="STEP 1: Keep your eyes OPEN normally, look at the camera.",
        )
        input("\nPress Enter, then close your eyes gently for the next step...")

        closed_samples = collect_samples(
            tracker, seconds=3,
            label="STEP 2: Keep your eyes CLOSED now.",
        )
    finally:
        tracker.release()

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    open_mean = summarize("Eyes open", open_samples)
    closed_mean = summarize("Eyes closed", closed_samples)

    if closed_mean >= open_mean:
        print("\nWarning: closed-eye EAR wasn't lower than open-eye EAR.")
        print("This usually means the face wasn't detected reliably, or eyes")
        print("weren't fully closed. Try again with better lighting / a more")
        print("front-on camera angle.")
        raise SystemExit(1)

    # Threshold closer to the open-eye value than the midpoint: we want to
    # catch "mostly closed" states without waiting for fully-shut eyes,
    # since PERCLOS is meant to catch sustained partial closure too.
    recommended = closed_mean + 0.35 * (open_mean - closed_mean)

    print(f"\nRecommended ear_closed_threshold: {recommended:.3f}")
    print("\nTo apply it, open backend/config.py and change this line inside EyeConfig:")
    print(f"    ear_closed_threshold: float = {recommended:.3f}")
    print("\n(Currently it's set to 0.21 -- the generic default.)")


if __name__ == "__main__":
    main()
