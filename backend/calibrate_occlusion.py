"""
Interactive calibration for the sunglasses/occlusion heuristic.

Measures the actual eye-patch brightness (mean) and texture (std-dev) under
three conditions -- eyes open, eyes closed, and wearing your sunglasses --
using your real webcam, camera angle and lighting, then recommends values
for occlusion_std_threshold, occlusion_dark_mean_threshold and
occlusion_bright_mean_threshold in config.py.

Run from the backend/ folder:
    python calibrate_occlusion.py

You'll need your sunglasses on hand.
"""

from __future__ import annotations

import statistics
import time

from eye_detection import WebcamEyeTracker


def collect(tracker: WebcamEyeTracker, seconds: float, label: str):
    print(f"\n{label}")
    print(f"Collecting for {seconds:.0f} seconds...", end="", flush=True)
    means, stds = [], []
    start = time.time()
    while time.time() - start < seconds:
        result = tracker.read_once()
        if result.eye_patch_mean is not None:
            means.append(result.eye_patch_mean)
            stds.append(result.eye_patch_std)
            print(".", end="", flush=True)
        else:
            print("x", end="", flush=True)  # no face detected this frame
    print()
    return means, stds


def summarize(name: str, means, stds):
    if not means:
        print(f"  No face detected during '{name}' -- check lighting/camera and try again.")
        raise SystemExit(1)
    mean_avg = statistics.mean(means)
    std_avg = statistics.mean(stds)
    print(f"  {name}: {len(means)} samples -- "
          f"brightness mean={mean_avg:.1f} (range {min(means):.1f}-{max(means):.1f}), "
          f"texture std={std_avg:.1f} (range {min(stds):.1f}-{max(stds):.1f})")
    return mean_avg, std_avg


def main():
    print("=" * 70)
    print("Occlusion Heuristic Calibration")
    print("=" * 70)
    print("This measures how bright/textured the pixels under your eyes look,")
    print("with eyes open, eyes closed, and wearing your sunglasses. Sit as")
    print("you normally would while wearing the helmet, in your usual lighting.")
    print("Have your sunglasses within reach before starting.")
    input("\nPress Enter when ready...")

    tracker = WebcamEyeTracker()
    try:
        tracker.open()

        open_means, open_stds = collect(
            tracker, seconds=5, label="STEP 1: Eyes OPEN, no sunglasses, look at the camera.",
        )
        input("\nPress Enter, then close your eyes gently...")
        closed_means, closed_stds = collect(
            tracker, seconds=3, label="STEP 2: Eyes CLOSED, no sunglasses.",
        )
        input("\nPress Enter, then put your sunglasses on and look at the camera...")
        glasses_means, glasses_stds = collect(
            tracker, seconds=5, label="STEP 3: Sunglasses ON, eyes open behind them.",
        )
    finally:
        tracker.release()

    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    open_mean, open_std = summarize("Eyes open (no glasses)", open_means, open_stds)
    closed_mean, closed_std = summarize("Eyes closed (no glasses)", closed_means, closed_stds)
    glasses_mean, glasses_std = summarize("Sunglasses on", glasses_means, glasses_stds)

    print("\n" + "-" * 70)

    # std threshold: should sit between the sunglasses' std and the lowest
    # "real eye" std (open or closed, whichever is lower), so real eyes
    # never get flagged but the flatter glasses patch does.
    real_eye_min_std = min(open_std, closed_std)
    if glasses_std < real_eye_min_std:
        recommended_std = glasses_std + 0.5 * (real_eye_min_std - glasses_std)
        std_ok = True
    else:
        recommended_std = real_eye_min_std * 0.6
        std_ok = False

    # dark/bright thresholds: should sit between the sunglasses' brightness
    # and the nearest real-eye brightness, on whichever side the glasses
    # fall.
    real_eye_means = [open_mean, closed_mean]
    recommended_dark = min(real_eye_means) * 0.5
    recommended_bright = max(real_eye_means) + (255 - max(real_eye_means)) * 0.5
    dark_bright_ok = False
    if glasses_mean < min(real_eye_means):
        recommended_dark = glasses_mean + 0.5 * (min(real_eye_means) - glasses_mean)
        dark_bright_ok = True
    elif glasses_mean > max(real_eye_means):
        recommended_bright = max(real_eye_means) + 0.5 * (glasses_mean - max(real_eye_means))
        dark_bright_ok = True

    print("\nRecommended config.py values (inside EyeConfig):")
    print(f"    occlusion_std_threshold: float = {recommended_std:.1f}")
    print(f"    occlusion_dark_mean_threshold: float = {recommended_dark:.1f}")
    print(f"    occlusion_bright_mean_threshold: float = {recommended_bright:.1f}")

    if not std_ok and not dark_bright_ok:
        print("\nWarning: your sunglasses' eye-patch stats (mean, std) landed inside")
        print("the same range as your real open/closed eyes on BOTH measures. This")
        print("heuristic likely will NOT reliably catch these specific glasses --")
        print("that's a real limitation of a pixel-brightness-only approach, not a")
        print("calibration problem. Document this as a known gap rather than")
        print("trusting the occlusion flag with these glasses on.")
    elif not std_ok:
        print("\nNote: the texture (std) measurement didn't separate cleanly, but")
        print("brightness did -- the recommended thresholds above should still work")
        print("reasonably, driven mainly by the dark/bright check.")
    elif not dark_bright_ok:
        print("\nNote: the brightness measurement didn't separate cleanly, but")
        print("texture (std) did -- the recommended thresholds above should still")
        print("work reasonably, driven mainly by the uniformity check.")
    else:
        print("\nGood separation on both measures -- these thresholds should")
        print("reliably distinguish your sunglasses from real open/closed eyes.")


if __name__ == "__main__":
    main()
