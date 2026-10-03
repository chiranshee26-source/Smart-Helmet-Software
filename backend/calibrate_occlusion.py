"""
Interactive calibration for the sunglasses/occlusion heuristic.

Measures, with your real webcam, camera angle and lighting:
  - eye-patch brightness and texture
  - cheek (reference) brightness just below each eye
  - the eye/cheek brightness RATIO, which the heuristic now decides on

under four conditions: eyes open, eyes closed, sunglasses on, and eyes open
in DIM light. The dim-light step checks the whole point of the ratio: bare
eyes in a darker room must NOT look like sunglasses.

It then recommends occlusion_dark_ratio_threshold and
occlusion_bright_ratio_threshold for config.py (plus the absolute fallback
values).

Run from the backend/ folder:
    python calibrate_occlusion.py

You'll need your sunglasses on hand, and a light you can dim or switch off.
"""

from __future__ import annotations

import statistics
import time

from config import EyeConfig
from eye_detection import WebcamEyeTracker


def collect(tracker: WebcamEyeTracker, seconds: float, label: str):
    print(f"\n{label}")
    print(f"Collecting for {seconds:.0f} seconds...", end="", flush=True)
    samples = []  # (eye_mean, eye_std, cheek_mean, ratio)
    start = time.time()
    while time.time() - start < seconds:
        r = tracker.read_once()
        if r.eye_patch_mean is not None:
            samples.append((r.eye_patch_mean, r.eye_patch_std, r.cheek_patch_mean, r.brightness_ratio))
            print("." if r.brightness_ratio is not None else "c", end="", flush=True)
        else:
            print("x", end="", flush=True)  # no face detected this frame
    print("   (. = ok, c = cheek not visible, x = no face)")
    return samples


def summarize(name: str, samples):
    if not samples:
        print(f"  {name}: no face detected -- check lighting/camera and try again.")
        raise SystemExit(1)
    eye = statistics.mean(s[0] for s in samples)
    std = statistics.mean(s[1] for s in samples)
    cheeks = [s[2] for s in samples if s[2] is not None]
    ratios = [s[3] for s in samples if s[3] is not None]
    cheek = statistics.mean(cheeks) if cheeks else None
    ratio = statistics.mean(ratios) if ratios else None
    line = f"  {name:<26} eye={eye:6.1f}  texture={std:5.1f}"
    line += f"  cheek={cheek:6.1f}" if cheek is not None else "  cheek=   n/a"
    if ratio is not None:
        line += f"  ratio={ratio:.2f} (range {min(ratios):.2f}-{max(ratios):.2f})"
    else:
        line += "  ratio=n/a"
    print(line)
    return {"eye": eye, "std": std, "cheek": cheek, "ratio": ratio,
            "ratio_min": min(ratios) if ratios else None,
            "ratio_max": max(ratios) if ratios else None}


def main():
    print("=" * 74)
    print("Occlusion Heuristic Calibration")
    print("=" * 74)
    print("Sit as you normally would, in your usual lighting. Keep your whole")
    print("face (including cheeks) in view. Have your sunglasses within reach,")
    print("and a light you can dim or switch off for the last step.")
    input("\nPress Enter when ready...")

    tracker = WebcamEyeTracker()
    try:
        tracker.open()
        open_s = collect(tracker, 5, "STEP 1: Eyes OPEN, no sunglasses, normal light.")
        input("\nPress Enter, then close your eyes gently...")
        closed_s = collect(tracker, 3, "STEP 2: Eyes CLOSED, no sunglasses.")
        input("\nPress Enter, then put your sunglasses on and look at the camera...")
        glasses_s = collect(tracker, 5, "STEP 3: Sunglasses ON, eyes open behind them.")
        input("\nTake the sunglasses OFF, DIM the lights (keep your face just visible), "
              "then press Enter...")
        dim_s = collect(tracker, 5, "STEP 4: Eyes OPEN, no sunglasses, DIM light.")
    finally:
        tracker.release()

    print("\n" + "=" * 74)
    print("Results")
    print("=" * 74)
    o = summarize("Eyes open", open_s)
    c = summarize("Eyes closed", closed_s)
    g = summarize("Sunglasses on", glasses_s)
    d = summarize("Eyes open, dim light", dim_s)

    cfg = EyeConfig()
    print("\n" + "-" * 74)
    real = [x for x in (o, c, d) if x["ratio"] is not None]
    if g["ratio"] is None or len(real) < 3:
        print("Cheek patch wasn't visible in some steps, so the ratio can't be")
        print("calibrated. Move back so your cheeks are in frame and re-run.")
        raise SystemExit(1)

    # Dark-ratio threshold sits between the sunglasses ratio and the lowest
    # ratio seen for real eyes (open, closed or dim). Using each step's
    # observed min/max, not just the mean, keeps a margin for noise.
    real_low = min(x["ratio_min"] for x in real)
    real_high = max(x["ratio_max"] for x in real)
    print("\nRecommended config.py values (inside EyeConfig):")
    separated = g["ratio_max"] < real_low
    if separated:
        dark_ratio = g["ratio_max"] + 0.5 * (real_low - g["ratio_max"])
    else:
        dark_ratio = min(cfg.occlusion_dark_ratio_threshold, real_low * 0.85)
    bright_ratio = max(real_high * 1.25, real_high + 0.2)
    print(f"    occlusion_dark_ratio_threshold: float = {dark_ratio:.2f}")
    print(f"    occlusion_bright_ratio_threshold: float = {bright_ratio:.2f}")

    # Absolute fallback (only used when the cheek isn't visible).
    abs_dark = g["eye"] + 0.5 * (min(o["eye"], c["eye"]) - g["eye"]) if g["eye"] < min(o["eye"], c["eye"]) else min(o["eye"], c["eye"]) * 0.5
    print(f"    occlusion_dark_mean_threshold: float = {abs_dark:.1f}   # fallback only")

    print()
    if separated:
        print("Good: sunglasses ratio is clearly below every real-eye ratio,")
        print("including in dim light, so the check should work in both.")
    else:
        print("Warning: the sunglasses ratio overlaps with real eyes (open, closed")
        print("or dim). These glasses may not be reliably detected by brightness")
        print("alone -- record this as a known limitation rather than trusting")
        print("the occlusion flag with these glasses.")
    if d["eye"] < cfg.occlusion_dark_mean_threshold:
        print(f"\nNote: in dim light your bare-eye brightness was {d['eye']:.0f}, below the")
        print(f"old fixed threshold of {cfg.occlusion_dark_mean_threshold:.1f}. The old check would have")
        print("flagged you as wearing sunglasses here; the ratio check does not.")


if __name__ == "__main__":
    main()
