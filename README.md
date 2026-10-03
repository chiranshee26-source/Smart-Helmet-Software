# IoT-Based Smart Anti-Drowsiness Helmet

**Real-Time Drowsiness Detection and Progressive Rider Safety Alerts**

A conventional helmet only protects a rider *after* an impact. This project
adds an active safety layer on top of it: a system that watches for
measurable signs of drowsiness — prolonged eye closure and repeated head
nodding — and escalates a warning to the rider before fatigue becomes
dangerous.

Hardware (ESP32-CAM + MPU6050) has not arrived yet, so this repository is
the **full software prototype**: the entire detection pipeline, decision
engine, and live dashboard, built and validated using a laptop webcam and a
simulated IMU standing in for the two sensors. Nothing here is a mockup —
the eye tracking, the scoring logic, and the alert state machine are the
real thing, ready to have the sensor layer swapped for real hardware later
with no changes to the logic above it.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [The Solution](#the-solution)
- [What's Working Right Now](#whats-working-right-now)
- [Architecture](#architecture)
- [How Drowsiness Detection Works](#how-drowsiness-detection-works)
- [Edge Cases We Ran Into (and How We Fixed Them)](#edge-cases-we-ran-into-and-how-we-fixed-them)
- [Calibration & Testing](#calibration--testing)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Development Roadmap](#development-roadmap)
- [Known Limitations](#known-limitations)
- [Future Scope](#future-scope)

---

## Problem Statement

Two-wheeler riders may experience fatigue and drowsiness during travel.
Reduced alertness affects attention and reaction time, directly increasing
accident risk. A conventional helmet provides passive physical protection
but has no way to monitor the rider's alertness — it does nothing to
prevent the accident from happening in the first place.

This project adds an **active safety-assistance layer**: a system that
detects measurable indicators of drowsiness in real time and warns the
rider progressively, before the situation becomes critical.

## The Solution

The system works in four stages:

1. **Watch** — a small camera observes eye and facial behaviour for signs
   of prolonged or repeated eye closure.
2. **Sense** — an IMU (MPU6050) measures head movement and orientation;
   repeated downward nods are a supporting signal.
3. **Decide** — both signals are analysed over a rolling time window, never
   on a single frame, and combined into one composite drowsiness score.
4. **Warn** — as signs persist, warnings escalate: vibration, then buzzer,
   then an optional GPS/IoT alert.

## What's Working Right Now

- **Real webcam-based eye tracking** — Eye Aspect Ratio (EAR) computed live
  via face-landmark detection (mediapipe). This is genuinely working, not a
  placeholder for hardware that hasn't arrived.
- **Simulated IMU** standing in for the MPU6050, with injectable head-nod
  events so the full alert escalation can be demoed on demand.
- **The full multi-signal decision engine**: PERCLOS-style eye-closure
  scoring, nod-frequency scoring, sensor fusion, and a GREEN / YELLOW / RED
  state machine.
- **A live web dashboard** (FastAPI + WebSocket) showing the current alert
  level, EAR, head pitch, composite score, and an event log in real time.
- **Automated unit tests** covering escalation, de-escalation, the
  sunglasses heuristic, and the edge cases described below.

## Architecture

```
                         RIDER
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
        ESP32-CAM                   MPU6050
     (Eye / Face Features)      (Head Movement)
              │                         │
              └────────────┬────────────┘
                           ▼
                 DROWSINESS ANALYSIS
                  (multi-signal fusion)
                           │
                           ▼
                     ALERT LEVEL
                (GREEN / YELLOW / RED)
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
           GREEN        YELLOW         RED
        No warning   Vibration +   Buzzer + optional
                       display          GPS/IoT alert
```

In the current software prototype, `ESP32-CAM` is a laptop webcam
(`backend/eye_detection.py`) and `MPU6050` is a simulator
(`backend/imu_simulator.py`). Everything from `DROWSINESS ANALYSIS`
downward is the real, hardware-independent logic in
`backend/decision_engine.py`.

## How Drowsiness Detection Works

No single frame or single head movement is ever treated as proof of
drowsiness — every signal is analysed over a rolling time window.

**Eye / Face Analysis**
- Eye Aspect Ratio (EAR) is computed from live camera frames.
- Sustained or repeated low EAR indicates prolonged eye closure.
- A PERCLOS-style rolling window (percentage of time eyes are closed) is
  used, not a single-frame check.

**Head-Movement Analysis**
- The IMU provides pitch data.
- A "nod event" is registered when pitch deviates from the rider's resting
  pitch (a slowly-adapting baseline) by at least 18°. Because it is measured
  against the baseline rather than sample-to-sample, a slow droop is caught
  the same as a sharp jerk, and the result doesn't depend on the IMU sample
  rate.
- Hysteresis: one nod counts once. The head must return near baseline before
  another nod can register, so the dip and the recovery aren't both counted.
  A deviation held for over 10 s is treated as a posture change and
  re-baselined.
- Nod frequency within the rolling window drives the head-severity score.

**Fusion**

```
composite = max(eye_score, head_score) + 0.3 * min(eye_score, head_score)
            (capped at 1.0)
```

One channel alone reaching a "severe" reading is enough to reach RED on its
own — a fully-closed-eyes-for-15s reading shouldn't need head-motion
corroboration to be treated as critical — while both channels agreeing
pushes the score up faster than either alone, which is the actual
multi-signal benefit called for in the project brief.

**State machine**
- **GREEN** — composite score below 0.35. No warning.
- **YELLOW** — 0.35 to 0.70. Vibration + display warning.
- **RED** — 0.70 or above. Buzzer + strong vibration, optional GPS/IoT
  alert.

Escalation to a higher level is **immediate** — the safety-critical
direction is never delayed. De-escalation requires the improved score to
hold for at least 5 seconds before stepping back down, so a single good
frame right after a bad stretch doesn't cancel a warning and cause flicker.

## Edge Cases We Ran Into (and How We Fixed Them)

Building and testing the prototype against a real camera surfaced real
problems the initial design didn't cover. Each one below was actually
fixed and verified with a test, not just noted as a caveat.

### 1. Camera occlusion / head fully turned away
**Problem:** Originally, if the camera lost the rider's face entirely, the
system simply stopped collecting eye data and stayed GREEN — the exact
opposite of what should happen.
**Fix:** Sustained "no face detected" is now treated the same as sustained
eye closure. `decision_engine.update_eye()` accepts `None` (meaning "no
reading this frame") and PERCLOS counts it as closed — a camera that can't
see the rider is itself a warning condition.

### 2. Sunglasses / lens occlusion
**Problem:** mediapipe's face-landmark model still predicts eye-region
coordinates even when sunglasses cover the eyes — it infers the full face
shape and doesn't know it can't see through the lenses — producing a
confident-looking but meaningless EAR reading.
**Fix (v1):** Added a pixel-based heuristic that inspected the eye-region
pixels and flagged "sunglasses" when the patch was unusually dark, bright
or flat, against fixed brightness thresholds calibrated in one session.

**Problem found with v1:** fixed brightness thresholds don't survive a change
of lighting. In a dimmer room, bare eyes read darker than the calibrated
"sunglasses" threshold, so *every* frame was flagged, every eye reading was
discarded, PERCLOS hit ~100% and the alert locked on RED. On a real helmet
that would happen at dusk or in a tunnel.

**Fix (v2, current):** the heuristic now compares the eye patch with a
bare-skin reference patch on the cheekbone below each eye
(`occlusion.py`). Dim light darkens the whole face roughly equally, so the
eye/cheek brightness ratio stays steady; a tinted lens darkens only the
eyes, so the ratio drops. The dashboard shows the live ratio and the
specific reason an eye reading was distrusted (tinted lens, mirrored
lens/glare, featureless patch, or too dark for the camera to see). The old
absolute thresholds remain only as a fallback when the cheek isn't in
frame. `calibrate_occlusion.py` now includes a dim-light step to check
that bare eyes in low light are *not* flagged.

**What calibration showed (4 Oct, laptop webcam):** the ratio fixed the
dim-light bug (bare-eye brightness fell from 85 to 40, while the ratio stayed
within 0.44–0.64). But the tested sunglasses read *brighter* than bare eyes
(ratio 0.83, range 0.44–1.25): the webcam's auto-exposure brightens the
frame when the lenses go on. **On this camera these sunglasses cannot be
detected by brightness**, so thresholds are set to never flag bare eyes, and
the check only catches extreme cases (opaque patch, strong glare). Reliable
sunglasses handling is listed under Future Scope.

### 3. False alarms on startup / brief blips
**Problem:** Early on, before the 15-second rolling window fills up, a
couple of dropped frames or a single long blink could look like a large
fraction of "bad" data and swing the score disproportionately.
**Fix:** PERCLOS is now weighted by how full the rolling window actually is
(`fill_weight = min(1.0, span / window_seconds)` in
`decision_engine.py._perclos()`), so brief gaps early on can't
disproportionately swing the score.

### 4. mediapipe API breakage across versions
**Problem:** Newer mediapipe releases (0.10.30+) no longer ship the legacy
`mediapipe.solutions.face_mesh` API used during early development, which
broke eye tracking entirely on a fresh install.
**Fix:** `eye_detection.py` auto-detects which API is available at runtime
and falls back from the legacy `FaceMesh` API to the modern
`FaceLandmarker` Tasks API automatically, so the same code works across
mediapipe versions without the user needing to pin an old release.

## Calibration & Testing

The project brief is explicit that thresholds must be calibrated against
real data, not treated as universal constants. Every threshold lives in one
file, `backend/config.py`, clearly labeled, and has been tuned against
actual measurements from the interactive calibration scripts in this repo
(`backend/calibrate.py`, `backend/calibrate_occlusion.py`).

**Eye-closure calibration (EAR):**

| Condition | Mean EAR |
|---|---|
| Eyes open | 0.30 – 0.39 |
| Eyes closed | < 0.15 |

**Sunglasses / occlusion calibration (eye-patch brightness, 0–255 grayscale):**

| Condition | Brightness |
|---|---|
| Eyes open | 106.1 |
| Eyes closed | 96.9 |
| Sunglasses on | 63.4 |

These absolute readings were the v1 calibration. They are now only used as a
fallback; the primary eye/cheek ratio thresholds in `config.py` are
provisional until `calibrate_occlusion.py` is re-run (see edge case 2).

**Automated test coverage** (`tests/`):

- Stays GREEN on normal, steady input
- Escalates to RED on sustained eye closure
- Escalates on repeated head-nod events alone
- Escalation is immediate, never delayed
- De-escalation requires sustained recovery
- Sustained no-face-detected escalates the alert
- Brief blips (a blink, a dropped frame) do not false-alarm
- Occlusion heuristic flags the calibrated sunglasses reading, bright
  reflective lenses, and flat uniform patches
- Occlusion heuristic does not flag the calibrated open- or closed-eye
  readings (a closed eye must not be mistaken for "can't see")
- Off-frame landmarks don't crash or false-flag

Run them with:

```bash
cd smart-helmet-software
python3 -m pytest tests/ -v
```

These test the decision engine and occlusion heuristic directly with
synthetic data — no webcam or IMU required, so they run anywhere, including
CI.

## Project Structure

```
smart-helmet-software/
  backend/
    config.py               # every calibratable threshold, with sourced comments
    decision_engine.py       # fusion + GREEN/YELLOW/RED state machine (no hardware deps)
    eye_detection.py          # webcam -> EAR, dual mediapipe backend
    occlusion.py              # sunglasses/occlusion heuristic (numpy only, unit-tested)
    imu_simulator.py           # stand-in for the MPU6050
    main.py                     # FastAPI app: wires it together, serves the dashboard
    calibrate.py                # interactive EAR calibration against your own camera
    calibrate_occlusion.py       # interactive sunglasses/occlusion calibration
  frontend/
    index.html, style.css, dashboard.js   # the live dashboard (plain JS, no build step)
  tests/
    test_decision_engine.py       # unit tests for the alert logic
    test_occlusion_heuristic.py    # unit tests for the sunglasses heuristic
  requirements.txt
  README.md
```

## Getting Started

```bash
cd smart-helmet-software
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

cd backend
uvicorn main:app --reload --port 8000
```

Then open **http://localhost:8000**. If your camera is free, you'll see
live EAR values as soon as the page loads. If no camera is found, the eye
channel just shows "no camera detected" and the dashboard still works fully
off the simulated head-motion channel — nothing crashes either way.

To see the full progressive alert system without waiting to actually get
drowsy: click **"Inject head-nod event"** on the dashboard. One click stays
GREEN, a second (within 15 s) raises YELLOW, a third raises RED. The alert
then holds for a few seconds after the nods age out of the window before
stepping back down.

## Development Roadmap

**Done:**
- [x] Software decision engine + progressive alert logic
- [x] Webcam-based eye tracking (EAR) integrated and tested
- [x] Simulated IMU + live dashboard, calibration scripts
- [x] Occlusion / sunglasses handling, edge-case fixes

**Planned:**
- [ ] Build ESP32 + MPU6050 hardware prototype
- [ ] Replace webcam/simulator with real camera + IMU streams
- [ ] Integrate vibration motor, buzzer, OLED display
- [ ] Package components on/inside the protective helmet
- [ ] Controlled test data collection, false-positive/negative evaluation
- [ ] Optional GPS/IoT communication, final demonstration

## Known Limitations

- Detects indicators **associated with** drowsiness; it cannot directly
  know whether the rider is asleep.
- Camera reliability is affected by lighting, occlusion, sunglasses,
  helmet fit, and mount placement.
- Head movement alone is ambiguous (e.g. adjusting a helmet strap could
  look like a "nod") — which is exactly why it's fused with the eye channel
  rather than trusted alone.
- False positives and false negatives are possible. No accuracy numbers
  are being claimed before controlled testing with real subjects.
- The occlusion heuristic (sunglasses detection) is a coarse pixel-based
  check, not true occlusion recognition, and needs per-camera/per-lighting
  calibration — it is a partial mitigation, not a fix. The eye/cheek ratio
  makes it robust to overall room brightness, but not to uneven lighting
  (e.g. a strong side light, or a helmet visor shading only the eyes).
- At night a normal camera can't see the rider at all; the system reports
  "too dark" and distrusts the eye channel. A real helmet would need IR
  illumination and an IR-capable camera.
- This is a safety-assistance prototype. It cannot guarantee accident
  prevention.

## Future Scope

| Limitation / Gap | Possible Future Solution |
|---|---|
| Sunglasses not detectable by brightness on the test webcam (auto-exposure compensates) | Near-IR camera + IR illumination (many sunglass lenses pass near-IR, so the eye stays visible; standard in commercial driver-monitoring systems), or a small trained eye-visibility classifier |
| No accuracy numbers yet | Controlled data collection with real test subjects once hardware exists, to measure actual false-positive/negative rates |
| Head motion alone is ambiguous | IMU-based accident/impact detection as an additional, independent signal (sudden deceleration + orientation change) |
| Fixed thresholds per install | Personalised alertness models based on an individual rider's historical behaviour |
| Detection runs only when a laptop/phone-grade CPU is available | Lightweight on-device AI models suitable for the ESP32-CAM's limited compute |
| No way to alert anyone beyond the rider | Automatic emergency notification with GPS after a critical (RED) event |
| No historical record for the rider | A companion mobile app for alertness history and system status |
| Current cost/power profile is for a prototype, not a product | Battery optimisation and a smaller PCB/enclosure for a production-ready build |

---

*Built as part of the Innovative Design Project (IDP) coursework at VIT
Vellore. This README is kept up to date as the project moves from software
prototype to full hardware integration.*
