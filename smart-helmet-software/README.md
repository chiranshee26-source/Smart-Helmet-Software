# Smart Anti-Drowsiness Helmet — Software (no-hardware mode)

This is the software side of the IDP project ("IoT-Based Smart Helmet with
Real-Time Drowsiness Detection and Progressive Rider Safety Alerts"), built
to run and demo **today, on your laptop, with no ESP32/MPU6050/ESP32-CAM
hardware needed**. When the hardware arrives, you swap two small input
modules and everything else — the decision logic, the dashboard, the alert
levels — stays exactly the same.

## What's actually working right now

- **Real eye-drowsiness detection** using your laptop webcam (EAR — Eye
  Aspect Ratio — via mediapipe FaceMesh). This is not a mock; it's the same
  technique the ESP32-CAM channel is meant to approximate.
- **Simulated IMU** standing in for the MPU6050 (a slowly drifting head
  pitch signal + a way to inject "nod" events, either randomly for an idle
  demo, or on demand from the dashboard for a controlled demo).
- **The multi-signal decision engine**: combines an eye-closure severity
  score (PERCLOS-style, over a rolling window) and a head-nod-frequency
  score into one composite drowsiness score, mapped to GREEN / YELLOW / RED
  with immediate escalation and debounced de-escalation (so it doesn't
  flicker).
- **A live web dashboard** showing the current alert level, EAR/PERCLOS,
  head pitch, the composite score against the YELLOW/RED thresholds, and an
  event log — this is also the natural home for the "GPS/IoT dashboard"
  mentioned in the project doc's future scope.
- **Demo controls** (buttons on the dashboard) to inject a head-nod event or
  reset to GREEN, so you can demonstrate the full progressive-alert
  behavior on stage without needing to actually get drowsy on camera.

## Project layout

```
smart-helmet-software/
  backend/
    config.py           # every calibratable threshold lives here
    decision_engine.py   # fusion + GREEN/YELLOW/RED state machine (no hardware deps)
    eye_detection.py      # webcam -> EAR, via mediapipe FaceMesh
    imu_simulator.py       # stand-in for the MPU6050
    main.py                 # FastAPI app: wires it together, serves the dashboard
  frontend/
    index.html, style.css, dashboard.js   # the live dashboard (plain JS, no build step)
  tests/
    test_decision_engine.py                # unit tests for the alert logic, no camera needed
  requirements.txt
```

## Running it

```bash
cd smart-helmet-software
python3 -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt

cd backend
uvicorn main:app --reload --port 8000
```

Then open **http://localhost:8000** in a browser. If your laptop has a
webcam and no other app is using it, you should see live EAR values as soon
as the page loads. If no camera is found, the eye channel just shows
"no camera detected" and the dashboard still works fully off the simulated
head-motion channel — nothing crashes either way.

To see the full progressive alert system without waiting to actually get
drowsy: click **"Inject head-nod event"** a few times in a row on the
dashboard. You'll see the composite score climb and the alert escalate
GREEN → YELLOW → RED, then hold RED for a few seconds before recovering.

### Running the tests

```bash
cd smart-helmet-software
python3 -m pytest tests/ -v
# or, without pytest installed:
python3 tests/test_decision_engine.py
```

These test the decision engine's escalation and de-escalation logic
directly with synthetic data — no webcam or IMU required, so they'll run
anywhere, including CI.

## How the detection actually works

This follows the project doc's "don't trust a single frame" principle:

1. **Eye channel**: every webcam frame produces one EAR value. Over a
   rolling 15-second window, the engine computes PERCLOS — the fraction of
   that window where EAR was below a "closed" threshold — and maps it to an
   eye-severity score (0 = fine, 1 = severe).
2. **Head channel**: every IMU sample (currently simulated) is checked
   against the previous one; a large enough pitch jump counts as a "nod
   event." The number of nod events in the last 15 seconds maps to a
   head-severity score the same way.
3. **Fusion**: `composite = max(eye_score, head_score) + 0.3 * min(eye_score, head_score)`,
   capped at 1.0. This means one channel alone hitting "severe" is enough to
   reach RED on its own (fully-closed-eyes-for-15s shouldn't need head-motion
   corroboration to be treated as critical) — while both channels agreeing
   pushes the score up faster than either alone, which is the actual
   "multi-signal decision" benefit the project doc calls for.
4. **State machine**: composite score is mapped to GREEN (<0.35) / YELLOW
   (0.35–0.70) / RED (≥0.70). Escalating to a higher level happens
   immediately — no debounce, because the safety-critical direction should
   never be delayed. Dropping to a lower level requires the lower score to
   hold for a few seconds first, so a single good frame right after a bad
   stretch doesn't cancel the warning.

**Every threshold above lives in `backend/config.py`, clearly labeled.**
The project doc is explicit that these numbers need experimental
calibration, not treated as universal — this is the one file to tune once
you have real test subjects and a real camera mount.

## Moving to real hardware later

Nothing above the sensor layer needs to change. Specifically:

- Replace `imu_simulator.py`'s `ImuSimulator.tick()` calls in `main.py`
  with whatever reads real pitch (and optionally roll/accel) off the
  MPU6050 — e.g. the ESP32 posts JSON over Wi-Fi to a small endpoint you add
  to `main.py`, or you read it over serial. As long as you keep calling
  `engine.update_head(pitch_deg, timestamp)`, the rest is unchanged.
- Replace `eye_detection.py`'s `WebcamEyeTracker` with a reader for the
  ESP32-CAM's stream (MJPEG over Wi-Fi is the common approach), running the
  same EAR computation on each received frame, and keep calling
  `engine.update_eye(ear, timestamp)`.
- `decision_engine.py`, `main.py`'s WebSocket loop, and the whole dashboard
  do not need to change at all.
- The vibration motor / buzzer / OLED are the next layer to add: when
  `main.py` sees `state.just_changed`, that's the exact moment to also send
  a command to the ESP32 (e.g. a tiny HTTP POST or MQTT message) telling it
  which actuator to trigger for the new alert level. That hook is already
  isolated in the sensor loop (`_sensor_loop` in `main.py`), so it's a
  small, contained addition when you get there.
- GPS/IoT critical-event notification (optional, per the doc) plugs in the
  same way: trigger it when `state.alert_level == AlertLevel.RED` and
  `state.just_changed`.

## Known limitations (carried over from the project doc, on purpose)

- This cannot know whether someone is actually asleep — it only detects
  measurable indicators (eye closure duration, head-motion patterns) that
  correlate with drowsiness.
- The webcam/camera channel will be less reliable with sunglasses, poor
  lighting, or an off-angle mount — same caveat the doc raises for the
  ESP32-CAM.
- Head-motion alone is ambiguous (e.g. adjusting a helmet strap could look
  like a "nod" to a naive detector) — which is exactly why this is fused
  with the eye channel rather than used alone for anything beyond YELLOW in
  practice; calibrate `head.nod_pitch_delta_threshold` in `config.py`
  against your real IMU noise floor once hardware is in hand.
- Don't present any accuracy numbers before you've actually run controlled
  tests with this — the doc's team checklist calls this out explicitly, and
  it applies just as much to this software prototype.

## Troubleshooting

- **"Webcam eye tracking DISABLED"** at startup: either no camera was found
  at index 0, or another app is using it. Current mediapipe releases
  (0.10.30+) no longer ship the legacy `mediapipe.solutions.face_mesh` API —
  `eye_detection.py` detects this automatically and falls back to
  mediapipe's modern Tasks API instead, which needs a small face-landmark
  model file (~4MB). That file is downloaded automatically on first run
  from `storage.googleapis.com` and cached in `~/.cache/smart_helmet/`, so
  the **first** startup with a working camera needs internet access once;
  after that it's cached and works offline. If your network blocks that
  domain, download `face_landmarker.task` manually from
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
  and place it at `~/.cache/smart_helmet/face_landmarker.task` yourself.
- **Port 8000 already in use**: `uvicorn main:app --reload --port 8001` and
  open that port instead.
