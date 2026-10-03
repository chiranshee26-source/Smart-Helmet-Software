# Smart Anti-Drowsiness Helmet — software prototype

IDP coursework (VIT Vellore). Detects drowsiness from eye closure (EAR/PERCLOS)
and head nods (IMU pitch), fuses them, and drives a GREEN/YELLOW/RED alert.
Hardware (ESP32-CAM + MPU6050) is not here yet: a laptop webcam and
`imu_simulator.py` stand in for it.

## Layout
- `backend/config.py` — EVERY threshold lives here. Change numbers here, not in logic.
- `backend/decision_engine.py` — pure logic, no hardware/OpenCV deps. Fusion + state machine.
- `backend/eye_detection.py` — webcam -> EAR via mediapipe (legacy FaceMesh or Tasks API), sunglasses heuristic.
- `backend/imu_simulator.py` — fake MPU6050 pitch stream with injectable nods.
- `backend/main.py` — FastAPI app, 5 Hz sensor loop, WebSocket `/ws`, demo endpoints `/demo/trigger_nod`, `/demo/reset`.
- `backend/calibrate*.py` — interactive calibration scripts (need a webcam).
- `frontend/` — plain HTML/CSS/JS dashboard, no build step.
- `tests/` — pytest, synthetic data only (no camera needed).

## Commands
- Tests: `python -m pytest tests/ -q` (from repo root)
- Run app: `cd backend && uvicorn main:app --reload --port 8000`

## Conventions
- Backend modules import each other flat (`from config import ...`); tests add `backend/` to `sys.path`.
- `decision_engine.py` must stay hardware-independent.
- Escalation is immediate; de-escalation waits `min_dwell_before_deescalate`.
- Don't claim accuracy numbers in docs — no real-subject evaluation exists yet.
- Work on a branch per change and open a PR; don't push to `main`.

## Don't read
`.venv/`, `__pycache__/`, `*.task` model files.
