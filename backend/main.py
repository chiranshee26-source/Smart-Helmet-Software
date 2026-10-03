"""
FastAPI backend for the Smart Anti-Drowsiness Helmet software (no-hardware mode).

Wires together:
  - WebcamEyeTracker  (real webcam, real EAR -- works today, no hardware needed)
  - ImuSimulator       (stand-in for the MPU6050 until hardware arrives)
  - DecisionEngine      (fusion + progressive GREEN/YELLOW/RED alert logic)

...and streams live state to the dashboard over a WebSocket, plus exposes a
couple of REST endpoints for demo control (manually injecting a "head nod"
event, resetting state) so you can demonstrate the full progressive-alert
behaviour on stage without actually having to fall asleep on camera.

Run:
    uvicorn main:app --reload --port 8000
Then open http://localhost:8000 in a browser.

Swapping to real hardware later: replace the `eye_source` and `head_source`
generator functions below with ones that read from your ESP32-CAM / MPU6050
(e.g. over serial or a small HTTP/MQTT endpoint the ESP32 posts to).
DecisionEngine and the dashboard do not need to change at all.
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config import AppConfig
from decision_engine import DecisionEngine

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

cfg = AppConfig()
engine = DecisionEngine(cfg)

# Try to use a real webcam; fall back to eyes-disabled mode if none is
# available (e.g. running this on a headless machine/sandbox). The dashboard
# will just show "no camera" for the eye channel and still demonstrate the
# head-motion-only path.
_eye_tracker = None
_eye_enabled = False


def _try_init_camera():
    global _eye_tracker, _eye_enabled
    try:
        from eye_detection import WebcamEyeTracker

        tracker = WebcamEyeTracker(eye_config=cfg.eye)
        tracker.open()
        _eye_tracker = tracker
        _eye_enabled = True
        print("[startup] Webcam eye tracking ENABLED.")
    except Exception as exc:  # noqa: BLE001 - broad on purpose, this is optional
        _eye_tracker = None
        _eye_enabled = False
        print(f"[startup] Webcam eye tracking DISABLED ({exc}). "
              f"Head-motion channel will still work via the IMU simulator.")


from imu_simulator import ImuSimulator  # noqa: E402

# Occasional spontaneous nods keep the idle demo from looking flat. At 5 Hz,
# 0.002/tick is roughly one nod every ~100 s -- rare enough that a single
# random nod (head score 0.25) never stacks into a false YELLOW on its own.
# (0.01 produced ~10-17 false escalations per 10 idle minutes.)
imu = ImuSimulator(random_nod_probability_per_tick=0.002)

_clients: set[WebSocket] = set()
_running = True


async def _sensor_loop():
    """Background task: pulls a sample from each source, feeds the engine,
    ticks it, and broadcasts the resulting state to all connected dashboards."""
    period = 1.0 / cfg.tick_hz
    while _running:
        now = time.time()

        occlusion_suspected = False
        occlusion_reason = None
        brightness_ratio = None
        eye_patch_mean = None
        seconds_since_blink = None
        face_found = False
        if _eye_enabled and _eye_tracker is not None:
            result = await asyncio.get_event_loop().run_in_executor(
                None, _eye_tracker.read_once
            )
            # Feed the sample even when no face was found (result.ear is
            # None) -- sustained "can't see the eyes at all" is itself a
            # warning condition, not neutral. See decision_engine.update_eye.
            engine.update_eye(result.ear, result.timestamp)
            occlusion_suspected = result.occlusion_suspected
            occlusion_reason = result.occlusion_reason
            brightness_ratio = result.brightness_ratio
            eye_patch_mean = result.eye_patch_mean
            seconds_since_blink = result.seconds_since_blink
            face_found = result.face_found

        imu_sample = imu.tick(now)
        engine.update_head(imu_sample.pitch_deg, imu_sample.timestamp)

        state = engine.tick(now)
        payload = {
            "timestamp": state.timestamp,
            "ear": state.ear,
            "camera_enabled": _eye_enabled,
            "face_found": face_found,
            "occlusion_suspected": occlusion_suspected,
            "occlusion_reason": occlusion_reason,
            "brightness_ratio": round(brightness_ratio, 3) if brightness_ratio is not None else None,
            "eye_patch_mean": round(eye_patch_mean, 1) if eye_patch_mean is not None else None,
            "seconds_since_blink": round(seconds_since_blink, 1) if seconds_since_blink is not None else None,
            "perclos": round(state.perclos, 3),
            "eye_score": round(state.eye_score, 3),
            "head_pitch_deg": round(state.head_pitch_deg, 2) if state.head_pitch_deg is not None else None,
            "nod_count": state.nod_count,
            "head_score": round(state.head_score, 3),
            "composite_score": round(state.composite_score, 3),
            "alert_level": state.alert_level.label,
            "action": state.action,
            "just_changed": state.just_changed,
            "events": engine.recent_events(10),
        }
        dead = []
        for ws in list(_clients):
            try:
                await ws.send_text(json.dumps(payload))
            except Exception:  # noqa: BLE001
                dead.append(ws)
        for ws in dead:
            _clients.discard(ws)

        await asyncio.sleep(period)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _try_init_camera()
    task = asyncio.create_task(_sensor_loop())
    yield
    global _running
    _running = False
    task.cancel()
    if _eye_tracker is not None:
        _eye_tracker.release()


app = FastAPI(title="Smart Helmet Drowsiness Detection (software-only mode)", lifespan=lifespan)


@app.get("/")
async def index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    _clients.add(websocket)
    try:
        while True:
            # We don't expect inbound messages, but keep the connection alive
            # and tolerate the client sending pings/keepalives.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _clients.discard(websocket)


@app.post("/demo/trigger_nod")
async def demo_trigger_nod():
    """Manually inject one head-nod event right now (demo control)."""
    imu.trigger_nod()
    return {"ok": True}


@app.post("/demo/reset")
async def demo_reset():
    """Reset the decision engine back to a clean GREEN state (demo control)."""
    global engine
    engine = DecisionEngine(cfg)
    return {"ok": True}


@app.get("/status")
async def status():
    return {
        "camera_enabled": _eye_enabled,
        "tick_hz": cfg.tick_hz,
        "connected_clients": len(_clients),
        "exposure_locked": getattr(_eye_tracker, "exposure_locked", None),
    }
