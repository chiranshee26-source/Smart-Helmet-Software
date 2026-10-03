"""
No-blink check: is the eye channel actually seeing real eyes?

Why this exists
---------------
Behind sunglasses the face-landmark model still *guesses* eye positions and
produces an EAR value -- usually a steady "eyes open" number. On the test
webcam the brightness-ratio check (occlusion.py) cannot catch this, because
auto-exposure brightens the frame when the lenses go on.

Real eyes blink every few seconds. A guessed EAR traced off a lens tends to
stay flat. So if no blink (a clear dip in EAR) has been seen for
`no_blink_seconds`, the EAR is treated as unverified, the same as any
other "can't see the eyes" frame. This doesn't depend on the camera,
lighting or sunglasses, and needs no calibration beyond the timeout.

Caveats (documented, not hidden):
- The 5 Hz sensor loop can miss individual short blinks. Over a 30 s
  window with several blinks, missing all of them is unlikely, but a rider
  who genuinely blinks rarely can trip it.
- A long eye closure counts as "eye movement seen" (EAR dips), so it never
  triggers this check -- and is already caught as eyes-closed by PERCLOS.

Pure Python, no OpenCV/mediapipe: unit-tested in tests/test_blink_monitor.py.
"""

from __future__ import annotations

import math
from typing import Optional

from config import EyeConfig


class BlinkMonitor:
    def __init__(self, cfg: Optional[EyeConfig] = None):
        self.cfg = cfg or EyeConfig()
        self._open_baseline: Optional[float] = None
        self._last_t: Optional[float] = None
        self._last_blink_t: Optional[float] = None

    def seconds_since_blink(self, timestamp: float) -> Optional[float]:
        if self._last_blink_t is None:
            return None
        return timestamp - self._last_blink_t

    def update(self, ear: Optional[float], timestamp: float) -> bool:
        """Feed one raw EAR sample (None = no face / no EAR this frame).
        Returns True if the eye reading should be treated as UNVERIFIED
        because no blink has been seen for too long."""
        c = self.cfg
        if ear is None:
            # No reading at all is already handled as "can't see"; restart
            # the clock so that when the face comes back it gets a fair
            # window to show a blink before being doubted.
            self._last_blink_t = timestamp
            self._last_t = timestamp
            return False

        if self._open_baseline is None:
            self._open_baseline = ear
            self._last_blink_t = timestamp
            self._last_t = timestamp
            return False

        dipped = (
            ear < self._open_baseline * c.blink_dip_ratio
            or ear < c.ear_closed_threshold
        )
        if dipped:
            self._last_blink_t = timestamp
        else:
            dt = max(0.0, timestamp - (self._last_t or timestamp))
            alpha = 1.0 - math.exp(-dt / c.blink_baseline_tau_seconds) if dt > 0 else 0.0
            self._open_baseline += alpha * (ear - self._open_baseline)
        self._last_t = timestamp

        if not c.enable_no_blink_check:
            return False
        return (timestamp - self._last_blink_t) >= c.no_blink_seconds
