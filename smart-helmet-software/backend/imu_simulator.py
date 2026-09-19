"""
Simulated MPU6050 head-motion stream.

Stands in for the real IMU until hardware arrives. Produces a slowly
drifting "head pitch" signal (degrees) with small natural jitter, and lets
you inject discrete "nod" events (a sharp pitch drop-and-recover, like a
head dipping forward) either randomly (to make an idle demo look alive) or
on demand (for a controlled demo of the progressive alert system).

Only produces (pitch_deg, timestamp) samples -- decision_engine.py doesn't
care whether they came from this file or a real MPU6050 over serial/Wi-Fi.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class ImuSample:
    timestamp: float
    pitch_deg: float
    nod_injected: bool = False


class ImuSimulator:
    def __init__(
        self,
        base_pitch_deg: float = 0.0,
        jitter_deg: float = 1.5,
        random_nod_probability_per_tick: float = 0.0,
        nod_amplitude_deg: float = 25.0,
        seed: Optional[int] = None,
    ):
        """
        base_pitch_deg: resting head pitch, roughly upright.
        jitter_deg: natural small head movement noise per tick.
        random_nod_probability_per_tick: chance [0-1] each tick spontaneously
            produces a "nod" event, for an idle demo that isn't perfectly flat.
            Set to 0 for fully manual control.
        nod_amplitude_deg: how large a simulated nod event is.
        """
        self.base_pitch = base_pitch_deg
        self.jitter = jitter_deg
        self.random_nod_p = random_nod_probability_per_tick
        self.nod_amplitude = nod_amplitude_deg
        self._rng = random.Random(seed)
        self._t0 = time.time()
        self._forced_nod_pending = False

    def trigger_nod(self) -> None:
        """Manually queue a single nod event on the next tick (for demo/API control)."""
        self._forced_nod_pending = True

    def tick(self, timestamp: Optional[float] = None) -> ImuSample:
        ts = timestamp if timestamp is not None else time.time()

        # Slow sinusoidal drift + noise so an idle chart isn't a flat line.
        drift = 2.0 * math.sin((ts - self._t0) / 20.0)
        noise = self._rng.uniform(-self.jitter, self.jitter)
        pitch = self.base_pitch + drift + noise

        nod = False
        if self._forced_nod_pending:
            nod = True
            self._forced_nod_pending = False
        elif self._rng.random() < self.random_nod_p:
            nod = True

        if nod:
            pitch += self.nod_amplitude * self._rng.choice([-1, 1])

        return ImuSample(timestamp=ts, pitch_deg=pitch, nod_injected=nod)


if __name__ == "__main__":
    sim = ImuSimulator(random_nod_probability_per_tick=0.05)
    for _ in range(50):
        sample = sim.tick()
        marker = " <-- NOD" if sample.nod_injected else ""
        print(f"{sample.timestamp:.2f}  pitch={sample.pitch_deg:6.2f} deg{marker}")
        time.sleep(0.1)
