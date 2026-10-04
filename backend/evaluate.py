"""
Replay recorded sessions through the decision engine and report how well
it did.

    python evaluate.py ../recordings/*.csv
    python evaluate.py ../recordings/*.csv --compare-legacy
    python evaluate.py ../recordings/*.csv --report ../recordings/results.md

Metrics (eye channel only -- the head channel is still simulated, so it is
held at a steady pitch during replay and doesn't affect results):

  Drowsy phases (acted drowsiness):
    detection rate     share of drowsy phases where the alert reached YELLOW
                       (or RED) before the phase ended + a short grace period
    time to YELLOW/RED seconds from the phase start to that alert level
  Alert phases:
    false alarms       separate escalations GREEN -> YELLOW/RED while the
                       person was alert, per hour of alert time
    time in warning    share of alert time spent at YELLOW or RED

Recovery time after a drowsy phase is excluded from the false-alarm count:
the alert is *designed* to hold for a while after drowsiness ends (the
PERCLOS window has to clear, plus the de-escalation dwell), so counting
that as a false alarm would be wrong.

--compare-legacy replays the same recordings with the pre-4-Oct eye logic
(every frame below EAR 0.21 counts as closed; no blink filter, no personal
baseline), so a report can show before/after on identical data.

Pure Python: no camera needed. Unit-tested in tests/test_evaluate.py.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import statistics
import sys
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

from config import AppConfig
from decision_engine import AlertLevel, DecisionEngine
from protocol import ALERT, DROWSY

DETECTION_GRACE_SECONDS = 5.0


@dataclass
class Sample:
    t: float
    phase: str
    label: str
    ear: Optional[float]


@dataclass
class PhaseResult:
    session: str
    phase: str
    label: str
    start: float
    end: float
    detected: bool = False                 # drowsy phases
    time_to_yellow: Optional[float] = None
    time_to_red: Optional[float] = None
    already_warning: bool = False          # alert was already >= YELLOW when the phase began
    false_alarms: int = 0                  # alert phases
    counted_seconds: float = 0.0           # alert time used for false-alarm stats
    warning_seconds: float = 0.0

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Summary:
    name: str
    phases: List[PhaseResult] = field(default_factory=list)

    def _drowsy(self):
        return [p for p in self.phases if p.label == DROWSY]

    def _alert(self):
        return [p for p in self.phases if p.label == ALERT]

    @property
    def drowsy_phases(self) -> int:
        return len(self._drowsy())

    @property
    def detected(self) -> int:
        return sum(p.detected for p in self._drowsy())

    @property
    def detection_rate(self) -> Optional[float]:
        return self.detected / self.drowsy_phases if self.drowsy_phases else None

    @property
    def mean_time_to_yellow(self) -> Optional[float]:
        vals = [p.time_to_yellow for p in self._drowsy() if p.time_to_yellow is not None]
        return statistics.mean(vals) if vals else None

    @property
    def mean_time_to_red(self) -> Optional[float]:
        vals = [p.time_to_red for p in self._drowsy() if p.time_to_red is not None]
        return statistics.mean(vals) if vals else None

    @property
    def alert_hours(self) -> float:
        return sum(p.counted_seconds for p in self._alert()) / 3600.0

    @property
    def false_alarms(self) -> int:
        return sum(p.false_alarms for p in self._alert())

    @property
    def false_alarms_per_hour(self) -> Optional[float]:
        return self.false_alarms / self.alert_hours if self.alert_hours > 0 else None

    @property
    def warning_fraction(self) -> Optional[float]:
        total = sum(p.counted_seconds for p in self._alert())
        return sum(p.warning_seconds for p in self._alert()) / total if total else None


def legacy_config() -> AppConfig:
    """The eye logic as it was before the 4 Oct blink/squint fix."""
    cfg = AppConfig()
    cfg.eye = replace(cfg.eye, min_closure_seconds=0.0, min_plausible_open_ear=99.0)
    return cfg


def load_session(path: str) -> List[Sample]:
    with open(path, newline="") as fh:
        return [
            Sample(
                t=float(row["t"]),
                phase=row["phase"],
                label=row["label"],
                ear=float(row["ear"]) if row["ear"] not in ("", None) else None,
            )
            for row in csv.DictReader(fh)
        ]


def _phase_spans(samples: List[Sample]):
    spans, start = [], 0
    for i in range(1, len(samples) + 1):
        if i == len(samples) or samples[i].phase != samples[start].phase:
            spans.append((start, i))
            start = i
    return spans


def evaluate_session(samples: List[Sample], cfg: AppConfig, session: str = "") -> List[PhaseResult]:
    if not samples:
        return []
    engine = DecisionEngine(cfg)
    levels: List[AlertLevel] = []
    for s in samples:
        engine.update_eye(s.ear, s.t)
        engine.update_head(0.0, s.t)
        levels.append(engine.tick(s.t).alert_level)

    recovery = cfg.eye.perclos_window_seconds + cfg.fusion.min_dwell_before_deescalate
    results: List[PhaseResult] = []
    last_drowsy_end: Optional[float] = None
    times = [s.t for s in samples]

    for a, b in _phase_spans(samples):
        first = samples[a]
        end = samples[b - 1].t
        pr = PhaseResult(session, first.phase, first.label, first.t, end)

        if first.label == DROWSY:
            pr.already_warning = levels[a] >= AlertLevel.YELLOW
            horizon = end + DETECTION_GRACE_SECONDS
            for i in range(a, len(samples)):
                if times[i] > horizon:
                    break
                if levels[i] >= AlertLevel.YELLOW and pr.time_to_yellow is None:
                    pr.time_to_yellow = times[i] - first.t
                    pr.detected = True
                if levels[i] >= AlertLevel.RED and pr.time_to_red is None:
                    pr.time_to_red = times[i] - first.t
            last_drowsy_end = end
        else:
            count_from = first.t
            if last_drowsy_end is not None:
                count_from = max(count_from, last_drowsy_end + recovery)
            for i in range(a, b):
                if times[i] < count_from:
                    continue
                dt = (times[i + 1] - times[i]) if i + 1 < len(times) else 0.0
                pr.counted_seconds += dt
                if levels[i] >= AlertLevel.YELLOW:
                    pr.warning_seconds += dt
                    prev = levels[i - 1] if i > 0 else AlertLevel.GREEN
                    if prev == AlertLevel.GREEN:
                        pr.false_alarms += 1
        results.append(pr)
    return results


def evaluate(paths: List[str], cfg: AppConfig, name: str) -> Summary:
    summary = Summary(name)
    for path in paths:
        summary.phases.extend(
            evaluate_session(load_session(path), cfg, session=os.path.basename(path))
        )
    return summary


def _fmt(x, pct=False, unit=""):
    if x is None:
        return "n/a"
    return f"{x * 100:.0f}%" if pct else f"{x:.1f}{unit}"


def render_report(summaries: List[Summary], n_sessions: int) -> str:
    out = ["# Evaluation results", ""]
    out.append(f"Sessions: {n_sessions} (seated, acted drowsiness; eye channel only).")
    out.append("")
    head = "| Metric | " + " | ".join(s.name for s in summaries) + " |"
    out += [head, "|---|" + "---|" * len(summaries)]
    rows = [
        ("Drowsy phases detected", lambda s: f"{s.detected}/{s.drowsy_phases} ({_fmt(s.detection_rate, pct=True)})"),
        ("Mean time to YELLOW", lambda s: _fmt(s.mean_time_to_yellow, unit=" s")),
        ("Mean time to RED", lambda s: _fmt(s.mean_time_to_red, unit=" s")),
        ("False alarms (alert phases)", lambda s: f"{s.false_alarms} in {s.alert_hours * 60:.1f} min"),
        ("False alarms per hour", lambda s: _fmt(s.false_alarms_per_hour)),
        ("Alert time spent in warning", lambda s: _fmt(s.warning_fraction, pct=True)),
    ]
    for label, fn in rows:
        out.append(f"| {label} | " + " | ".join(fn(s) for s in summaries) + " |")

    for s in summaries:
        out += ["", f"## Per phase: {s.name}", "",
                "| Session | Phase | Expected | Result |", "|---|---|---|---|"]
        for p in s.phases:
            if p.label == DROWSY:
                res = (f"detected: YELLOW at {p.time_to_yellow:.1f} s"
                       + (f", RED at {p.time_to_red:.1f} s" if p.time_to_red is not None else "")
                       if p.detected else "**missed**")
                if p.already_warning:
                    res += " (already in warning when the phase began -- carried over, not a fresh detection)"
            else:
                res = (f"{p.false_alarms} false alarm(s), {p.warning_seconds:.0f} s in warning"
                       f" (of {p.counted_seconds:.0f} s counted)")
            out.append(f"| {p.session} | {p.phase} | {p.label} | {res} |")
    out += ["", "_Recovery time after each drowsy phase "
            "(PERCLOS window + de-escalation dwell) is excluded from false-alarm counts._"]
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="Recorded session CSVs (wildcards OK)")
    ap.add_argument("--compare-legacy", action="store_true",
                    help="Also replay with the pre-4-Oct eye logic for a before/after table")
    ap.add_argument("--report", help="Write the report to this Markdown file too")
    args = ap.parse_args(argv)

    paths = sorted({p for pat in args.paths for p in (glob.glob(pat) or [pat])})
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        print(f"Not found: {', '.join(missing)}")
        return 1

    summaries = [evaluate(paths, AppConfig(), "Current")]
    if args.compare_legacy:
        summaries.append(evaluate(paths, legacy_config(), "Legacy (before 4 Oct)"))
    report = render_report(summaries, len(paths))
    print(report)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            fh.write(report + "\n")
        print(f"\nWritten to {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
