"""Per-run driving metrics."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from utils.types import Action, Decision, WorldModel


class MetricsRecorder:
    def __init__(self, dt: float):
        self.dt = dt
        self.steps = 0
        self.speeds: list[float] = []
        self.lane_errors: list[float] = []       # route deviation while plain lane following
        self.actions: Counter = Counter()
        self.lane_frames = 0
        self.lane_expected = 0
        self.min_ped_gap = float("inf")
        self.overtakes = 0
        self.stopped_for_pedestrian = False
        self.perception_ms: list[float] = []
        self._prev_action: Action | None = None
        self.status: dict = {}

    def record(self, world: WorldModel, decision: Decision, status: dict, perception_ms: float | None = None):
        self.steps += 1
        self.status = status
        self.speeds.append(world.speed)
        self.actions[decision.action.value] += 1
        if perception_ms is not None:
            self.perception_ms.append(perception_ms)
        if not world.at_junction and not status.get("in_junction", False):
            self.lane_expected += 1
            if world.lane is not None and world.lane.confidence >= 0.5:
                self.lane_frames += 1
        if decision.action == Action.LANE_FOLLOW and "lateral_error" in status:
            self.lane_errors.append(abs(status["lateral_error"]))
        if decision.action == Action.AVOID_OBSTACLE and self._prev_action != Action.AVOID_OBSTACLE:
            self.overtakes += 1
        if decision.action == Action.EMERGENCY_STOP and world.speed < 0.3:
            self.stopped_for_pedestrian = True
        for gt in status.get("ground_truth", []):
            if gt.cls == "pedestrian" and gt.x > 0:
                self.min_ped_gap = min(self.min_ped_gap, float(np.hypot(gt.x, gt.y)))
        self._prev_action = decision.action

    def summary(self) -> dict:
        s = self.status
        errs = np.array(self.lane_errors) if self.lane_errors else np.zeros(1)
        out = {
            "steps": self.steps,
            "sim_time_s": round(self.steps * self.dt, 2),
            "collisions": s.get("collisions", 0),
            "lane_invasions": s.get("lane_invasions"),
            "progress_m": round(s.get("progress", 0.0), 1),
            "route_length_m": round(s.get("route_length", 0.0), 1),
            "completion": round(s.get("progress", 0.0) / s["route_length"], 3) if s.get("route_length") else None,
            "avg_speed_mps": round(float(np.mean(self.speeds)) if self.speeds else 0.0, 2),
            "lane_error_mean_m": round(float(errs.mean()), 3),
            "lane_error_rms_m": round(float(np.sqrt((errs ** 2).mean())), 3),
            "lane_error_max_m": round(float(errs.max()), 3),
            "lane_detection_rate": round(self.lane_frames / max(1, self.lane_expected), 3),
            "overtakes": self.overtakes,
            "stopped_for_pedestrian": self.stopped_for_pedestrian,
            "min_pedestrian_distance_m": None if self.min_ped_gap == float("inf") else round(self.min_ped_gap, 2),
            "perception_ms_mean": round(float(np.mean(self.perception_ms)), 1) if self.perception_ms else None,
            "actions": dict(self.actions),
        }
        return {k: v for k, v in out.items() if v is not None}

    def save(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.summary(), indent=2))
