"""Lateral control: PID on the pure-pursuit steering angle toward the look-ahead point.

The geometric pure-pursuit angle  delta = atan(2 L sin(alpha) / ld)  is what a perfect
bicycle-model car would need to reach the target point; the PID corrects for actuator lag,
model mismatch and steady-state bias (e.g. road camber in CARLA).
"""
from __future__ import annotations

import math

import numpy as np

from control.pid import PID


class SteeringController:
    def __init__(self, pid_cfg: dict, wheelbase: float, max_steer_deg: float):
        # The integral only trims small biases; capping it at 15% of full lock avoids wind-up in long turns.
        ki = pid_cfg["ki"]
        self.pid = PID(pid_cfg["kp"], ki, pid_cfg["kd"], pid_cfg.get("limit", 1.0),
                       integral_limit=0.15 / ki if ki > 0 else 0.0)
        self.wheelbase = wheelbase
        self.max_steer = math.radians(max_steer_deg)

    def set_max_steer(self, max_steer_deg: float):
        self.max_steer = math.radians(max_steer_deg)

    def step(self, target: np.ndarray, dt: float) -> float:
        """target: (x, y) look-ahead point in the vehicle frame. Returns CARLA steer (+1 = right)."""
        x, y = float(target[0]), float(target[1])
        ld = max(math.hypot(x, y), 1e-3)
        alpha = math.atan2(y, x)
        delta = math.atan2(2.0 * self.wheelbase * math.sin(alpha), ld)     # + = turn left
        left_cmd = self.pid.step(delta / self.max_steer, dt)
        return -left_cmd

    def reset(self):
        self.pid.reset()
