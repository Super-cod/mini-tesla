"""Longitudinal control: PID on speed error, split into throttle (positive) and brake (negative)."""
from __future__ import annotations

from control.pid import PID
from utils.types import Control


class SpeedController:
    def __init__(self, pid_cfg: dict, max_brake_decel: float = 8.0):
        self.pid = PID(pid_cfg["kp"], pid_cfg["ki"], pid_cfg["kd"], pid_cfg.get("limit", 1.0))
        self.max_brake_decel = max_brake_decel

    def step(self, target_speed: float, speed: float, dt: float, hard_brake: bool = False) -> tuple[float, float]:
        """Returns (throttle, brake), each in [0, 1]."""
        if hard_brake:
            self.pid.reset()
            return 0.0, 1.0
        if target_speed < 0.05 and speed < 0.5:
            self.pid.reset()
            return 0.0, 0.6                                     # hold the car at standstill
        error = target_speed - speed
        if error < -1.5:
            # Need to slow down noticeably: brake proportional to the required deceleration.
            self.pid.integral = min(self.pid.integral, 0.0)
            return 0.0, min(1.0, 0.12 + (-error) * 0.15)
        u = self.pid.step(error, dt)
        if u >= 0:
            return u, 0.0
        return 0.0, min(1.0, -u * 0.6) if u < -0.05 else 0.0

    def control(self, target_speed: float, speed: float, steer: float, dt: float, hard_brake: bool = False) -> Control:
        throttle, brake = self.step(target_speed, speed, dt, hard_brake)
        return Control(throttle=throttle, steer=steer, brake=brake)

    def reset(self):
        self.pid.reset()
