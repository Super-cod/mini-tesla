"""Discrete PID with output clamping, integral anti-windup and derivative filtering."""
from __future__ import annotations


class PID:
    def __init__(self, kp: float, ki: float = 0.0, kd: float = 0.0, limit: float = 1.0,
                 integral_limit: float | None = None, d_filter: float = 0.5):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.limit = limit
        self.integral_limit = integral_limit if integral_limit is not None else limit / max(ki, 1e-6)
        self.d_filter = d_filter          # 0 = no filtering, ->1 = heavy filtering
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.prev_error: float | None = None
        self.d_state = 0.0

    def step(self, error: float, dt: float) -> float:
        if self.prev_error is None:
            self.prev_error = error
        raw_d = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.d_state = self.d_filter * self.d_state + (1 - self.d_filter) * raw_d
        self.prev_error = error
        candidate = self.integral + error * dt
        out = self.kp * error + self.ki * candidate + self.kd * self.d_state
        # Anti-windup: only integrate when not saturated, or when integrating pulls back from the limit.
        if abs(out) < self.limit or (out > 0) != (error > 0):
            self.integral = max(-self.integral_limit, min(self.integral_limit, candidate))
        out = self.kp * error + self.ki * self.integral + self.kd * self.d_state
        return max(-self.limit, min(self.limit, out))
