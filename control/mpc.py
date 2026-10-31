"""Sampling-based Model Predictive Control for steering.

Every tick we simulate the car's own future: ~100 candidate steering plans (an immediate
steering angle followed by a second one after `switch_time`) are rolled out over a 2 s horizon
with a kinematic bicycle model that includes the steering actuator lag. Each rollout is scored
on distance to the reference path, heading error and steering effort / smoothness, and the
first steering angle of the best plan is applied. Next tick the whole thing is repeated
(receding horizon).
"""
from __future__ import annotations

import math

import numpy as np


class MPCSteering:
    def __init__(self, wheelbase: float, max_steer_deg: float, cfg: dict | None = None):
        cfg = cfg or {}
        self.wheelbase = wheelbase
        self.max_steer = math.radians(max_steer_deg)
        self.horizon = cfg.get("horizon", 2.0)
        self.sim_dt = cfg.get("step", 0.1)
        self.switch_time = cfg.get("switch_time", 0.6)
        self.actuator_tau = cfg.get("actuator_tau", 0.1)
        self.w_lat = cfg.get("w_lateral", 1.0)
        self.w_head = cfg.get("w_heading", 2.0)
        self.w_steer = cfg.get("w_steer", 0.05)
        self.w_rate = cfg.get("w_rate", 8.0)
        # Candidate steering changes (rad): fine steps near the previous command for smooth
        # small corrections, coarse ones for large manoeuvres.
        fine = np.array([0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.15])
        self.d1_offsets = np.concatenate([-fine[::-1], [0.0], fine])   # around the previous command
        coarse = np.array([0.01, 0.03, 0.08, 0.2])
        self.d2_offsets = np.concatenate([-coarse[::-1], [0.0], coarse])  # change for the second phase
        self.prev = 0.0                                         # last commanded wheel angle (+ = left)
        self.last_rollouts: np.ndarray | None = None            # for visualisation
        self.last_best: np.ndarray | None = None

    def set_max_steer(self, max_steer_deg: float):
        self.max_steer = math.radians(max_steer_deg)

    def reset(self):
        self.prev = 0.0

    def _candidates(self) -> np.ndarray:
        d1 = np.clip(self.prev + self.d1_offsets, -self.max_steer, self.max_steer)
        d1g, d2g = np.meshgrid(d1, self.d2_offsets, indexing="ij")
        d2g = np.clip(d1g + d2g, -self.max_steer, self.max_steer)
        return np.stack([d1g.ravel(), d2g.ravel()], axis=1)      # (C, 2)

    def rollout(self, plans: np.ndarray, speed: float, current_wheel: float) -> np.ndarray:
        """Simulate every plan. Returns (C, H, 4): x, y, yaw, wheel angle."""
        C = len(plans)
        H = int(round(self.horizon / self.sim_dt))
        dt = self.sim_dt
        x = np.zeros(C)
        y = np.zeros(C)
        yaw = np.zeros(C)
        wheel = np.full(C, current_wheel)
        out = np.empty((C, H, 4))
        alpha = min(1.0, dt / max(self.actuator_tau, 1e-3))
        v = max(speed, 1.0)
        for k in range(H):
            cmd = plans[:, 0] if k * dt < self.switch_time else plans[:, 1]
            wheel = wheel + (cmd - wheel) * alpha
            x = x + v * np.cos(yaw) * dt
            y = y + v * np.sin(yaw) * dt
            yaw = yaw + v / self.wheelbase * np.tan(wheel) * dt
            out[:, k] = np.stack([x, y, yaw, wheel], axis=1)
        return out

    def cost(self, traj: np.ndarray, plans: np.ndarray, path: np.ndarray) -> np.ndarray:
        # Dense reference path with headings.
        if len(path) < 2:
            path = np.array([[0.0, 0.0], [50.0, 0.0]])
        seg = np.diff(path, axis=0)
        lengths = np.hypot(seg[:, 0], seg[:, 1])
        s = np.concatenate([[0], np.cumsum(lengths)])
        dense_s = np.arange(0, s[-1], 0.25)
        px = np.interp(dense_s, s, path[:, 0])
        py = np.interp(dense_s, s, path[:, 1])
        heading = np.arctan2(np.gradient(py), np.gradient(px))
        pts = np.stack([px, py], 1)                                  # (P, 2)
        d = traj[:, :, None, :2] - pts[None, None]                   # (C, H, P, 2)
        dist2 = (d ** 2).sum(-1)
        idx = dist2.argmin(-1)                                       # (C, H)
        lat2 = np.take_along_axis(dist2, idx[..., None], -1)[..., 0]
        head_err = (traj[:, :, 2] - heading[idx] + np.pi) % (2 * np.pi) - np.pi
        H = traj.shape[1]
        w = np.linspace(0.5, 1.5, H)[None, :]                        # care more about later states
        c = (w * (self.w_lat * lat2 + self.w_head * head_err ** 2)).sum(1) / H
        c += self.w_steer * (plans ** 2).sum(1)
        c += self.w_rate * ((plans[:, 0] - self.prev) ** 2 + 0.5 * (plans[:, 1] - plans[:, 0]) ** 2)
        return c

    def step(self, path: np.ndarray, speed: float, dt: float) -> float:
        """Returns CARLA-convention steer in [-1, 1] (+1 = full right)."""
        plans = self._candidates()
        traj = self.rollout(plans, speed, self.prev)
        costs = self.cost(traj, plans, path)
        best = int(np.argmin(costs))
        self.last_rollouts = traj[:, :, :2]
        self.last_best = traj[best, :, :2]
        self.prev = float(plans[best, 0])
        return float(np.clip(-self.prev / self.max_steer, -1.0, 1.0))
