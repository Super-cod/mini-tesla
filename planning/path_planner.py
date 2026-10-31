"""Turns the decision + world model into a reference path and a look-ahead target point.

Two path sources:
  * the perceived lane centre (camera lane following), and
  * the navigation route from the map (CARLA waypoints / sim route) - used at intersections,
    during lane changes and when the lane markings are lost.
"""
from __future__ import annotations

import numpy as np

from utils.types import Decision, WorldModel


def offset_path(path: np.ndarray, offset: float) -> np.ndarray:
    """Shift an Nx2 polyline sideways by `offset` metres (positive = left)."""
    if abs(offset) < 1e-6 or len(path) < 2:
        return path.copy()
    d = np.gradient(path, axis=0)
    n = np.stack([-d[:, 1], d[:, 0]], axis=1)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    return path + offset * n


def lookahead_point(path: np.ndarray, distance: float) -> np.ndarray:
    """First point of the path whose straight-line distance from the vehicle reaches `distance`."""
    pts = path[path[:, 0] > 0.5]
    if len(pts) == 0:
        return np.array([distance, 0.0])
    r = np.hypot(pts[:, 0], pts[:, 1])
    idx = np.nonzero(r >= distance)[0]
    if len(idx) == 0:
        return pts[-1]
    k = idx[0]
    if k == 0:
        return pts[0]
    r0, r1 = r[k - 1], r[k]
    t = (distance - r0) / max(r1 - r0, 1e-6)
    return pts[k - 1] + t * (pts[k] - pts[k - 1])


def path_curvature(path: np.ndarray, x_max: float = 25.0) -> float:
    """Max |curvature| of the path within x_max metres (for curve speed limiting)."""
    pts = path[(path[:, 0] > 0) & (path[:, 0] < x_max)]
    if len(pts) < 5:
        return 0.0
    c = np.polyfit(pts[:, 0], pts[:, 1], 2)
    return float(abs(2 * c[0]))


class PathPlanner:
    def __init__(self, cfg: dict):
        p = cfg["planning"]
        self.ld_min = p["lookahead_min"]
        self.ld_gain = p["lookahead_gain"]
        self.offset_rate = 1.8                 # m/s of lateral shift during a lane change
        self.current_offset = 0.0

    def reference_path(self, world: WorldModel, decision: Decision, dt: float) -> np.ndarray:
        step = self.offset_rate * dt
        self.current_offset += float(np.clip(decision.lateral_offset - self.current_offset, -step, step))
        lane = world.lane
        if decision.use_route or lane is None or abs(self.current_offset) > 0.05:
            return offset_path(world.route, self.current_offset)
        xs = np.arange(0.0, 30.0, 1.0)
        return np.stack([xs, lane.y_at(xs)], axis=1)

    def target(self, path: np.ndarray, speed: float, tight: bool = False) -> tuple[np.ndarray, float]:
        """`tight` shortens the look-ahead so pure pursuit does not cut corners in sharp turns."""
        ld = self.ld_min + self.ld_gain * speed
        if tight:
            ld = max(3.5, 0.6 * ld)
        return lookahead_point(path, ld), ld
