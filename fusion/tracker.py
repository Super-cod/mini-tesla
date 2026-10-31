"""Multi-object tracking with one constant-velocity Kalman filter per object.

State per track: [x, y, vx, vy] in the *current* vehicle frame, where (vx, vy) is the object's
velocity over the ground. Every tick:
  1. ego-motion compensation - the vehicle frame moved (speed * dt forward, yaw_rate * dt turn),
     so every track is shifted/rotated into the new frame;
  2. prediction - constant-velocity motion over dt;
  3. association - gated nearest neighbour (Mahalanobis distance) with the new measurements
     from all cameras;
  4. update - Kalman update; unmatched measurements start new tracks; tracks that are not seen
     for a while (e.g. after leaving every camera's view) are dropped.
Each output object carries its estimated velocity and a 3 s predicted trajectory.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from utils.types import TrackedObject

ACCEL_NOISE = {"pedestrian": 1.2, "cyclist": 2.0, "car": 2.0}
CHI2_GATE = 13.8                       # 99.9 % for 2 degrees of freedom


@dataclass
class Track:
    id: int
    cls: str
    x: np.ndarray                      # [x, y, vx, vy]
    P: np.ndarray
    last: TrackedObject
    hits: int = 1
    age: float = 0.0
    unseen: float = 0.0
    history: list = field(default_factory=list)


def predict_trajectory(x: float, y: float, vx: float, vy: float, horizon: float = 3.0,
                       step: float = 0.25) -> np.ndarray:
    """Constant-velocity forecast: Kx3 array of (t, x, y)."""
    t = np.arange(step, horizon + 1e-9, step)
    return np.stack([t, x + vx * t, y + vy * t], axis=1)


class MultiObjectTracker:
    def __init__(self, dt: float, max_unseen: float = 1.0, confirm_hits: int = 3,
                 horizon: float = 3.0):
        self.dt = dt
        self.max_unseen = max_unseen
        self.confirm_hits = confirm_hits
        self.horizon = horizon
        self.tracks: list[Track] = []
        self._ids = itertools.count(1)

    # ------------------------------------------------------------------ filter steps
    def _compensate_ego_motion(self, speed: float, yaw_rate: float):
        dt = self.dt
        dth = yaw_rate * dt
        c, s = np.cos(-dth), np.sin(-dth)
        rot = np.array([[c, -s], [s, c]])
        T = np.zeros((4, 4))
        T[:2, :2] = rot
        T[2:, 2:] = rot
        shift = np.array([speed * dt, 0.0])
        for tr in self.tracks:
            tr.x[:2] = rot @ (tr.x[:2] - shift)
            tr.x[2:] = rot @ tr.x[2:]
            tr.P = T @ tr.P @ T.T

    def _predict(self):
        dt = self.dt
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        for tr in self.tracks:
            q = ACCEL_NOISE.get(tr.cls, 1.5) ** 2
            G = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
            tr.x = F @ tr.x
            tr.P = F @ tr.P @ F.T + q * G @ G.T

    @staticmethod
    def _innovation(tr: Track, z: np.ndarray, R: np.ndarray):
        y = z - tr.x[:2]
        S = tr.P[:2, :2] + R
        return y, S

    def step(self, measurements, speed: float, yaw_rate: float) -> list[TrackedObject]:
        """measurements: list of fusion.camera_fusion.Measurement. Returns confirmed tracks."""
        self._compensate_ego_motion(speed, yaw_rate)
        self._predict()

        # Gated greedy association, cheapest pairs first.
        pairs = []
        for ti, tr in enumerate(self.tracks):
            for mi, m in enumerate(measurements):
                if m.obj.cls != tr.cls:
                    continue
                y, S = self._innovation(tr, np.array([m.obj.x, m.obj.y]), m.R)
                d2 = float(y @ np.linalg.solve(S, y))
                if d2 < CHI2_GATE:
                    pairs.append((d2, ti, mi))
        used_t, used_m = set(), set()
        for _, ti, mi in sorted(pairs):
            if ti in used_t or mi in used_m:
                continue
            used_t.add(ti)
            used_m.add(mi)
            tr, m = self.tracks[ti], measurements[mi]
            z = np.array([m.obj.x, m.obj.y])
            y, S = self._innovation(tr, z, m.R)
            K = tr.P[:, :2] @ np.linalg.inv(S)
            tr.x = tr.x + K @ y
            tr.P = (np.eye(4) - K[:, :2] @ np.eye(2, 4)) @ tr.P
            tr.last, tr.hits, tr.unseen = m.obj, tr.hits + 1, 0.0

        for ti, tr in enumerate(self.tracks):
            tr.age += self.dt
            if ti not in used_t:
                tr.unseen += self.dt
                tr.last = TrackedObject(**{**tr.last.__dict__, "bbox": None})   # not visible this frame
        for mi, m in enumerate(measurements):
            if mi in used_m:
                continue
            P = np.zeros((4, 4))
            P[:2, :2] = m.R
            P[2:, 2:] = np.eye(2) * 6.0 ** 2          # unknown velocity: +-6 m/s
            self.tracks.append(Track(next(self._ids), m.obj.cls, np.array([m.obj.x, m.obj.y, 0.0, 0.0]), P, m.obj))
        self.tracks = [t for t in self.tracks if t.unseen <= self.max_unseen
                       and not (t.hits < self.confirm_hits and t.unseen > 0.15)]
        return self.objects()

    def objects(self) -> list[TrackedObject]:
        out = []
        for tr in self.tracks:
            if tr.hits < self.confirm_hits:
                continue
            x, y, vx, vy = (float(v) for v in tr.x)
            v_std = float(np.sqrt(max(tr.P[2, 2], tr.P[3, 3])))
            out.append(TrackedObject(
                tr.cls, x, y, float(np.hypot(x, y)), tr.last.conf, tr.last.source, tr.last.bbox,
                camera=tr.last.camera, track_id=tr.id, vx=vx, vy=vy,
                velocity_valid=tr.age >= 0.6 and v_std < 1.2,
                predicted=predict_trajectory(x, y, vx, vy, self.horizon), age=tr.age))
        return out
