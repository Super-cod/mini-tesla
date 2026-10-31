"""Pinhole camera model shared by the simulators and the perception stack.

Vehicle frame: x forward, y left, z up. Camera optical frame: X right, Y down, Z forward.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class PinholeCamera:
    width: int
    height: int
    fov_deg: float
    position: np.ndarray          # (x, y, z) mount position in the vehicle frame
    pitch_deg: float = 0.0        # negative = looking down

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=np.float64)
        self.fx = self.width / (2.0 * math.tan(math.radians(self.fov_deg) / 2.0))
        self.fy = self.fx
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0
        p = math.radians(self.pitch_deg)
        # Camera axes expressed in the vehicle frame.
        self.forward = np.array([math.cos(p), 0.0, math.sin(p)])
        self.left = np.array([0.0, 1.0, 0.0])
        self.up = np.array([-math.sin(p), 0.0, math.cos(p)])
        # Rows map vehicle-frame offsets to optical (X right, Y down, Z forward).
        self.R = np.stack([-self.left, -self.up, self.forward])

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    def to_camera(self, pts: np.ndarray) -> np.ndarray:
        """Nx3 vehicle-frame points -> Nx3 optical-frame points."""
        return (np.asarray(pts, dtype=np.float64) - self.position) @ self.R.T

    def project(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (Nx2 pixel coords, N depths). Points behind the camera get depth <= 0."""
        c = self.to_camera(np.atleast_2d(pts))
        z = c[:, 2]
        safe = np.where(np.abs(z) < 1e-6, 1e-6, z)
        u = self.cx + self.fx * c[:, 0] / safe
        v = self.cy + self.fy * c[:, 1] / safe
        return np.stack([u, v], axis=1), z

    def ray(self, uv: np.ndarray) -> np.ndarray:
        """Unnormalised viewing ray directions in the vehicle frame for Nx2 pixels."""
        uv = np.atleast_2d(np.asarray(uv, dtype=np.float64))
        xn = (uv[:, 0] - self.cx) / self.fx
        yn = (uv[:, 1] - self.cy) / self.fy
        return self.forward[None, :] + xn[:, None] * (-self.left)[None, :] + yn[:, None] * (-self.up)[None, :]

    def pixel_to_ground(self, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Inverse perspective mapping onto the z=0 road plane.

        Returns (Nx2 ground points (x, y), boolean mask of pixels whose ray hits the ground ahead).
        """
        d = self.ray(uv)
        valid = d[:, 2] < -1e-6
        t = np.where(valid, -self.position[2] / np.where(valid, d[:, 2], -1.0), 0.0)
        pts = self.position[None, :] + t[:, None] * d
        return pts[:, :2], valid

    def pixel_depth_to_vehicle(self, u: float, v: float, depth: float) -> np.ndarray:
        """Back-project a pixel with known optical depth Z into the vehicle frame."""
        d = self.ray(np.array([[u, v]]))[0]
        return self.position + depth * d


def make_stereo_pair(cam_cfg: dict) -> tuple[PinholeCamera, PinholeCamera]:
    half = cam_cfg["baseline"] / 2.0
    common = dict(width=cam_cfg["width"], height=cam_cfg["height"], fov_deg=cam_cfg["fov"],
                  pitch_deg=cam_cfg["pitch_deg"])
    left = PinholeCamera(position=np.array([cam_cfg["mount_x"], +half, cam_cfg["mount_z"]]), **common)
    right = PinholeCamera(position=np.array([cam_cfg["mount_x"], -half, cam_cfg["mount_z"]]), **common)
    return left, right


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def world_to_vehicle(points_xy: np.ndarray, ego_xy: np.ndarray, ego_yaw: float) -> np.ndarray:
    """Nx2 world points -> vehicle frame (x fwd, y left). Both frames right-handed."""
    d = np.asarray(points_xy, dtype=np.float64) - np.asarray(ego_xy, dtype=np.float64)
    c, s = math.cos(ego_yaw), math.sin(ego_yaw)
    return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], axis=1)
