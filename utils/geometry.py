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
    yaw_deg: float = 0.0          # positive = turned to the left (90 = looking out of the left side)
    name: str = ""

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=np.float64)
        self.fx = self.width / (2.0 * math.tan(math.radians(self.fov_deg) / 2.0))
        self.fy = self.fx
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0
        p, y = math.radians(self.pitch_deg), math.radians(self.yaw_deg)
        # Camera axes expressed in the vehicle frame (yaw first, then pitch, no roll).
        self.forward = np.array([math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p)])
        self.left = np.array([-math.sin(y), math.cos(y), 0.0])
        self.up = np.array([-math.sin(p) * math.cos(y), -math.sin(p) * math.sin(y), math.cos(p)])
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


def make_camera(name: str, cam_cfg: dict) -> PinholeCamera:
    c = cam_cfg["rig"][name]
    return PinholeCamera(width=cam_cfg["width"], height=cam_cfg["height"], fov_deg=c["fov"],
                         position=np.array([c["x"], c["y"], c["z"]]), pitch_deg=c.get("pitch", 0.0),
                         yaw_deg=c.get("yaw", 0.0), name=name)


def make_rig(cam_cfg: dict) -> dict[str, PinholeCamera]:
    """All cameras of the rig, keyed by name (the stereo pair first)."""
    names = list(cam_cfg["stereo_pair"]) + [n for n in cam_cfg["rig"] if n not in cam_cfg["stereo_pair"]]
    return {n: make_camera(n, cam_cfg) for n in names}


def make_stereo_pair(cam_cfg: dict) -> tuple[PinholeCamera, PinholeCamera]:
    left, right = cam_cfg["stereo_pair"]
    return make_camera(left, cam_cfg), make_camera(right, cam_cfg)


def stereo_focal_baseline(cam_cfg: dict) -> float:
    """fx * B of the front stereo pair: depth = fx * B / disparity."""
    left, right = make_stereo_pair(cam_cfg)
    return left.fx * float(np.linalg.norm(left.position - right.position))


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def world_to_vehicle(points_xy: np.ndarray, ego_xy: np.ndarray, ego_yaw: float) -> np.ndarray:
    """Nx2 world points -> vehicle frame (x fwd, y left). Both frames right-handed."""
    d = np.asarray(points_xy, dtype=np.float64) - np.asarray(ego_xy, dtype=np.float64)
    c, s = math.cos(ego_yaw), math.sin(ego_yaw)
    return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], axis=1)
