"""Data containers passed between the layers of the stack."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


@dataclass
class Frame:
    """One synchronised sensor snapshot from a simulator backend."""
    left: np.ndarray                 # HxWx3 BGR image from the front-left camera
    right: np.ndarray                # HxWx3 BGR image from the front-right camera
    speed: float                     # m/s
    timestamp: float
    route: np.ndarray                # Nx2 route centreline ahead, vehicle frame (x fwd, y left)
    at_junction: bool = False        # vehicle is inside / about to enter an intersection
    lane_width: float = 3.5
    left_lane_available: bool = True
    ground_truth: list = field(default_factory=list)   # GroundTruthObject list (sim / CARLA debug only)
    route_finished: bool = False


@dataclass
class GroundTruthObject:
    cls: str                         # "car" | "pedestrian"
    x: float                         # centre in vehicle frame
    y: float
    length: float
    width: float
    height: float
    yaw: float = 0.0                 # relative to the ego heading


@dataclass
class Detection:
    cls: str                         # car | pedestrian | cyclist | stop_sign | traffic_light
    bbox: tuple[float, float, float, float]   # x1, y1, x2, y2 in pixels
    conf: float = 1.0

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0


@dataclass
class LaneEstimate:
    """Lane centre as y = c0 + c1*x + c2*x^2 in the vehicle frame."""
    coeffs: np.ndarray               # (c0, c1, c2)
    confidence: float                # 0..1
    left_coeffs: np.ndarray | None = None
    right_coeffs: np.ndarray | None = None
    width: float | None = None

    def y_at(self, x: float | np.ndarray) -> float | np.ndarray:
        c0, c1, c2 = self.coeffs
        return c0 + c1 * x + c2 * x * x

    @property
    def offset(self) -> float:
        """Lateral distance from the vehicle to the lane centre (positive = centre is to the left)."""
        return float(self.coeffs[0])

    @property
    def heading(self) -> float:
        return float(np.arctan(self.coeffs[1]))


@dataclass
class TrackedObject:
    cls: str
    x: float                         # metres ahead
    y: float                         # metres to the left
    distance: float
    conf: float
    source: str                      # "stereo" | "mono"
    bbox: tuple[float, float, float, float] | None = None


@dataclass
class WorldModel:
    lane: LaneEstimate | None
    objects: list[TrackedObject]
    speed: float
    route: np.ndarray
    at_junction: bool
    lane_width: float
    left_lane_available: bool = True


class Action(str, Enum):
    LANE_FOLLOW = "LANE_FOLLOW"
    FOLLOW_VEHICLE = "FOLLOW_VEHICLE"
    AVOID_OBSTACLE = "AVOID_OBSTACLE"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    STOP_SIGN = "STOP_SIGN"
    TURN = "TURN"
    SLOW_DOWN = "SLOW_DOWN"
    ROUTE_DONE = "ROUTE_DONE"


@dataclass
class Decision:
    action: Action
    target_speed: float
    lateral_offset: float = 0.0      # metres to shift the path left (lane change)
    use_route: bool = False          # steer by map route instead of perceived lane
    hard_brake: bool = False
    reason: str = ""


@dataclass
class Control:
    """Normalised command, CARLA convention: steer in [-1, 1] with +1 = full right."""
    throttle: float = 0.0
    steer: float = 0.0
    brake: float = 0.0
