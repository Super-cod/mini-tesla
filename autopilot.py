"""The full driving stack for one tick: perception -> fusion -> decision -> planning -> control."""
from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from control.speed_controller import SpeedController
from control.steering_pid import SteeringController
from fusion.camera_fusion import CameraFusion
from perception.lane_detection import LaneDetector
from perception.object_detection import make_detector
from planning.decision_engine import DecisionEngine
from planning.path_planner import PathPlanner
from utils.geometry import make_stereo_pair
from utils.types import Control, Decision, Frame, WorldModel


@dataclass
class TickResult:
    control: Control
    world: WorldModel
    decision: Decision
    path: np.ndarray
    target: np.ndarray
    detections: tuple
    lane_mask: np.ndarray
    perception_ms: float


class Autopilot:
    def __init__(self, cfg: dict, backend: str = "sim"):
        self.cfg = cfg
        self.dt = cfg["dt"]
        self.cam_left, self.cam_right = make_stereo_pair(cfg["cameras"])
        self.lanes = LaneDetector(cfg)
        self.detector, self.detector_kind = make_detector(cfg, backend)
        self.fusion = CameraFusion(cfg, self.cam_left, self.cam_right)
        self.decisions = DecisionEngine(cfg)
        self.planner = PathPlanner(cfg)
        v = cfg["vehicle"]
        self.steering = SteeringController(cfg["control"]["steering"], v["wheelbase"], v["max_steer_deg"])
        self.speed = SpeedController(cfg["control"]["speed"], cfg["control"]["max_brake_decel"])
        self._mode = None

    def tick(self, frame: Frame) -> TickResult:
        t0 = time.perf_counter()
        lane_l, mask = self.lanes.detect(frame.left, self.cam_left)
        lane_r, _ = self.lanes.detect(frame.right, self.cam_right)
        dets_l, dets_r = self.detector.detect_pair(frame, (self.cam_left, self.cam_right))
        world = self.fusion.fuse(frame, lane_l, lane_r, dets_l, dets_r)
        perception_ms = (time.perf_counter() - t0) * 1000

        self.decisions.current_offset = self.planner.current_offset
        decision = self.decisions.decide(world, frame.route_finished)
        mode = (decision.use_route, world.at_junction)
        if mode != self._mode:                       # new path source: drop accumulated steering bias
            self.steering.reset()
            self._mode = mode
        path = self.planner.reference_path(world, decision, self.dt)
        target, _ = self.planner.target(path, world.speed, tight=world.at_junction)
        steer = self.steering.step(target, self.dt)
        control = self.speed.control(decision.target_speed, world.speed, steer, self.dt, decision.hard_brake)
        return TickResult(control, world, decision, path, target, (dets_l, dets_r), mask, perception_ms)


# ------------------------------------------------------------------------- visualisation
def draw_overlay(frame: Frame, res: TickResult, cam) -> np.ndarray:
    img = frame.left.copy()
    # Planned path and look-ahead target projected onto the left image.
    pts = np.hstack([res.path, np.zeros((len(res.path), 1))])
    uv, depth = cam.project(pts)
    uv = uv[depth > 1.0].astype(np.int32)
    if len(uv) > 1:
        cv2.polylines(img, [uv], False, (0, 255, 255), 2, cv2.LINE_AA)
    tuv, td = cam.project(np.array([[res.target[0], res.target[1], 0.0]]))
    if td[0] > 0:
        cv2.circle(img, tuple(int(v) for v in tuv[0]), 6, (0, 0, 255), -1)
    # Lane marking mask tinted green.
    m = res.lane_mask > 0.5
    img[m] = (0.5 * img[m] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
    # Objects with fused distances.
    for o in res.world.objects:
        if o.bbox is None:
            continue
        x1, y1, x2, y2 = (int(v) for v in o.bbox)
        color = (0, 0, 255) if o.cls == "pedestrian" else (255, 128, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{o.cls} {o.x:.1f}m", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    c, d = res.control, res.decision
    lines = [f"{d.action.value}  {d.reason}",
             f"speed {frame.speed * 3.6:5.1f} km/h  target {d.target_speed * 3.6:5.1f}",
             f"steer {c.steer:+.2f}  throttle {c.throttle:.2f}  brake {c.brake:.2f}",
             f"lane {'--' if res.world.lane is None else f'{res.world.lane.offset:+.2f} m  conf {res.world.lane.confidence:.1f}'}"]
    cv2.rectangle(img, (0, 0), (img.shape[1], 18 * len(lines) + 8), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (8, 18 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    right = cv2.resize(frame.right, (img.shape[1] // 3, img.shape[0] // 3))
    img[-right.shape[0]:, -right.shape[1]:] = right
    return img
