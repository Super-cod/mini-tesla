"""The full driving stack for one tick:
perception (4 cameras) -> 3D fusion + tracking + prediction -> decision -> planning -> control."""
from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from control.mpc import MPCSteering
from control.speed_controller import SpeedController
from control.steering_pid import SteeringController
from fusion.camera_fusion import CameraFusion
from perception.lane_detection import LaneDetector
from perception.object_detection import make_detector
from planning.decision_engine import DecisionEngine
from planning.path_planner import PathPlanner
from utils.geometry import make_rig
from utils.types import Control, Decision, Frame, WorldModel


@dataclass
class TickResult:
    control: Control
    world: WorldModel
    decision: Decision
    path: np.ndarray
    target: np.ndarray
    detections: dict
    lane_mask: np.ndarray
    perception_ms: float
    mpc_plan: np.ndarray | None = None


class Autopilot:
    def __init__(self, cfg: dict, backend: str = "sim"):
        self.cfg = cfg
        self.dt = cfg["dt"]
        self.cams = make_rig(cfg["cameras"])
        left, right = cfg["cameras"]["stereo_pair"]
        self.cam_left, self.cam_right = self.cams[left], self.cams[right]
        self.lanes = LaneDetector(cfg)
        self.detector, self.detector_kind = make_detector(cfg, backend)
        self.fusion = CameraFusion(cfg, self.cams)
        self.decisions = DecisionEngine(cfg)
        self.planner = PathPlanner(cfg)
        v = cfg["vehicle"]
        c = cfg["control"]
        self.lateral = c.get("lateral", "pid")
        self.steering = SteeringController(c["steering"], v["wheelbase"], v["max_steer_deg"])
        self.mpc = MPCSteering(v["wheelbase"], v["max_steer_deg"], c.get("mpc"))
        self.speed = SpeedController(c["speed"], c["max_brake_decel"])
        self._mode = None

    def set_max_steer(self, max_steer_deg: float):
        self.steering.set_max_steer(max_steer_deg)
        self.mpc.set_max_steer(max_steer_deg)

    def tick(self, frame: Frame) -> TickResult:
        t0 = time.perf_counter()
        lane_l, mask = self.lanes.detect(frame.left, self.cam_left)
        lane_r, _ = self.lanes.detect(frame.right, self.cam_right)
        dets = self.detector.detect_all(frame, self.cams)
        world = self.fusion.fuse(frame, lane_l, lane_r, dets)
        perception_ms = (time.perf_counter() - t0) * 1000

        self.decisions.current_offset = self.planner.current_offset
        decision = self.decisions.decide(world, frame.route_finished)
        mode = (decision.use_route, world.at_junction)
        if mode != self._mode:                       # new path source: drop accumulated steering bias
            self.steering.reset()
            self._mode = mode
        path = self.planner.reference_path(world, decision, self.dt)
        target, _ = self.planner.target(path, world.speed, tight=world.at_junction)
        if self.lateral == "mpc":
            steer = self.mpc.step(path, world.speed, self.dt)
            plan = self.mpc.last_best
        else:
            steer = self.steering.step(target, self.dt)
            plan = None
        control = self.speed.control(decision.target_speed, world.speed, steer, self.dt, decision.hard_brake)
        return TickResult(control, world, decision, path, target, dets, mask, perception_ms, plan)


# ------------------------------------------------------------------------- visualisation
CLASS_COLORS = {"pedestrian": (60, 60, 255), "car": (255, 160, 0), "cyclist": (0, 200, 255), "stop_sign": (0, 0, 200)}


def draw_front_view(frame: Frame, res: TickResult, cam) -> np.ndarray:
    img = frame.left.copy()
    # Lane marking mask tinted green.
    m = res.lane_mask > 0.5
    img[m] = (0.5 * img[m] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
    # Planned path, MPC's predicted trajectory and the look-ahead target projected onto the image.
    for pts, color in ((res.path, (0, 255, 255)), (res.mpc_plan, (255, 0, 255))):
        if pts is None or len(pts) < 2:
            continue
        uv, depth = cam.project(np.hstack([pts, np.zeros((len(pts), 1))]))
        uv = uv[depth > 1.0].astype(np.int32)
        if len(uv) > 1:
            cv2.polylines(img, [uv], False, color, 2, cv2.LINE_AA)
    tuv, td = cam.project(np.array([[res.target[0], res.target[1], 0.0]]))
    if td[0] > 0:
        cv2.circle(img, tuple(int(v) for v in tuv[0]), 5, (0, 0, 255), -1)
    for o in res.world.objects:
        if o.bbox is None or o.camera != cam.name:
            continue
        x1, y1, x2, y2 = (int(v) for v in o.bbox)
        color = CLASS_COLORS.get(o.cls, (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"#{o.track_id} {o.cls} {o.x:.1f}m" if o.track_id else f"{o.cls} {o.x:.1f}m"
        if o.velocity_valid:
            label += f" {o.speed:.1f}m/s"
        cv2.putText(img, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    c, d = res.control, res.decision
    lines = [f"{d.action.value}  {d.reason}",
             f"speed {frame.speed * 3.6:5.1f} km/h  target {d.target_speed * 3.6:5.1f}",
             f"steer {c.steer:+.2f}  throttle {c.throttle:.2f}  brake {c.brake:.2f}",
             f"lane {'--' if res.world.lane is None else f'{res.world.lane.offset:+.2f} m  conf {res.world.lane.confidence:.1f}'}"]
    cv2.rectangle(img, (0, 0), (img.shape[1], 18 * len(lines) + 8), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (8, 18 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def draw_bev(res: TickResult, cams: dict, size=(330, 360), scale: float = 6.0, ego_row: int = 250,
             vehicle: dict | None = None) -> np.ndarray:
    """Bird's-eye view of the 3D world model: camera coverage, lane, route, tracked objects,
    their velocities and 3 s predictions, and the MPC plan."""
    w, h = size
    img = np.full((h, w, 3), 28, np.uint8)
    cx = w // 2

    def px(x, y):
        return int(round(cx - y * scale)), int(round(ego_row - x * scale))

    def poly(pts):
        return np.array([px(x, y) for x, y in pts], np.int32)

    for m in range(-40, 70, 10):                                       # 10 m grid
        r = ego_row - int(m * scale)
        if 0 <= r < h:
            cv2.line(img, (0, r), (w, r), (45, 45, 45), 1)
    # Camera fields of view.
    overlay = img.copy()
    for cam in cams.values():
        half = np.radians(cam.fov_deg / 2)
        yaw = np.radians(cam.yaw_deg)
        rng = 45.0
        pts = [cam.position[:2]] + [cam.position[:2] + rng * np.array([np.cos(a), np.sin(a)])
                                    for a in np.linspace(yaw - half, yaw + half, 12)]
        cv2.fillPoly(overlay, [poly(pts)], (70, 60, 40) if cam.yaw_deg == 0 else (40, 60, 70))
    img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)
    # Route, lane boundaries, planned path, MPC plan.
    if len(res.world.route) > 1:
        cv2.polylines(img, [poly(res.world.route)], False, (110, 110, 110), 1, cv2.LINE_AA)
    lane = res.world.lane
    if lane is not None:
        xs = np.arange(0, 30, 1.0)
        half_w = (lane.width or res.world.lane_width) / 2
        for side in (-half_w, half_w):
            cv2.polylines(img, [poly(np.stack([xs, lane.y_at(xs) + side], 1))], False, (0, 200, 0), 2, cv2.LINE_AA)
    cv2.polylines(img, [poly(res.path)], False, (0, 220, 220), 1, cv2.LINE_AA)
    if res.mpc_plan is not None:
        cv2.polylines(img, [poly(res.mpc_plan)], False, (255, 0, 255), 2, cv2.LINE_AA)
    # Tracked objects with velocity and predicted trajectory.
    for o in res.world.objects:
        color = CLASS_COLORS.get(o.cls, (255, 255, 255))
        if o.predicted is not None and o.velocity_valid and o.speed > 0.3:
            for t, x, y in o.predicted[::2]:
                cv2.circle(img, px(x, y), 2, tuple(int(c * (1 - t / 4)) for c in color), -1)
        if o.cls == "pedestrian":
            cv2.circle(img, px(o.x, o.y), 4, color, -1)
        else:
            L, W = 2 * 2.25, 1.9
            cxo = o.x + (2.25 if o.x > 2 else -2.25 if o.x < -2 else 0.0)  # x is the near face
            cv2.rectangle(img, px(cxo + L / 2, o.y + W / 2), px(cxo - L / 2, o.y - W / 2), color, -1 if o.bbox is not None else 1)
        if o.velocity_valid and o.speed > 0.3:
            cv2.arrowedLine(img, px(o.x, o.y), px(o.x + o.vx, o.y + o.vy), (255, 255, 255), 1, tipLength=0.3)
        if o.track_id is not None:
            cv2.putText(img, f"#{o.track_id}", (px(o.x, o.y)[0] + 6, px(o.x, o.y)[1]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, (230, 230, 230), 1, cv2.LINE_AA)
    # Ego car (rear axle at the origin).
    veh = vehicle or {"wheelbase": 2.9, "length": 4.7, "width": 1.9}
    xc = veh["wheelbase"] / 2
    cv2.rectangle(img, px(xc + veh["length"] / 2, veh["width"] / 2), px(xc - veh["length"] / 2, -veh["width"] / 2),
                  (255, 255, 255), -1)
    cv2.putText(img, "3D world model (top view)", (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(img, "o ped  [] car  ... 3 s prediction", (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
    return img


def draw_overlay(frame: Frame, res: TickResult, cams: dict, vehicle: dict | None = None) -> np.ndarray:
    """Front view + bird's-eye 3D view on top, all camera thumbnails underneath."""
    stereo_left = cams[frame.stereo_pair[0]]
    front = draw_front_view(frame, res, stereo_left)
    bev = draw_bev(res, cams, size=(330, front.shape[0]), vehicle=vehicle)
    top = np.hstack([front, bev])
    names = list(frame.images)
    tw = top.shape[1] // len(names)
    th = int(round(tw * front.shape[0] / front.shape[1]))
    thumbs = []
    for n in names:
        t = cv2.resize(frame.images[n], (tw, th))
        for o in res.world.objects:
            if o.camera == n and o.bbox is not None:
                sx, sy = tw / frame.images[n].shape[1], th / frame.images[n].shape[0]
                x1, y1, x2, y2 = o.bbox
                cv2.rectangle(t, (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy)),
                              CLASS_COLORS.get(o.cls, (255, 255, 255)), 1)
        cv2.putText(t, n, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(t, n, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        thumbs.append(t)
    bottom = np.hstack(thumbs)
    if bottom.shape[1] < top.shape[1]:
        bottom = np.hstack([bottom, np.zeros((th, top.shape[1] - bottom.shape[1], 3), np.uint8)])
    return np.vstack([top, bottom])
