"""Built-in lightweight driving simulator.

It exists so the whole stack (two cameras -> lane CNN -> stereo -> decisions -> PID) can be
run and tested on any laptop without CARLA. It renders real perspective images from the same
pinhole model the perception code uses, so perception has to actually work for the car to
stay on the road. Objects are drawn as simple cuboids; their image boxes come from
`ground_truth` (the same idea as CARLA's ground-truth bounding boxes), because YOLO is not
trained on these cartoon renders.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from simulation.base import Simulator
from utils.geometry import PinholeCamera, make_rig, world_to_vehicle
from utils.types import Control, Frame, GroundTruthObject

SKY = np.array([235, 206, 150], dtype=np.float32)
GRASS = np.array([70, 120, 70], dtype=np.float32)
ASPHALT = np.array([85, 85, 85], dtype=np.float32)
PAINT = (235, 235, 235)
CAR_COLORS = [(180, 60, 40), (40, 40, 170), (30, 140, 200), (120, 120, 120), (20, 20, 20)]
PED_COLOR = (60, 90, 210)
NEAR = 0.6       # metres; geometry closer than this to a camera is not drawn


# --------------------------------------------------------------------------- road geometry
@dataclass
class Segment:
    length: float
    curvature: float = 0.0          # 1/radius, positive = left turn
    junction: bool = False


class Road:
    """Route centreline integrated from constant-curvature segments. The ego lane is centred on
    the route; a second same-direction lane lies to its left (used for overtaking)."""

    def __init__(self, segments: list[Segment], lane_width: float = 3.5, ds: float = 0.25):
        self.ds = ds
        self.lane_width = lane_width
        xs, ys, yaws, junc = [0.0], [0.0], [0.0], [segments[0].junction]
        x = y = yaw = 0.0
        for seg in segments:
            for _ in range(max(1, int(round(seg.length / ds)))):
                yaw += seg.curvature * ds / 2
                x += math.cos(yaw) * ds
                y += math.sin(yaw) * ds
                yaw += seg.curvature * ds / 2
                xs.append(x); ys.append(y); yaws.append(yaw); junc.append(seg.junction)
        self.x, self.y, self.yaw = np.array(xs), np.array(ys), np.array(yaws)
        self.junction = np.array(junc)
        self.length = (len(xs) - 1) * ds

    def _idx(self, s):
        return np.clip(np.asarray(s) / self.ds, 0, len(self.x) - 1.001)

    def pose(self, s):
        i = self._idx(s)
        i0 = np.floor(i).astype(int)
        t = i - i0
        x = self.x[i0] * (1 - t) + self.x[i0 + 1] * t
        y = self.y[i0] * (1 - t) + self.y[i0 + 1] * t
        yaw = self.yaw[i0] * (1 - t) + self.yaw[i0 + 1] * t
        return x, y, yaw

    def point(self, s, d=0.0):
        """World xy of the point at arc length s, lateral offset d (left positive)."""
        x, y, yaw = self.pose(s)
        return np.stack([x - np.sin(yaw) * d, y + np.cos(yaw) * d], axis=-1)

    def is_junction(self, s0: float, s1: float) -> bool:
        a, b = int(self._idx(s0)), int(math.ceil(self._idx(s1)))
        return bool(self.junction[a:b + 1].any())

    def project(self, xy, hint: float, window: float = 30.0) -> tuple[float, float]:
        """Arc length and signed lateral offset of a world point (searched near `hint`)."""
        a = int(self._idx(max(0.0, hint - window)))
        b = int(self._idx(hint + window)) + 2
        dx, dy = self.x[a:b] - xy[0], self.y[a:b] - xy[1]
        k = int(np.argmin(dx * dx + dy * dy))
        i = a + k
        yaw = self.yaw[i]
        rel = np.array([xy[0] - self.x[i], xy[1] - self.y[i]])
        along = rel[0] * math.cos(yaw) + rel[1] * math.sin(yaw)
        lateral = -rel[0] * math.sin(yaw) + rel[1] * math.cos(yaw)
        return i * self.ds + along, lateral


# --------------------------------------------------------------------------- actors
@dataclass
class Actor:
    cls: str
    s: float
    d: float
    length: float
    width: float
    height: float
    speed: float = 0.0               # along the road
    lateral_speed: float = 0.0       # pedestrians crossing
    d_end: float | None = None
    trigger_s: float | None = None   # starts moving once ego arc length passes this
    color: tuple = (40, 40, 170)
    spawn_on_trigger: bool = False   # invisible (not in the world at all) until triggered
    active: bool = field(default=False)

    @property
    def exists(self) -> bool:
        return self.active or not self.spawn_on_trigger

    def update(self, dt: float, ego_s: float):
        if not self.active and (self.trigger_s is None or ego_s >= self.trigger_s):
            self.active = True
        if not self.active:
            return
        self.s += self.speed * dt
        if self.d_end is not None and self.lateral_speed:
            step = math.copysign(self.lateral_speed * dt, self.d_end - self.d)
            self.d = self.d_end if abs(self.d_end - self.d) <= abs(step) else self.d + step


def car(s, d=0.0, speed=0.0, color=None, trigger_s=None, spawn_on_trigger=False):
    return Actor("car", s, d, 4.5, 1.9, 1.5, speed=speed, trigger_s=trigger_s,
                 color=color or CAR_COLORS[int(s) % len(CAR_COLORS)], spawn_on_trigger=spawn_on_trigger)


def pedestrian(s, d_start, d_end, trigger_s, speed=1.3):
    return Actor("pedestrian", s, d_start, 0.5, 0.6, 1.75, lateral_speed=speed, d_end=d_end,
                 trigger_s=trigger_s, color=PED_COLOR)


SCENARIO_NAMES = ("lane_keep", "curve", "pedestrian", "obstacle", "follow", "turn", "blind_spot", "full")


def build_scenario(name: str, lane_width: float = 3.5) -> tuple[Road, list[Actor]]:
    w = lane_width
    turn = Segment(math.pi / 2 * 12.0, 1 / 12.0, junction=True)
    if name == "lane_keep":
        segs = [Segment(60), Segment(60, 1 / 120), Segment(60), Segment(60, -1 / 100), Segment(60)]
        actors = []
    elif name == "curve":
        segs = [Segment(30), Segment(70, 1 / 50), Segment(20), Segment(70, -1 / 45), Segment(40)]
        actors = []
    elif name == "pedestrian":
        segs = [Segment(160)]
        actors = [pedestrian(70, -w / 2 - 2.5, 1.5 * w + 2.5, trigger_s=40)]
    elif name == "obstacle":
        segs = [Segment(220)]
        actors = [car(90, 0.0)]
    elif name == "follow":
        segs = [Segment(100), Segment(60, 1 / 150), Segment(140)]
        actors = [car(25, 0.0, speed=4.0)]
    elif name == "turn":
        segs = [Segment(50), turn, Segment(70)]
        actors = []
    elif name == "blind_spot":
        # A parked car blocks our lane while a faster car comes up from behind in the left lane.
        # Only the rear-facing side cameras can see it when we want to pull out.
        segs = [Segment(260)]
        actors = [car(110, 0.0),
                  car(45, w, speed=12.5, trigger_s=85, spawn_on_trigger=True, color=(30, 160, 230))]
    elif name == "full":
        segs = [Segment(40), Segment(50, 1 / 90), Segment(60), Segment(90),
                Segment(50, -1 / 80), Segment(40), turn, Segment(70)]
        actors = [pedestrian(115, -w / 2 - 2.5, 1.5 * w + 2.5, trigger_s=85),
                  car(190, 0.0)]
    else:
        raise ValueError(f"unknown scenario {name!r}")
    return Road(segs, lane_width=w), actors


# --------------------------------------------------------------------------- rendering
def _cuboid_corners(cx, cy, yaw, length, width, height):
    c, s = math.cos(yaw), math.sin(yaw)
    pts = []
    for dx in (-length / 2, length / 2):
        for dy in (-width / 2, width / 2):
            for z in (0.0, height):
                pts.append((cx + c * dx - s * dy, cy + s * dx + c * dy, z))
    return np.array(pts)


def _to_px(uv):
    return np.round(np.clip(uv, -2e4, 2e4) * 16).astype(np.int32)


class Renderer:
    def __init__(self, road: Road, far: float = 75.0, behind: float = 45.0):
        self.road = road
        self.far = far
        self.behind = behind                 # road drawn behind the car, for rear-facing cameras
        self._seg_cache: dict = {}

    def _strip(self, img, cam, ego, s0, s1, d0, d1, color, dash=None):
        step = 0.5
        s = np.arange(s0, s1, step)
        if len(s) < 2:
            return
        if dash is not None:
            on = (np.mod(s, dash[0] + dash[1]) < dash[0])
        a = world_to_vehicle(self.road.point(s, d0), ego[:2], ego[2])
        b = world_to_vehicle(self.road.point(s, d1), ego[:2], ego[2])
        za = np.zeros((len(s), 1))
        ua, da = cam.project(np.hstack([a, za]))
        ub, db = cam.project(np.hstack([b, za]))
        ok = (da > NEAR) & (db > NEAR)
        pa, pb = _to_px(ua), _to_px(ub)
        # Cull quads that are behind the camera or entirely off-screen (vectorised).
        h16, w16 = cam.height * 16, cam.width * 16
        quads = np.stack([pa[:-1], pa[1:], pb[1:], pb[:-1]], axis=1)          # (N-1, 4, 2)
        lo, hi = quads.min(1), quads.max(1)
        keep = ok[:-1] & ok[1:] & (hi[:, 0] >= 0) & (lo[:, 0] < w16) & (hi[:, 1] >= 0) & (lo[:, 1] < h16)
        if dash is not None:
            keep &= on[:-1]
        for quad in quads[keep]:
            cv2.fillConvexPoly(img, quad, color, lineType=cv2.LINE_AA, shift=4)

    def _segments(self, s_from, s_to, want_junction):
        """Split [s_from, s_to] into runs where the junction flag equals want_junction."""
        key = (round(s_from, 3), round(s_to, 3), want_junction)
        if key in self._seg_cache:
            return self._seg_cache[key]
        if len(self._seg_cache) > 64:
            self._seg_cache.clear()
        out, start = [], None
        for s in np.arange(s_from, s_to, 1.0):
            j = self.road.is_junction(s, s + 1.0) if want_junction else not self.road.is_junction(s - 0.5, s + 1.5)
            if j and start is None:
                start = s
            elif not j and start is not None:
                out.append((start, s)); start = None
        if start is not None:
            out.append((start, s_to))
        self._seg_cache[key] = out
        return out

    def render(self, cam: PinholeCamera, ego: np.ndarray, ego_s: float, actors: list[Actor],
               rng: np.random.Generator | None = None, augment: bool = False,
               with_mask: bool = False):
        h, w = cam.height, cam.width
        rng = rng or np.random.default_rng(0)
        bright = rng.uniform(0.65, 1.3) if augment else 1.0
        asphalt = ASPHALT * (rng.uniform(0.7, 1.25) if augment else 1.0)

        img = np.empty((h, w, 3), dtype=np.float32)
        horizon = int(np.clip(cam.cy + cam.fy * math.tan(math.radians(cam.pitch_deg)), 0, h))
        rows = np.linspace(0, 1, max(horizon, 1))[:, None]
        img[:horizon] = (SKY * (0.85 + 0.15 * rows))[:, None, :]
        img[horizon:] = GRASS
        img = img.astype(np.uint8)
        mask = np.zeros((h, w), np.uint8) if with_mask else None

        lw = self.road.lane_width
        s0, s1 = max(0.0, ego_s - self.behind), min(self.road.length, ego_s + self.far)
        a_col = tuple(int(c) for c in asphalt)
        self._strip(img, cam, ego, s0, s1, -lw / 2 - 0.6, 1.5 * lw + 0.6, a_col)
        for js0, js1 in self._segments(s0, s1, True):     # intersection box: wide asphalt, no paint
            self._strip(img, cam, ego, js0, js1, -lw / 2 - 9, 1.5 * lw + 9, a_col)

        if augment and rng.random() < 0.5:                  # random shadows / patches on the road
            for _ in range(rng.integers(1, 4)):
                ss = ego_s + rng.uniform(3, 40)
                dd = rng.uniform(-lw, 2 * lw)
                shade = tuple(int(c * rng.uniform(0.4, 0.8)) for c in asphalt)
                self._strip(img, cam, ego, ss, ss + rng.uniform(1, 6), dd, dd + rng.uniform(0.5, 3), shade)

        lines = [(-lw / 2, None), (lw / 2, (3.0, 3.0)), (1.5 * lw, None)]
        for ps0, ps1 in self._segments(s0, s1, False):
            for d, dash in lines:
                self._strip(img, cam, ego, ps0, ps1, d - 0.075, d + 0.075, PAINT, dash)
                if mask is not None:
                    self._strip(mask, cam, ego, ps0, ps1, d - 0.12, d + 0.12, 255, dash)

        # Actors, far to near.
        drawn = []
        for act in actors:
            if not act.exists:
                continue
            p = self.road.point(act.s, act.d)
            _, _, ryaw = self.road.pose(act.s)
            rel = world_to_vehicle(p[None, :], ego[:2], ego[2])[0]
            corners = _cuboid_corners(rel[0], rel[1], ryaw - ego[2], act.length, act.width, act.height)
            uv, depth = cam.project(corners)
            if (depth > NEAR).all():
                drawn.append((depth.mean(), uv, act))
        for _, uv, act in sorted(drawn, key=lambda t: -t[0]):
            hull = cv2.convexHull(_to_px(uv))
            cv2.fillConvexPoly(img, hull, act.color, lineType=cv2.LINE_AA, shift=4)
            cv2.polylines(img, [hull], True, (20, 20, 20), 1, cv2.LINE_AA, shift=4)
            if mask is not None:
                cv2.fillConvexPoly(mask, hull, 0, shift=4)

        if augment or bright != 1.0:
            noise = rng.normal(0, 4.0, img.shape) if augment else 0.0
            img = np.clip(img.astype(np.float32) * bright + noise, 0, 255).astype(np.uint8)
        else:
            img = np.clip(img.astype(np.int16) + rng.integers(-3, 4, img.shape, dtype=np.int16), 0, 255).astype(np.uint8)
        return (img, mask) if with_mask else img


# --------------------------------------------------------------------------- simulator
class OfflineSimulator(Simulator):
    def __init__(self, cfg: dict, scenario: str | None = None):
        self.cfg = cfg
        self.dt = cfg["dt"]
        self.scenario = scenario or cfg["scenario"]
        self.veh = cfg["vehicle"]
        self.lane_width = cfg["planning"]["lane_width"]
        self.cams = make_rig(cfg["cameras"])
        self.cam_left, self.cam_right = (self.cams[n] for n in cfg["cameras"]["stereo_pair"])
        self.stereo_pair = tuple(cfg["cameras"]["stereo_pair"])
        self.rng = np.random.default_rng(cfg.get("seed", 0))

    # -- lifecycle
    def reset(self) -> Frame:
        self.road, self.actors = build_scenario(self.scenario, self.lane_width)
        self.renderer = Renderer(self.road)
        x, y, yaw = self.road.pose(0.0)
        self.ego = np.array([float(x), float(y), float(yaw)])
        self.v = 0.0
        self.t = 0.0
        self.s, self.d = 0.0, 0.0
        self.collisions = 0
        self._in_collision = False
        self.steer_applied = 0.0
        self.yaw_rate = 0.0
        return self._frame()

    def step(self, control: Control) -> Frame:
        dt = self.dt
        max_steer = math.radians(self.veh["max_steer_deg"])
        # First-order steering actuator lag, like a real steering rack.
        target = -float(np.clip(control.steer, -1, 1)) * max_steer      # +steer = right = negative yaw rate
        self.steer_applied += (target - self.steer_applied) * min(1.0, dt / 0.1)
        accel = 3.5 * float(np.clip(control.throttle, 0, 1)) - 8.0 * float(np.clip(control.brake, 0, 1))
        accel -= 0.02 * self.v * self.v / 10 + (0.1 if self.v > 0 else 0.0)  # drag + rolling resistance
        self.v = max(0.0, self.v + accel * dt)
        L = self.veh["wheelbase"]
        self.ego[0] += self.v * math.cos(self.ego[2]) * dt
        self.ego[1] += self.v * math.sin(self.ego[2]) * dt
        self.yaw_rate = self.v / L * math.tan(self.steer_applied)
        self.ego[2] += self.yaw_rate * dt
        self.t += dt
        self.s, self.d = self.road.project(self.ego[:2], self.s)
        for a in self.actors:
            a.update(dt, self.s)
        self._check_collision()
        return self._frame()

    def _check_collision(self):
        hit = False
        L, W = self.veh["length"], self.veh["width"]
        for gt in self._ground_truth():
            if abs(gt.x - self.veh["wheelbase"] / 2) < (L + gt.length) / 2 * 0.95 and abs(gt.y) < (W + gt.width) / 2 * 0.95:
                hit = True
        if hit and not self._in_collision:
            self.collisions += 1
        self._in_collision = hit

    def _ground_truth(self) -> list[GroundTruthObject]:
        out = []
        for a in self.actors:
            if not a.exists:
                continue
            p = self.road.point(a.s, a.d)
            _, _, ryaw = self.road.pose(a.s)
            rel = world_to_vehicle(p[None, :], self.ego[:2], self.ego[2])[0]
            out.append(GroundTruthObject(a.cls, float(rel[0]), float(rel[1]), a.length, a.width,
                                         a.height, float(ryaw - self.ego[2])))
        return out

    def _frame(self) -> Frame:
        s_ahead = np.arange(self.s, min(self.road.length, self.s + 60.0), 1.0)
        if len(s_ahead) < 2:
            s_ahead = np.array([self.road.length - 1.0, self.road.length])
        route = world_to_vehicle(self.road.point(s_ahead, 0.0), self.ego[:2], self.ego[2])
        junction = self.road.is_junction(self.s - 2.0, self.s + 15.0)
        images = {name: self.renderer.render(cam, self.ego, self.s, self.actors, self.rng)
                  for name, cam in self.cams.items()}
        return Frame(images=images, speed=self.v, timestamp=self.t, route=route,
                     yaw_rate=self.yaw_rate, stereo_pair=self.stereo_pair,
                     at_junction=junction, lane_width=self.lane_width,
                     left_lane_available=not junction, ground_truth=self._ground_truth(),
                     route_finished=self.s >= self.road.length - 12.0)

    def status(self) -> dict:
        return {"lateral_error": float(self.d), "collisions": self.collisions,
                "progress": float(self.s), "route_length": float(self.road.length),
                "in_junction": self.road.is_junction(self.s - 1, self.s + 1),
                "ground_truth": self._ground_truth()}
