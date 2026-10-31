"""Combine all cameras into one 3D world model.

* Front stereo pair: lane estimate (both cameras) + objects positioned by stereo disparity.
* Every camera (including the rear-facing side cameras): objects that the stereo pair cannot
  see are positioned with the ground-plane (monocular) estimate.
* Each 3D measurement carries a covariance (stereo depth error grows with distance squared,
  mono error grows linearly), which the tracker uses to weight it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fusion.tracker import MultiObjectTracker
from perception.stereo_depth import DenseStereo, depth_from_disparity, match_detections, mono_ground_position
from utils.geometry import PinholeCamera
from utils.types import Detection, Frame, LaneEstimate, TrackedObject, WorldModel


@dataclass
class Measurement:
    obj: TrackedObject
    R: np.ndarray                    # 2x2 covariance of (x, y) in the vehicle frame


def fuse_lanes(estimates: list[LaneEstimate | None]) -> LaneEstimate | None:
    good = [e for e in estimates if e is not None]
    if not good:
        return None
    w = np.array([e.confidence for e in good])
    coeffs = (np.stack([e.coeffs for e in good]) * w[:, None]).sum(0) / w.sum()
    widths = [e.width for e in good if e.width is not None]
    conf = float(min(1.0, w.max() + 0.1 * (len(good) - 1)))
    return LaneEstimate(coeffs=coeffs, confidence=conf, width=float(np.mean(widths)) if widths else None)


def _rotated_cov(origin: np.ndarray, p: np.ndarray, sigma_range: float, sigma_lateral: float) -> np.ndarray:
    """Covariance with `sigma_range` along the viewing ray from `origin` to `p`."""
    d = p[:2] - origin[:2]
    a = np.arctan2(d[1], d[0])
    c, s = np.cos(a), np.sin(a)
    rot = np.array([[c, -s], [s, c]])
    return rot @ np.diag([sigma_range ** 2, sigma_lateral ** 2]) @ rot.T


class CameraFusion:
    def __init__(self, cfg: dict, cams: dict[str, PinholeCamera]):
        self.cams = cams
        self.left_name, self.right_name = cfg["cameras"]["stereo_pair"]
        self.cam_left, self.cam_right = cams[self.left_name], cams[self.right_name]
        self.baseline = float(np.linalg.norm(self.cam_left.position - self.cam_right.position))
        self.stereo_method = cfg["perception"].get("stereo_method", "boxes")
        self.dense = DenseStereo() if self.stereo_method == "sgbm" else None
        self.tracker = MultiObjectTracker(cfg["dt"])
        self.smooth = 0.5          # EMA weight of the newest lane measurement
        self._lane: LaneEstimate | None = None
        self._lane_age = 0

    # ------------------------------------------------------------------ lanes
    def _smooth_lane(self, lane: LaneEstimate | None) -> LaneEstimate | None:
        if lane is None:
            self._lane_age += 1
            if self._lane is not None and self._lane_age <= 3:      # bridge short dropouts
                return LaneEstimate(self._lane.coeffs.copy(), self._lane.confidence * 0.6, width=self._lane.width)
            self._lane = None
            return None
        self._lane_age = 0
        if self._lane is not None and abs(lane.coeffs[0] - self._lane.coeffs[0]) < 1.0:
            a = self.smooth
            lane = LaneEstimate(a * lane.coeffs + (1 - a) * self._lane.coeffs, lane.confidence, width=lane.width)
        self._lane = lane
        return lane

    # ------------------------------------------------------------------ objects
    @staticmethod
    def _truncated(det: Detection, cam: PinholeCamera, margin: float = 3.0) -> bool:
        """Box cut off by the image border: its centre / bottom no longer describe the object."""
        x1, _, x2, y2 = det.bbox
        return x1 <= margin or x2 >= cam.width - 1 - margin or y2 >= cam.height - 1 - margin

    def measure_objects(self, frame: Frame, dets: dict[str, list[Detection]]) -> list[Measurement]:
        out: list[Measurement] = []
        dets_l, dets_r = dets.get(self.left_name, []), dets.get(self.right_name, [])
        # Stereo only for boxes fully inside both images (a cut-off box has a shifted centre).
        ok_l = [k for k, d in enumerate(dets_l) if not self._truncated(d, self.cam_left)]
        ok_r = [k for k, d in enumerate(dets_r) if not self._truncated(d, self.cam_right)]
        matches = [(ok_l[i], ok_r[j], disp) for i, j, disp in
                   match_detections([dets_l[k] for k in ok_l], [dets_r[k] for k in ok_r], self.cam_left.height)]
        disp_map = self.dense.disparity(frame.left, frame.right) if self.dense is not None else None
        used = {self.left_name: {i for i, _, _ in matches}, self.right_name: {j for _, j, _ in matches}}
        fb = self.cam_left.fx * self.baseline
        for i, j, disp in matches:
            det = dets_l[i]
            if disp_map is not None:
                d2 = DenseStereo.box_disparity(disp_map, det.bbox)
                disp = d2 if d2 is not None else disp
            z = depth_from_disparity(disp, self.cam_left.fx, self.baseline)
            u, v = det.center
            p = self.cam_left.pixel_depth_to_vehicle(u, v, z)
            R = _rotated_cov(self.cam_left.position, p, z * z / fb * 1.0 + 0.2, 0.2 + 0.01 * z)
            obj = TrackedObject(det.cls, float(p[0]), float(p[1]), float(np.hypot(p[0], p[1])),
                                det.conf, "stereo", det.bbox, camera=self.left_name)
            out.append(Measurement(obj, R))
        # Everything else (side cameras, edge of the stereo view): ground-plane estimate.
        mono: list[Measurement] = []
        for name, cam in self.cams.items():
            for k, det in enumerate(dets.get(name, [])):
                if k in used.get(name, ()) or det.cls in ("traffic_light", "stop_sign"):
                    continue
                p = mono_ground_position(det, cam)
                if p is None:
                    continue
                rng = float(np.hypot(*(p - cam.position[:2])))
                scale = 2.5 if self._truncated(det, cam) else 1.0
                R = _rotated_cov(cam.position, p, (0.3 + 0.08 * rng) * scale, (0.2 + 0.02 * rng) * scale)
                obj = TrackedObject(det.cls, float(p[0]), float(p[1]), float(np.hypot(*p)),
                                    det.conf * 0.8, "mono", det.bbox, camera=name)
                mono.append(Measurement(obj, R))
        # The same object seen by several cameras: if two measurements are statistically
        # compatible (Mahalanobis gate), fuse them weighted by their covariances.
        for m in sorted(mono, key=lambda m: np.linalg.det(m.R)):
            z = np.array([m.obj.x, m.obj.y])
            merged = False
            for o in out:
                if o.obj.cls != m.obj.cls:
                    continue
                zo = np.array([o.obj.x, o.obj.y])
                d = z - zo
                if d @ np.linalg.solve(o.R + m.R, d) < 9.2:                 # 99 % gate, 2 dof
                    if o.obj.source == "mono":
                        info_o, info_m = np.linalg.inv(o.R), np.linalg.inv(m.R)
                        R = np.linalg.inv(info_o + info_m)
                        x, y = R @ (info_o @ zo + info_m @ z)
                        o.obj.x, o.obj.y, o.obj.distance = float(x), float(y), float(np.hypot(x, y))
                        o.R = R
                    merged = True
                    break
            if not merged:
                out.append(m)
        return out

    def locate_objects(self, frame: Frame, dets: dict[str, list[Detection]]) -> list[TrackedObject]:
        """Single-frame 3D positions (no tracking)."""
        return [m.obj for m in self.measure_objects(frame, dets)]

    def signs(self, frame: Frame, dets: dict[str, list[Detection]]) -> list[TrackedObject]:
        """Stop signs from the stereo pair (static, so they are not tracked)."""
        out = []
        dets_l, dets_r = dets.get(self.left_name, []), dets.get(self.right_name, [])
        for i, j, disp in match_detections(dets_l, dets_r, self.cam_left.height):
            det = dets_l[i]
            if det.cls != "stop_sign":
                continue
            z = depth_from_disparity(disp, self.cam_left.fx, self.baseline)
            p = self.cam_left.pixel_depth_to_vehicle(*det.center, z)
            out.append(TrackedObject("stop_sign", float(p[0]), float(p[1]), float(np.hypot(p[0], p[1])),
                                     det.conf, "stereo", det.bbox, camera=self.left_name))
        return out

    def fuse(self, frame: Frame, lane_l: LaneEstimate | None, lane_r: LaneEstimate | None,
             dets: dict[str, list[Detection]]) -> WorldModel:
        lane = self._smooth_lane(fuse_lanes([lane_l, lane_r]))
        tracks = self.tracker.step(self.measure_objects(frame, dets), frame.speed, frame.yaw_rate)
        return WorldModel(lane=lane, objects=tracks + self.signs(frame, dets), speed=frame.speed,
                          route=frame.route, at_junction=frame.at_junction, lane_width=frame.lane_width,
                          left_lane_available=frame.left_lane_available)
