"""Combine both camera views into one WorldModel: a single lane estimate and a list of objects
positioned in metres in the vehicle frame."""
from __future__ import annotations

import numpy as np

from perception.stereo_depth import DenseStereo, depth_from_disparity, match_detections, mono_ground_position
from utils.geometry import PinholeCamera
from utils.types import Detection, Frame, LaneEstimate, TrackedObject, WorldModel


def fuse_lanes(estimates: list[LaneEstimate | None]) -> LaneEstimate | None:
    good = [e for e in estimates if e is not None]
    if not good:
        return None
    w = np.array([e.confidence for e in good])
    coeffs = (np.stack([e.coeffs for e in good]) * w[:, None]).sum(0) / w.sum()
    widths = [e.width for e in good if e.width is not None]
    conf = float(min(1.0, w.max() + 0.1 * (len(good) - 1)))
    return LaneEstimate(coeffs=coeffs, confidence=conf, width=float(np.mean(widths)) if widths else None)


class CameraFusion:
    def __init__(self, cfg: dict, cam_left: PinholeCamera, cam_right: PinholeCamera):
        self.cam_left, self.cam_right = cam_left, cam_right
        self.baseline = float(np.linalg.norm(cam_left.position - cam_right.position))
        self.stereo_method = cfg["perception"].get("stereo_method", "boxes")
        self.dense = DenseStereo() if self.stereo_method == "sgbm" else None
        self.smooth = 0.5          # EMA weight of the newest lane measurement
        self._lane: LaneEstimate | None = None
        self._lane_age = 0

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

    def locate_objects(self, frame: Frame, dets_l: list[Detection], dets_r: list[Detection]) -> list[TrackedObject]:
        objects: list[TrackedObject] = []
        matches = match_detections(dets_l, dets_r, self.cam_left.height)
        disp_map = self.dense.disparity(frame.left, frame.right) if self.dense is not None else None
        matched_l = {i for i, _, _ in matches}
        matched_r = {j for _, j, _ in matches}
        for i, j, disp in matches:
            det = dets_l[i]
            if disp_map is not None:
                d2 = DenseStereo.box_disparity(disp_map, det.bbox)
                disp = d2 if d2 is not None else disp
            z = depth_from_disparity(disp, self.cam_left.fx, self.baseline)
            u, v = det.center
            p = self.cam_left.pixel_depth_to_vehicle(u, v, z)
            objects.append(TrackedObject(det.cls, float(p[0]), float(p[1]), float(np.hypot(p[0], p[1])),
                                         det.conf, "stereo", det.bbox))
        # Objects seen by only one camera (edge of the field of view): monocular ground-plane estimate.
        for dets, used, cam in ((dets_l, matched_l, self.cam_left), (dets_r, matched_r, self.cam_right)):
            for k, det in enumerate(dets):
                if k in used or det.cls in ("traffic_light", "stop_sign"):
                    continue
                p = mono_ground_position(det, cam)
                if p is None:
                    continue
                if any(np.hypot(o.x - p[0], o.y - p[1]) < 1.5 for o in objects):
                    continue
                objects.append(TrackedObject(det.cls, float(p[0]), float(p[1]), float(np.hypot(*p)),
                                             det.conf * 0.8, "mono", det.bbox))
        # Signs: mono distance from apparent size is unreliable, so only keep stereo ones above.
        return objects

    def fuse(self, frame: Frame, lane_l: LaneEstimate | None, lane_r: LaneEstimate | None,
             dets_l: list[Detection], dets_r: list[Detection]) -> WorldModel:
        lane = self._smooth_lane(fuse_lanes([lane_l, lane_r]))
        return WorldModel(lane=lane, objects=self.locate_objects(frame, dets_l, dets_r), speed=frame.speed,
                          route=frame.route, at_junction=frame.at_junction, lane_width=frame.lane_width,
                          left_lane_available=frame.left_lane_available)
