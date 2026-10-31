"""Distance estimation from the two front cameras.

The cameras share orientation and are offset only sideways (the baseline), so the image pair
is already rectified: a point at depth Z appears `d = fx * B / Z` pixels further right in the
left image than in the right one.
"""
from __future__ import annotations

import cv2
import numpy as np

from utils.geometry import PinholeCamera
from utils.types import Detection


def depth_from_disparity(disparity: float, fx: float, baseline: float) -> float:
    if disparity <= 0:
        return float("inf")
    return fx * baseline / disparity


def match_detections(left: list[Detection], right: list[Detection], img_h: int,
                     max_disparity: float = 250.0) -> list[tuple[int, int, float]]:
    """Greedy left/right box matching. Returns (left_idx, right_idx, disparity)."""
    candidates = []
    for i, dl in enumerate(left):
        lx, ly = dl.center
        lh = dl.bbox[3] - dl.bbox[1]
        for j, dr in enumerate(right):
            if dr.cls != dl.cls:
                continue
            rx, ry = dr.center
            rh = dr.bbox[3] - dr.bbox[1]
            disp = lx - rx
            if not (0.2 < disp < max_disparity):
                continue
            dy = abs(ly - ry)
            if dy > 0.15 * max(lh, rh) + 3:
                continue
            size = abs(lh - rh) / max(lh, rh, 1.0)
            candidates.append((dy / img_h + size, i, j, disp))
    used_l, used_r, out = set(), set(), []
    for _, i, j, disp in sorted(candidates):
        if i in used_l or j in used_r:
            continue
        used_l.add(i)
        used_r.add(j)
        out.append((i, j, disp))
    return out


def mono_ground_position(det: Detection, cam: PinholeCamera) -> np.ndarray | None:
    """Single-camera fallback: the bottom-centre of the box touches the road plane."""
    x1, _, x2, y2 = det.bbox
    pts, ok = cam.pixel_to_ground(np.array([[(x1 + x2) / 2, y2]]))
    return pts[0] if ok[0] else None


class DenseStereo:
    """Semi-global block matching disparity map (optional, used when stereo_method == 'sgbm')."""

    def __init__(self, num_disparities: int = 128, block_size: int = 7):
        self.matcher = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=num_disparities, blockSize=block_size,
            P1=8 * block_size ** 2, P2=32 * block_size ** 2, uniquenessRatio=8,
            speckleWindowSize=60, speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    def disparity(self, left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
        gl = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
        gr = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)
        return self.matcher.compute(gl, gr).astype(np.float32) / 16.0

    @staticmethod
    def box_disparity(disp: np.ndarray, bbox) -> float | None:
        x1, y1, x2, y2 = (int(round(v)) for v in bbox)
        w, h = x2 - x1, y2 - y1
        roi = disp[y1 + h // 4: y2 - h // 4, x1 + w // 4: x2 - w // 4]
        vals = roi[roi > 0.5]
        return float(np.median(vals)) if vals.size >= 5 else None
