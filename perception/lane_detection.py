"""Lane detection: pixel segmentation (CNN or classical colour thresholds) followed by a
geometric fit of the two lane boundaries on the road plane (inverse perspective mapping)."""
from __future__ import annotations

import logging

import cv2
import numpy as np

from utils.config import resolve_path
from utils.geometry import PinholeCamera
from utils.types import LaneEstimate

log = logging.getLogger(__name__)


class ClassicalSegmenter:
    """White / yellow paint thresholding in HLS space. Works without any training."""

    def __call__(self, img_bgr: np.ndarray, horizon: int) -> np.ndarray:
        hls = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)
        h, l, s = hls[..., 0], hls[..., 1], hls[..., 2]
        white = (l > 190) & (s < 90)
        yellow = (h > 12) & (h < 35) & (s > 90) & (l > 100)
        mask = (white | yellow).astype(np.float32)
        mask[:horizon] = 0
        return mask


class CNNSegmenter:
    def __init__(self, weights_path, input_hw=(128, 256), device: str | None = None):
        import torch
        from models.lane_cnn import load_lane_cnn

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_lane_cnn(weights_path, self.device)
        self.input_hw = tuple(input_hw)

    def __call__(self, img_bgr: np.ndarray, horizon: int) -> np.ndarray:
        torch = self.torch
        h, w = img_bgr.shape[:2]
        x = cv2.resize(img_bgr, (self.input_hw[1], self.input_hw[0]), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(x[..., ::-1].copy()).permute(2, 0, 1).float().div(255.0)[None].to(self.device)
        with torch.no_grad():
            prob = torch.sigmoid(self.model(x))[0, 0].cpu().numpy()
        prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
        prob[:horizon] = 0
        return prob


def make_segmenter(cfg: dict):
    p = cfg["perception"]
    method = p.get("lane_method", "auto")
    weights = resolve_path(p["lane_weights"])
    if method in ("auto", "cnn"):
        if weights.exists():
            try:
                return CNNSegmenter(weights, p.get("lane_cnn_input", (128, 256))), "cnn"
            except Exception as e:  # pragma: no cover - only on broken installs
                if method == "cnn":
                    raise
                log.warning("Lane CNN unavailable (%s); using classical segmentation", e)
        elif method == "cnn":
            raise FileNotFoundError(f"lane CNN weights not found: {weights} (run models/train_lane_cnn.py)")
    return ClassicalSegmenter(), "classical"


class LaneFitter:
    """Turns a marking mask into a LaneEstimate for the lane the car is driving in."""

    def __init__(self, lane_width: float = 3.5, x_range=(4.0, 30.0), threshold: float = 0.5,
                 max_points: int = 4000):
        self.lane_width = lane_width
        self.x_min, self.x_max = x_range
        self.threshold = threshold
        self.max_points = max_points
        self.rng = np.random.default_rng(0)

    def ground_points(self, mask: np.ndarray, cam: PinholeCamera) -> np.ndarray:
        vs, us = np.nonzero(mask > self.threshold)
        if len(us) == 0:
            return np.empty((0, 2))
        if len(us) > self.max_points:
            keep = self.rng.choice(len(us), self.max_points, replace=False)
            us, vs = us[keep], vs[keep]
        pts, ok = cam.pixel_to_ground(np.stack([us + 0.5, vs + 0.5], 1))
        pts = pts[ok]
        sel = (pts[:, 0] > self.x_min) & (pts[:, 0] < self.x_max) & (np.abs(pts[:, 1]) < 14)
        return pts[sel]

    @staticmethod
    def _peaks(y: np.ndarray, lo: float, hi: float, bin_w: float = 0.15) -> list[tuple[float, float]]:
        if len(y) == 0:
            return []
        bins = np.arange(lo, hi + bin_w, bin_w)
        hist, edges = np.histogram(y, bins=bins)
        hist = np.convolve(hist, [1, 2, 1], mode="same").astype(float)
        if hist.max() <= 0:
            return []
        floor = max(6.0, 0.08 * hist.max())
        out = []
        for i in range(1, len(hist) - 1):
            if hist[i] >= floor and hist[i] >= hist[i - 1] and hist[i] > hist[i + 1]:
                out.append(((edges[i] + edges[i + 1]) / 2, hist[i]))
        return out

    def _track(self, pts: np.ndarray, y0: float) -> np.ndarray | None:
        """Sliding-window search along x starting from lateral position y0 near the car."""
        win, half = 2.0, 0.55
        chosen = []
        yc, slope = y0, 0.0
        for x0 in np.arange(self.x_min, self.x_max, win):
            xm = x0 + win / 2
            pred = yc + slope * win
            sel = (pts[:, 0] >= x0) & (pts[:, 0] < x0 + win) & (np.abs(pts[:, 1] - pred) < half + 0.02 * xm)
            if sel.sum() >= 3:
                chosen.append(pts[sel])
                ny = float(np.median(pts[sel, 1]))
                slope = 0.5 * slope + 0.5 * (ny - yc) / win if chosen[:-1] else slope
                yc = ny
            else:
                yc = pred
        if not chosen:
            return None
        c = np.vstack(chosen)
        if len(c) < 8 or np.ptp(c[:, 0]) < 4.0:
            return None
        deg = 2 if np.ptp(c[:, 0]) > 10.0 else 1
        coef = np.polyfit(c[:, 0], c[:, 1], deg)[::-1]          # ascending order
        return np.pad(coef, (0, 3 - len(coef)))

    def fit(self, mask: np.ndarray, cam: PinholeCamera) -> LaneEstimate | None:
        pts = self.ground_points(mask, cam)
        if len(pts) < 10:
            return None
        near = pts[pts[:, 0] < self.x_min + 10.0]
        peaks = self._peaks(near[:, 1], -6.0, 6.0)
        lefts = [p for p in peaks if p[0] > 0.2]
        rights = [p for p in peaks if p[0] < -0.2]
        left = right = None
        if lefts:
            left = self._track(pts, min(lefts, key=lambda p: p[0])[0])
        if rights:
            right = self._track(pts, max(rights, key=lambda p: p[0])[0])

        w = self.lane_width
        if left is not None and right is not None:
            width = left[0] - right[0]
            if not (0.6 * w < width < 1.5 * w):
                # Implausible pair: keep the boundary closest to its expected place.
                if abs(left[0] - w / 2) < abs(right[0] + w / 2):
                    right = None
                else:
                    left = None
        if left is not None and right is not None:
            centre = (left + right) / 2
            conf, width = 1.0, float(left[0] - right[0])
        elif left is not None:
            centre = left - np.array([w / 2, 0, 0])
            conf, width = 0.6, None
        elif right is not None:
            centre = right + np.array([w / 2, 0, 0])
            conf, width = 0.6, None
        else:
            return None
        return LaneEstimate(coeffs=centre, confidence=conf, left_coeffs=left, right_coeffs=right, width=width)


class LaneDetector:
    def __init__(self, cfg: dict):
        self.segment, self.method = make_segmenter(cfg)
        p = cfg["perception"]
        self.fitter = LaneFitter(cfg["planning"]["lane_width"], tuple(p["lane_x_range"]), p["lane_threshold"])

    def detect(self, img: np.ndarray, cam: PinholeCamera) -> tuple[LaneEstimate | None, np.ndarray]:
        horizon = int(np.clip(cam.cy + cam.fy * np.tan(np.radians(cam.pitch_deg)) + 4, 0, cam.height))
        mask = self.segment(img, horizon)
        return self.fitter.fit(mask, cam), mask
