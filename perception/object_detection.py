"""Object detection on both camera images.

* YoloDetector       - Ultralytics YOLO (COCO weights) for CARLA / real images.
* GroundTruthDetector - projects the simulator's known actors into each camera. Used by the
                        offline simulator (its cartoon renders are not what YOLO was trained on)
                        and handy in CARLA for debugging the rest of the stack.
"""
from __future__ import annotations

import numpy as np

from utils.geometry import PinholeCamera
from utils.types import Detection, Frame

# COCO class id -> our class name
COCO_TO_CLASS = {0: "pedestrian", 1: "cyclist", 2: "car", 3: "cyclist", 5: "car", 7: "car",
                 9: "traffic_light", 11: "stop_sign"}


class YoloDetector:
    def __init__(self, weights: str = "yolov8n.pt", conf: float = 0.35, device: str | None = None):
        try:
            from ultralytics import YOLO
        except ImportError as e:  # pragma: no cover - depends on optional install
            raise ImportError("YOLO needs the 'ultralytics' package: pip install ultralytics") from e
        import torch

        self.model = YOLO(weights)
        self.conf = conf
        self.device = device or (0 if torch.cuda.is_available() else "cpu")

    def detect_images(self, images: list[np.ndarray]) -> list[list[Detection]]:
        results = self.model.predict(images, conf=self.conf, device=self.device, verbose=False,
                                     classes=list(COCO_TO_CLASS))
        out = []
        for r in results:
            dets = []
            for box, cls, conf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(),
                                      r.boxes.conf.cpu().numpy()):
                name = COCO_TO_CLASS.get(int(cls))
                if name:
                    dets.append(Detection(name, tuple(float(v) for v in box), float(conf)))
            out.append(dets)
        return out

    def detect_pair(self, frame: Frame, cams) -> tuple[list[Detection], list[Detection]]:
        left, right = self.detect_images([frame.left, frame.right])
        return left, right


class GroundTruthDetector:
    def __init__(self, max_range: float = 70.0, pixel_noise: float = 0.7, seed: int = 0):
        self.max_range = max_range
        self.noise = pixel_noise
        self.rng = np.random.default_rng(seed)

    def detect(self, frame: Frame, cam: PinholeCamera) -> list[Detection]:
        out = []
        for obj in frame.ground_truth:
            if obj.x < 0 or np.hypot(obj.x, obj.y) > self.max_range:
                continue
            c, s = np.cos(obj.yaw), np.sin(obj.yaw)
            corners = []
            for dx in (-obj.length / 2, obj.length / 2):
                for dy in (-obj.width / 2, obj.width / 2):
                    for z in (0.0, obj.height):
                        corners.append((obj.x + c * dx - s * dy, obj.y + s * dx + c * dy, z))
            uv, depth = cam.project(np.array(corners))
            if (depth <= 0.6).any():
                continue
            x1, y1 = uv.min(0)
            x2, y2 = uv.max(0)
            x1, x2 = np.clip([x1, x2], 0, cam.width - 1)
            y1, y2 = np.clip([y1, y2], 0, cam.height - 1)
            if (x2 - x1) * (y2 - y1) < 12 or x2 - x1 < 2:
                continue
            jitter = self.rng.normal(0, self.noise, 4)
            out.append(Detection(obj.cls, (x1 + jitter[0], y1 + jitter[1], x2 + jitter[2], y2 + jitter[3]), 0.99))
        return out

    def detect_pair(self, frame: Frame, cams) -> tuple[list[Detection], list[Detection]]:
        return self.detect(frame, cams[0]), self.detect(frame, cams[1])


def make_detector(cfg: dict, backend: str):
    p = cfg["perception"]
    kind = p.get("detector", "auto")
    if kind == "auto":
        kind = "ground_truth" if backend == "sim" else "yolo"
    if kind == "yolo":
        return YoloDetector(p["yolo_weights"], p["yolo_conf"]), "yolo"
    return GroundTruthDetector(), "ground_truth"
