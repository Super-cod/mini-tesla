"""Collect lane-marking training data from CARLA (images + masks for models/train_lane_cnn.py).

A CARLA autopilot drives the ego car around; an RGB camera and a semantic-segmentation camera
share the left-camera mount. Pixels tagged RoadLine become the lane mask.

    python scripts/collect_carla_data.py --frames 3000 --out data/carla_lanes
    python -m models.train_lane_cnn --data data/carla_lanes --epochs 20
"""
from __future__ import annotations

import argparse
import queue
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulation.camera_setup import camera_transforms, image_to_bgr  # noqa: E402
from simulation.carla_connection import CarlaConnection  # noqa: E402
from simulation.vehicle import EgoVehicle  # noqa: E402
from utils.config import load_config  # noqa: E402

ROADLINE_TAG = 24   # CARLA >= 0.9.14 (Cityscapes palette). Use 6 for CARLA <= 0.9.13.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--every", type=int, default=5, help="save one frame every N ticks")
    ap.add_argument("--out", type=Path, default=Path("data/carla_lanes"))
    ap.add_argument("--roadline-tag", type=int, default=ROADLINE_TAG)
    args = ap.parse_args()

    cfg = load_config()
    conn = CarlaConnection(cfg)
    carla, world = conn.carla, conn.world
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    (args.out / "masks").mkdir(parents=True, exist_ok=True)
    cam = cfg["cameras"]
    x, y, z, pitch = camera_transforms(cam, cfg["vehicle"]["wheelbase"])["left"]
    tf = carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(pitch=pitch))

    ego = EgoVehicle(carla, world, cfg["carla"]["vehicle_blueprint"], conn.map.get_spawn_points()[0])
    ego.actor.set_autopilot(True, conn.traffic_manager.get_port())
    sensors, queues = [], []
    for kind in ("sensor.camera.rgb", "sensor.camera.semantic_segmentation"):
        bp = world.get_blueprint_library().find(kind)
        bp.set_attribute("image_size_x", str(cam["width"]))
        bp.set_attribute("image_size_y", str(cam["height"]))
        bp.set_attribute("fov", str(cam["fov"]))
        s = world.spawn_actor(bp, tf, attach_to=ego.actor)
        q: queue.Queue = queue.Queue()
        s.listen(q.put)
        sensors.append(s)
        queues.append(q)
    conn.spawn_traffic(cfg["carla"]["traffic_vehicles"], cfg["carla"]["traffic_walkers"])

    saved = tick = 0
    weathers = [carla.WeatherParameters.ClearNoon, carla.WeatherParameters.CloudyNoon,
                carla.WeatherParameters.WetNoon, carla.WeatherParameters.ClearSunset]
    try:
        while saved < args.frames:
            frame_id = conn.tick()
            rgb = queues[0].get(timeout=5.0)
            sem = queues[1].get(timeout=5.0)
            tick += 1
            if tick % 400 == 0:
                world.set_weather(weathers[(tick // 400) % len(weathers)])
            if tick % args.every or rgb.frame != frame_id:
                continue
            tags = image_to_bgr(sem)[:, :, 2]                   # semantic tag lives in the red channel
            mask = np.where(tags == args.roadline_tag, 255, 0).astype(np.uint8)
            name = f"{saved:06d}.png"
            cv2.imwrite(str(args.out / "images" / name), image_to_bgr(rgb))
            cv2.imwrite(str(args.out / "masks" / name), mask)
            saved += 1
            if saved % 100 == 0:
                print(f"{saved}/{args.frames}")
    finally:
        for s in sensors:
            s.stop()
            s.destroy()
        ego.destroy()
        conn.close()


if __name__ == "__main__":
    main()
