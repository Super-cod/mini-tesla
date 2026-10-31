"""Two forward RGB cameras in CARLA forming a parallel stereo rig.

The same `cameras` config block builds the PinholeCamera models used by perception
(utils.geometry.make_stereo_pair), so intrinsics/extrinsics always agree.
"""
from __future__ import annotations

import queue

import numpy as np


def camera_transforms(cam_cfg: dict, wheelbase: float) -> dict[str, tuple[float, float, float, float]]:
    """CARLA mount (x, y, z, pitch) relative to the vehicle actor origin (its centre, y = right).

    Our config measures mount_x from the rear axle; CARLA's actor origin is mid-wheelbase.
    """
    x = cam_cfg["mount_x"] - wheelbase / 2
    half = cam_cfg["baseline"] / 2
    return {"left": (x, -half, cam_cfg["mount_z"], cam_cfg["pitch_deg"]),
            "right": (x, +half, cam_cfg["mount_z"], cam_cfg["pitch_deg"])}


def image_to_bgr(image) -> np.ndarray:
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    return arr[:, :, :3].copy()


class StereoRig:
    def __init__(self, carla, world, vehicle, cam_cfg: dict, wheelbase: float):
        bp = world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(cam_cfg["width"]))
        bp.set_attribute("image_size_y", str(cam_cfg["height"]))
        bp.set_attribute("fov", str(cam_cfg["fov"]))
        self.sensors, self.queues = {}, {}
        for name, (x, y, z, pitch) in camera_transforms(cam_cfg, wheelbase).items():
            tf = carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(pitch=pitch))
            sensor = world.spawn_actor(bp, tf, attach_to=vehicle)
            q: queue.Queue = queue.Queue()
            sensor.listen(q.put)
            self.sensors[name], self.queues[name] = sensor, q

    def read(self, frame_id: int, timeout: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
        """Images captured on world tick `frame_id` (synchronous mode)."""
        out = {}
        for name, q in self.queues.items():
            while True:
                img = q.get(timeout=timeout)
                if img.frame >= frame_id:
                    out[name] = image_to_bgr(img)
                    break
        return out["left"], out["right"]

    def destroy(self):
        for s in self.sensors.values():
            s.stop()
            s.destroy()
