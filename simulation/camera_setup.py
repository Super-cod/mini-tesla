"""The camera rig in CARLA: the front stereo pair plus the rear-facing side cameras.

The same `cameras` config block builds the PinholeCamera models used by perception
(utils.geometry.make_rig), so intrinsics/extrinsics always agree.
"""
from __future__ import annotations

import queue

import numpy as np


def camera_transforms(cam_cfg: dict, wheelbase: float) -> dict[str, tuple[float, float, float, float, float]]:
    """CARLA mount (x, y, z, pitch, yaw) per camera, relative to the vehicle actor origin.

    Our config measures x from the rear axle in a right-handed frame (y left, yaw + = left);
    CARLA's actor origin is mid-wheelbase and its frame is left-handed (y right, yaw + = right).
    """
    out = {}
    for name, c in cam_cfg["rig"].items():
        out[name] = (c["x"] - wheelbase / 2, -c["y"], c["z"], c.get("pitch", 0.0), -c.get("yaw", 0.0))
    return out


def image_to_bgr(image) -> np.ndarray:
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    return arr[:, :, :3].copy()


class CameraRig:
    def __init__(self, carla, world, vehicle, cam_cfg: dict, wheelbase: float):
        lib = world.get_blueprint_library()
        self.sensors, self.queues = {}, {}
        for name, (x, y, z, pitch, yaw) in camera_transforms(cam_cfg, wheelbase).items():
            bp = lib.find("sensor.camera.rgb")
            bp.set_attribute("image_size_x", str(cam_cfg["width"]))
            bp.set_attribute("image_size_y", str(cam_cfg["height"]))
            bp.set_attribute("fov", str(cam_cfg["rig"][name]["fov"]))
            tf = carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(pitch=pitch, yaw=yaw))
            sensor = world.spawn_actor(bp, tf, attach_to=vehicle)
            q: queue.Queue = queue.Queue()
            sensor.listen(q.put)
            self.sensors[name], self.queues[name] = sensor, q

    def read(self, frame_id: int, timeout: float = 5.0) -> dict[str, np.ndarray]:
        """Images of every camera captured on world tick `frame_id` (synchronous mode)."""
        out = {}
        for name, q in self.queues.items():
            while True:
                img = q.get(timeout=timeout)
                if img.frame >= frame_id:
                    out[name] = image_to_bgr(img)
                    break
        return out

    def destroy(self):
        for s in self.sensors.values():
            s.stop()
            s.destroy()
