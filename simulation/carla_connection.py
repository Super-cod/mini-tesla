"""Connect to a running CARLA server, configure synchronous mode and spawn background traffic.

Tested API surface: CARLA 0.9.14 - 0.9.16 (`pip install carla==0.9.16` must match the server).
"""
from __future__ import annotations

import logging
import math
import random

import numpy as np

log = logging.getLogger(__name__)


def import_carla():
    try:
        import carla  # noqa: F401
    except ImportError as e:  # pragma: no cover - depends on the CARLA install
        raise ImportError("CARLA Python API not found. Install the wheel that matches your server, "
                          "e.g. `pip install carla==0.9.16`, and start CarlaUE4 first.") from e
    return carla


# ---------------------------------------------------------------- coordinate conversion
# CARLA uses a left-handed frame (x forward, y right, z up, yaw clockwise from above).
# The stack uses a right-handed one (x forward, y left). Flipping y (and yaw) converts.
def carla_to_rh(x: float, y: float) -> tuple[float, float]:
    return x, -y


def carla_yaw_to_rh(yaw_deg: float) -> float:
    return -math.radians(yaw_deg)


def rear_axle_pose(transform, wheelbase: float) -> tuple[np.ndarray, float]:
    """Ego reference point (rear axle, right-handed world coordinates) and heading."""
    yaw = carla_yaw_to_rh(transform.rotation.yaw)
    x, y = carla_to_rh(transform.location.x, transform.location.y)
    return np.array([x - math.cos(yaw) * wheelbase / 2, y - math.sin(yaw) * wheelbase / 2]), yaw


class CarlaConnection:
    def __init__(self, cfg: dict):
        self.carla = import_carla()
        c = cfg["carla"]
        self.cfg = cfg
        self.client = self.carla.Client(c["host"], c["port"])
        self.client.set_timeout(c["timeout"])
        world = self.client.get_world()
        if c.get("town") and not world.get_map().name.endswith(c["town"]):
            log.info("loading %s", c["town"])
            world = self.client.load_world(c["town"])
        self.world = world
        self.map = world.get_map()
        self._original_settings = world.get_settings()
        self.traffic_manager = self.client.get_trafficmanager(8000)
        self.actors: list = []                       # everything we spawn, destroyed on close()
        self.walker_controllers: list = []
        if c.get("synchronous", True):
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = cfg["dt"]
            world.apply_settings(settings)
            self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_random_device_seed(cfg.get("seed", 0))

    def tick(self) -> int:
        return self.world.tick()

    def spawn_traffic(self, n_vehicles: int, n_walkers: int, avoid_location=None):
        carla = self.carla
        rng = random.Random(self.cfg.get("seed", 0))
        bps = [b for b in self.world.get_blueprint_library().filter("vehicle.*")
               if int(b.get_attribute("number_of_wheels")) == 4]
        points = self.map.get_spawn_points()
        rng.shuffle(points)
        for tf in points:
            if n_vehicles <= 0:
                break
            if avoid_location is not None and tf.location.distance(avoid_location) < 15:
                continue
            actor = self.world.try_spawn_actor(rng.choice(bps), tf)
            if actor:
                actor.set_autopilot(True, self.traffic_manager.get_port())
                self.actors.append(actor)
                n_vehicles -= 1

        walker_bps = self.world.get_blueprint_library().filter("walker.pedestrian.*")
        ctrl_bp = self.world.get_blueprint_library().find("controller.ai.walker")
        for _ in range(n_walkers * 3):
            if n_walkers <= 0:
                break
            loc = self.world.get_random_location_from_navigation()
            if loc is None:
                continue
            walker = self.world.try_spawn_actor(rng.choice(walker_bps), carla.Transform(loc))
            if not walker:
                continue
            ctrl = self.world.spawn_actor(ctrl_bp, carla.Transform(), attach_to=walker)
            self.actors.append(walker)
            self.walker_controllers.append(ctrl)
            n_walkers -= 1
        self.tick()
        for ctrl in self.walker_controllers:
            ctrl.start()
            ctrl.go_to_location(self.world.get_random_location_from_navigation())
            ctrl.set_max_speed(1.2 + rng.random() * 0.6)
        log.info("spawned %d traffic actors", len(self.actors))

    def close(self):
        for ctrl in self.walker_controllers:
            try:
                ctrl.stop()
            except RuntimeError:
                pass
        ids = [a.id for a in self.actors + self.walker_controllers]
        if ids:
            self.client.apply_batch([self.carla.command.DestroyActor(i) for i in ids])
        self.actors, self.walker_controllers = [], []
        self.traffic_manager.set_synchronous_mode(False)
        self.world.apply_settings(self._original_settings)
