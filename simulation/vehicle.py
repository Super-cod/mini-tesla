"""The ego vehicle in CARLA plus its collision and lane-invasion sensors."""
from __future__ import annotations

import math

from utils.types import Control


class EgoVehicle:
    def __init__(self, carla, world, blueprint: str, transform):
        bp = world.get_blueprint_library().find(blueprint)
        bp.set_attribute("role_name", "hero")
        self.carla = carla
        self.actor = world.try_spawn_actor(bp, transform)
        if self.actor is None:
            raise RuntimeError("could not spawn the ego vehicle (spawn point occupied?)")
        self.collisions = 0
        self.lane_invasions = 0
        self._last_collision_frame = -100
        lib = world.get_blueprint_library()
        self.collision_sensor = world.spawn_actor(lib.find("sensor.other.collision"), carla.Transform(),
                                                  attach_to=self.actor)
        self.collision_sensor.listen(self._on_collision)
        self.lane_sensor = world.spawn_actor(lib.find("sensor.other.lane_invasion"), carla.Transform(),
                                             attach_to=self.actor)
        self.lane_sensor.listen(self._on_lane_invasion)

    def _on_collision(self, event):
        if event.frame - self._last_collision_frame > 20:      # one collision = one count
            self.collisions += 1
        self._last_collision_frame = event.frame

    def _on_lane_invasion(self, event):
        self.lane_invasions += 1

    @property
    def max_steer_deg(self) -> float:
        wheels = self.actor.get_physics_control().wheels
        return float(wheels[0].max_steer_angle) if wheels else 70.0

    @property
    def speed(self) -> float:
        v = self.actor.get_velocity()
        return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)

    def apply(self, control: Control):
        self.actor.apply_control(self.carla.VehicleControl(
            throttle=float(control.throttle), steer=float(control.steer), brake=float(control.brake)))

    def destroy(self):
        for s in (self.collision_sensor, self.lane_sensor):
            s.stop()
            s.destroy()
        self.actor.destroy()
