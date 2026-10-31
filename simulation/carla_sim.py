"""CARLA backend: ego Tesla Model 3 with the stereo rig, a navigation route through the town
(turns at intersections come from the route), and background traffic."""
from __future__ import annotations

import logging
import math

import numpy as np

from simulation.base import Simulator
from simulation.camera_setup import CameraRig
from simulation.carla_connection import CarlaConnection, carla_to_rh, carla_yaw_to_rh, rear_axle_pose
from simulation.vehicle import EgoVehicle
from utils.geometry import world_to_vehicle
from utils.types import Control, Frame, GroundTruthObject

log = logging.getLogger(__name__)
ROUTE_STEP = 2.0          # metres between route waypoints


def plan_route(carla_map, start_loc, end_loc, max_len: float = 1500.0) -> list:
    """Waypoints from start to end. Uses CARLA's GlobalRoutePlanner when the PythonAPI `agents`
    package is importable, otherwise just follows the lane (first branch at each junction)."""
    try:
        from agents.navigation.global_route_planner import GlobalRoutePlanner
        grp = GlobalRoutePlanner(carla_map, ROUTE_STEP)
        route = [wp for wp, _ in grp.trace_route(start_loc, end_loc)]
        if len(route) > 5:
            return route
    except ImportError:
        log.warning("CARLA 'agents' package not on PYTHONPATH (add <CARLA_ROOT>/PythonAPI/carla); "
                    "falling back to simple lane following route")
    wp = carla_map.get_waypoint(start_loc)
    route = [wp]
    for _ in range(int(max_len / ROUTE_STEP)):
        nxt = wp.next(ROUTE_STEP)
        if not nxt:
            break
        wp = nxt[0]
        route.append(wp)
    return route


class CarlaSimulator(Simulator):
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.wheelbase = cfg["vehicle"]["wheelbase"]
        self.conn = CarlaConnection(cfg)
        self.carla = self.conn.carla
        self.ego: EgoVehicle | None = None
        self.rig: CameraRig | None = None

    @property
    def max_steer_deg(self) -> float | None:
        return self.ego.max_steer_deg if self.ego else None

    def reset(self) -> Frame:
        c = self.cfg["carla"]
        points = self.conn.map.get_spawn_points()
        start = points[c["spawn_point"] % len(points)]
        end = points[c["destination_point"] % len(points)]
        self.ego = EgoVehicle(self.carla, self.conn.world, c["vehicle_blueprint"], start)
        self.rig = CameraRig(self.carla, self.conn.world, self.ego.actor, self.cfg["cameras"], self.wheelbase)
        self.route = plan_route(self.conn.map, start.location, end.location)
        self.route_xy = np.array([carla_to_rh(w.transform.location.x, w.transform.location.y) for w in self.route])
        self.route_idx = 0
        self.conn.spawn_traffic(c["traffic_vehicles"], c["traffic_walkers"], avoid_location=start.location)
        self.t = 0.0
        log.info("route: %d waypoints (%.0f m)", len(self.route), len(self.route) * ROUTE_STEP)
        return self._frame(self.conn.tick())

    def step(self, control: Control) -> Frame:
        self.ego.apply(control)
        frame_id = self.conn.tick()
        self.t += self.cfg["dt"]
        return self._frame(frame_id)

    # ----------------------------------------------------------------- helpers
    def _follow_spectator(self, tf):
        yaw = math.radians(tf.rotation.yaw)
        loc = tf.location - self.carla.Location(x=8 * math.cos(yaw), y=8 * math.sin(yaw)) + self.carla.Location(z=4)
        self.conn.world.get_spectator().set_transform(self.carla.Transform(loc, self.carla.Rotation(pitch=-15, yaw=tf.rotation.yaw)))

    def _ground_truth(self, ego_xy, ego_yaw) -> list[GroundTruthObject]:
        out = []
        for actor in self.conn.world.get_actors():
            tid = actor.type_id
            if actor.id == self.ego.actor.id or not (tid.startswith("vehicle.") or tid.startswith("walker.")):
                continue
            tf = actor.get_transform()
            p = world_to_vehicle(np.array([carla_to_rh(tf.location.x, tf.location.y)]), ego_xy, ego_yaw)[0]
            if np.hypot(*p) > 80:
                continue
            ext = actor.bounding_box.extent
            out.append(GroundTruthObject("pedestrian" if tid.startswith("walker.") else "car", float(p[0]), float(p[1]),
                                         2 * ext.x, 2 * ext.y, 2 * ext.z, carla_yaw_to_rh(tf.rotation.yaw) - ego_yaw))
        return out

    def _frame(self, frame_id: int) -> Frame:
        images = self.rig.read(frame_id)
        tf = self.ego.actor.get_transform()
        self._follow_spectator(tf)
        ego_xy, ego_yaw = rear_axle_pose(tf, self.wheelbase)

        # Advance along the route (search a window ahead of the last index).
        win = self.route_xy[self.route_idx:self.route_idx + 30]
        if len(win):
            self.route_idx += int(np.argmin(np.hypot(*(win - ego_xy).T)))
        ahead = self.route_xy[self.route_idx:self.route_idx + 30]
        if len(ahead) < 2:
            ahead = self.route_xy[-2:]
        route = world_to_vehicle(ahead, ego_xy, ego_yaw)

        wp = self.conn.map.get_waypoint(tf.location)
        upcoming = self.route[self.route_idx:self.route_idx + 8]
        at_junction = wp.is_junction or any(w.is_junction for w in upcoming)
        left_wp = wp.get_left_lane()
        left_ok = bool(left_wp is not None and left_wp.lane_type == self.carla.LaneType.Driving
                       and left_wp.lane_id * wp.lane_id > 0
                       and (wp.lane_change & self.carla.LaneChange.Left))
        self._last = (ego_xy, ego_yaw)
        yaw_rate = -math.radians(self.ego.actor.get_angular_velocity().z)     # CARLA: deg/s, clockwise +
        return Frame(images=images, speed=self.ego.speed, timestamp=self.t, route=route, yaw_rate=yaw_rate,
                     stereo_pair=tuple(self.cfg["cameras"]["stereo_pair"]),
                     at_junction=at_junction, lane_width=float(wp.lane_width), left_lane_available=left_ok,
                     ground_truth=self._ground_truth(ego_xy, ego_yaw),
                     route_finished=self.route_idx >= len(self.route) - 6)

    def status(self) -> dict:
        ego_xy, ego_yaw = self._last
        i = self.route_idx
        lateral = 0.0
        if i + 1 < len(self.route_xy):
            d = self.route_xy[i + 1] - self.route_xy[i]
            r = ego_xy - self.route_xy[i]
            lateral = float((d[0] * r[1] - d[1] * r[0]) / (np.linalg.norm(d) + 1e-9))
        return {"collisions": self.ego.collisions, "lane_invasions": self.ego.lane_invasions,
                "lateral_error": lateral, "progress": i * ROUTE_STEP,
                "route_length": len(self.route) * ROUTE_STEP,
                "in_junction": bool(self.route[min(i, len(self.route) - 1)].is_junction)}

    def close(self):
        for thing in (self.rig, self.ego):
            if thing is not None:
                try:
                    thing.destroy()
                except RuntimeError:
                    pass
        self.conn.close()
