"""Rule-based driving brain, working on the tracked 3D world model.

Priority (most restrictive speed wins):
    pedestrian in path, or predicted to walk into it -> stop / emergency brake
    stop sign            -> stop, wait, go
    stopped obstacle     -> overtake through the left lane if it is free now and stays free
                            (rear-facing cameras + motion prediction), otherwise stop behind it
    moving vehicle ahead, or predicted to cut in -> adaptive cruise control (keep a time gap)
    intersection         -> slow down and follow the route (turn)
    lane lost            -> slow down and follow the route
    otherwise            -> lane following at cruise speed (limited by curvature)
"""
from __future__ import annotations

import math

import numpy as np

from planning.path_planner import offset_path, path_curvature
from planning.prediction import corridor_intrusion, lane_change_safe, path_y
from utils.types import Action, Decision, TrackedObject, WorldModel

HALF_LENGTH = {"car": 2.25, "pedestrian": 0.3, "cyclist": 0.9}
VEHICLES = ("car", "cyclist")


class DecisionEngine:
    def __init__(self, cfg: dict):
        p = cfg["planning"]
        v = cfg["vehicle"]
        self.p = p
        self.dt = cfg["dt"]
        self.front = v["wheelbase"] / 2 + v["length"] / 2      # rear axle -> front bumper
        self.max_lat_acc = 2.0
        self.comfort_decel = 3.0
        self.ped_horizon = p.get("pedestrian_prediction_horizon", 3.0)
        self.cut_in_horizon = p.get("cut_in_prediction_horizon", 1.5)
        self.t = 0.0
        # Overtake state machine: none -> out -> back -> none
        self.avoid_state = "none"
        self.pass_remaining = 0.0
        self.back_timer = 0.0
        self.lead_stopped_for = 0.0
        self.waiting_reason = ""
        # Lateral offset the path planner is actually applying (set by the autopilot each tick).
        self.current_offset: float | None = None
        # Pedestrian stop latch: stay stopped until the path has been clear for a moment.
        self.ped_latched = False
        self.ped_clear_time = 0.0
        # Stop-sign handling
        self.sign_stopped_for = 0.0
        self.sign_cooldown = 0.0

    # ----------------------------------------------------------------- helpers
    def gap(self, obj: TrackedObject) -> float:
        """Bumper-to-object distance. Stereo box centres and mono box bottoms both measure the
        object's visible (near) face, so no half-length is subtracted."""
        return obj.x - self.front

    def _in_corridor(self, obj: TrackedObject, path: np.ndarray, half_width: float) -> bool:
        return 0.0 < obj.x < self.p["obstacle_detect_distance"] + self.front and \
            abs(obj.y - float(path_y(path, obj.x))) < half_width

    def _base_path(self, world: WorldModel) -> np.ndarray:
        lane = world.lane
        if world.at_junction or lane is None or lane.confidence < 0.5 or self.avoid_state != "none":
            return world.route
        xs = np.arange(0.0, 40.0, 1.0)
        return np.stack([xs, lane.y_at(xs)], axis=1)

    @staticmethod
    def _lead_speed(obj: TrackedObject) -> float | None:
        """Forward speed of a tracked object over the ground (None until the tracker is sure)."""
        return max(0.0, obj.vx) if obj.velocity_valid else None

    # ----------------------------------------------------------------- main entry
    def decide(self, world: WorldModel, route_finished: bool = False) -> Decision:
        dt = self.dt
        self.t += dt
        p, v = self.p, world.speed
        lw = world.lane_width
        if route_finished:
            return Decision(Action.ROUTE_DONE, 0.0, use_route=True, reason="destination reached")

        base = self._base_path(world)
        offset = lw if self.avoid_state == "out" else 0.0
        my_path = offset_path(base, offset)
        half = p["path_half_width"]
        use_route = world.at_junction or world.lane is None or world.lane.confidence < 0.5 or self.avoid_state != "none"

        lane_known = world.lane is not None and world.lane.confidence >= 0.5
        cruise = p["cruise_speed"]
        k = path_curvature(world.route if not lane_known else base)
        candidates: list[tuple[float, Action, str, bool]] = [
            (cruise, Action.LANE_FOLLOW, "lane following", False),
            (math.sqrt(self.max_lat_acc / max(k, 1e-4)), Action.LANE_FOLLOW, "curve speed limit", False),
        ]
        if world.at_junction:
            candidates.append((p["junction_speed"], Action.TURN, "intersection: following route", False))
        elif not lane_known:
            candidates.append((p["lane_lost_speed"], Action.SLOW_DOWN, "lane markings lost", False))

        # --- pedestrians: in the corridor now, or predicted to walk into it within a few seconds.
        # Once stopped we wait until the (wider) corridor has been clear for 1 s before driving on.
        margin = 1.6 if self.ped_latched else 1.1
        ped_gaps: list[tuple[float, str]] = []
        for o in world.objects:
            if o.cls != "pedestrian":
                continue
            if self._in_corridor(o, my_path, half + margin):
                ped_gaps.append((self.gap(o), "in path"))
                continue
            hit = corridor_intrusion(o, my_path, half + 0.6, self.ped_horizon)
            if hit is not None and hit[1] - self.front < p["obstacle_detect_distance"]:
                ped_gaps.append((hit[1] - self.front, f"will cross in {hit[0]:.1f} s"))
        if ped_gaps:
            self.ped_clear_time = 0.0
        elif self.ped_latched:
            self.ped_clear_time += dt
            if self.ped_clear_time > 1.0:
                self.ped_latched = False
            else:
                candidates.append((0.0, Action.EMERGENCY_STOP, "waiting for pedestrian to clear", False))
        if ped_gaps:
            g, why = min(ped_gaps)
            stop_d = p["pedestrian_stop_distance"]
            if g < stop_d:
                self.ped_latched = True
                hard = g < v * v / (2 * 6.0) + 2.0
                candidates.append((0.0, Action.EMERGENCY_STOP, f"pedestrian {why} {g:.1f} m", hard))
            else:
                vt = math.sqrt(max(0.0, 2 * self.comfort_decel * (g - stop_d)))
                candidates.append((vt, Action.SLOW_DOWN, f"pedestrian {why} {g:.1f} m", False))

        # --- stop signs (YOLO only; the offline simulator has none)
        self.sign_cooldown = max(0.0, self.sign_cooldown - dt)
        signs = [o for o in world.objects if o.cls == "stop_sign" and 2.0 < o.x < 20.0 and abs(o.y) < 8.0]
        if signs and self.sign_cooldown <= 0:
            if v < 0.3:
                self.sign_stopped_for += dt
            if self.sign_stopped_for >= p["stop_sign_wait"]:
                self.sign_cooldown, self.sign_stopped_for = 10.0, 0.0
            else:
                g = min(o.x for o in signs) - self.front
                vt = math.sqrt(max(0.0, 2 * self.comfort_decel * max(0.0, g - 1.0)))
                candidates.append((vt, Action.STOP_SIGN, "stop sign", False))

        # --- overtaking state machine
        if self.avoid_state == "out":
            in_my_lane = [o for o in world.objects if o.cls in VEHICLES and self._in_corridor(o, world.route, half)]
            if in_my_lane:
                o = min(in_my_lane, key=lambda o: o.x)
                self.pass_remaining = max(self.pass_remaining, o.x + 2 * HALF_LENGTH.get(o.cls, 1.0) + 7.0)
            else:
                self.pass_remaining -= v * dt
            committed = (self.current_offset or 0.0) > 1.2
            safe, why = lane_change_safe(world.objects, world.route, lw, v, self.front)
            if not committed and not safe:
                self.avoid_state = "none"                       # abort before crossing the line
                self.waiting_reason = why
            elif self.pass_remaining <= 0:
                self.avoid_state, self.back_timer = "back", 0.0
        elif self.avoid_state == "back":
            self.back_timer += dt
            if self.back_timer > 4.0:
                self.avoid_state = "none"
        if self.avoid_state != "none":
            candidates.append((min(cruise, 7.0), Action.AVOID_OBSTACLE,
                               "overtaking" if self.avoid_state == "out" else "returning to lane", False))

        # --- mid lane change: the car is still partly in front of whatever it is passing.
        # Check a car-width corridor around where we actually are and hold back until clear sideways.
        if self.avoid_state != "none":
            now_path = offset_path(base, self.current_offset if self.current_offset is not None else offset)
            blocking = [o for o in world.objects if o.cls in VEHICLES and self._in_corridor(o, now_path, half + 0.6)]
            if blocking:
                g = min(self.gap(o) for o in blocking)
                vt = math.sqrt(max(0.0, 2 * self.comfort_decel * (g - 2.5)))
                if g > 2.0:
                    vt = max(vt, 1.5)        # keep creeping: a stopped car cannot steer out sideways
                candidates.append((vt, Action.AVOID_OBSTACLE, f"changing lane, obstacle {g:.1f} m", g < 1.0))

        # --- vehicles predicted to cut into my lane: keep a gap to where they will be
        if self.avoid_state != "out":
            for o in world.objects:
                if o.cls not in VEHICLES or o.x <= self.front or self._in_corridor(o, my_path, half):
                    continue
                side = o.y - float(path_y(my_path, o.x))
                if not o.velocity_valid or abs(o.vy) < 0.5 or np.sign(o.vy) == np.sign(side):
                    continue                                    # not clearly moving towards my lane
                hit = corridor_intrusion(o, my_path, half, self.cut_in_horizon)
                if hit is not None:
                    g = hit[1] - self.front
                    desired = p["min_follow_gap"] + p["time_gap"] * v
                    vt = max(0.0, max(0.0, o.vx) + 0.35 * (g - desired))
                    candidates.append((vt, Action.FOLLOW_VEHICLE, f"car cutting in, {g:.1f} m", False))

        # --- vehicles ahead in my path
        cars = []
        if self.avoid_state != "out":
            cars = [(o, self.gap(o)) for o in world.objects
                    if o.cls in VEHICLES and self._in_corridor(o, my_path, half)]
        if cars:
            lead, g = min(cars, key=lambda c: c[1])
            lead_v = self._lead_speed(lead)
            est_v = v if lead_v is None else lead_v
            # Only call it "stopped" once the tracker has said so for half a second.
            self.lead_stopped_for = self.lead_stopped_for + dt if (lead_v is not None and lead_v < 1.0) else 0.0
            stopped = self.lead_stopped_for >= 0.5
            safe, why = lane_change_safe(world.objects, world.route, lw, v, self.front)
            if (stopped and self.avoid_state == "none" and g < p["avoid_trigger_distance"]
                    and world.left_lane_available and not world.at_junction and safe):
                self.avoid_state = "out"
                self.waiting_reason = ""
                self.pass_remaining = lead.x + 2 * HALF_LENGTH.get(lead.cls, 1.0) + 7.0
                candidates.append((min(cruise, 7.0), Action.AVOID_OBSTACLE, "stopped vehicle: overtaking", False))
                use_route = True
            else:
                desired = p["min_follow_gap"] + p["time_gap"] * v
                vt = max(0.0, est_v + 0.35 * (g - desired))
                if g < p["min_follow_gap"]:
                    vt = 0.0
                hard = g < max(1.5, v * v / (2 * 6.0))
                if stopped and not safe:
                    reason = f"blocked, waiting: {why}"
                else:
                    reason = f"vehicle ahead {g:.1f} m (~{est_v:.1f} m/s)"
                candidates.append((vt, Action.FOLLOW_VEHICLE, reason, hard))
        else:
            self.lead_stopped_for = 0.0

        target, action, reason, hard = min(candidates, key=lambda c: (c[0], -int(c[3])))
        if action == Action.LANE_FOLLOW and world.at_junction:
            action, reason = Action.TURN, "intersection: following route"
        elif action == Action.LANE_FOLLOW and not lane_known:
            action, reason = Action.SLOW_DOWN, "lane markings lost"
        return Decision(action, float(target), lateral_offset=lw if self.avoid_state == "out" else 0.0,
                        use_route=use_route, hard_brake=hard, reason=reason)
