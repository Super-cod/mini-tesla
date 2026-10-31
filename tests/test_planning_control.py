import numpy as np

from control.pid import PID
from control.speed_controller import SpeedController
from control.steering_pid import SteeringController
from planning.decision_engine import DecisionEngine
from fusion.tracker import predict_trajectory
from planning.path_planner import PathPlanner, lookahead_point, offset_path
from planning.prediction import corridor_intrusion, lane_change_safe
from utils.types import Action, Decision, LaneEstimate, TrackedObject, WorldModel

STRAIGHT = np.stack([np.arange(0, 60.0), np.zeros(60)], axis=1)


def world(objects=(), speed=8.0, lane=True, junction=False, left_ok=True):
    ln = LaneEstimate(np.array([0.0, 0.0, 0.0]), 1.0, width=3.5) if lane else None
    return WorldModel(lane=ln, objects=list(objects), speed=speed, route=STRAIGHT, at_junction=junction,
                      lane_width=3.5, left_lane_available=left_ok)


def obj(cls, x, y=0.0, vx=None, vy=0.0):
    """A tracked object; pass vx to give it a (trusted) velocity and a 3 s prediction."""
    o = TrackedObject(cls, x, y, float(np.hypot(x, y)), 0.9, "stereo")
    if vx is not None:
        o.vx, o.vy, o.velocity_valid = vx, vy, True
        o.predicted = predict_trajectory(x, y, vx, vy)
    return o


# ---------------------------------------------------------------- PID / controllers
def test_pid_drives_first_order_plant_to_setpoint():
    pid = PID(0.8, 0.5, 0.0, limit=1.0)
    y = 0.0
    for _ in range(400):
        u = pid.step(1.0 - y, 0.05)
        y += (u * 2.0 - y) * 0.05
    assert abs(y - 1.0) < 0.02


def test_pid_anti_windup_limits_integral():
    pid = PID(1.0, 1.0, 0.0, limit=1.0)
    for _ in range(1000):
        pid.step(10.0, 0.05)
    assert pid.integral <= 1.0 + 1e-9                       # stops integrating while saturated


def test_steering_integral_is_capped(cfg):
    sc = SteeringController(cfg["control"]["steering"], 2.9, 35.0)
    for _ in range(2000):
        sc.step(np.array([10.0, 3.0]), 0.05)
    assert sc.pid.ki * sc.pid.integral <= 0.15 + 1e-9


def test_steering_sign_convention(cfg):
    sc = SteeringController(cfg["control"]["steering"], 2.9, 35.0)
    assert sc.step(np.array([10.0, 2.0]), 0.05) < 0       # target on the left -> steer left (negative in CARLA)
    sc.reset()
    assert sc.step(np.array([10.0, -2.0]), 0.05) > 0
    sc.reset()
    assert abs(sc.step(np.array([10.0, 0.0]), 0.05)) < 1e-9


def test_speed_controller_throttle_and_brake(cfg):
    spd = SpeedController(cfg["control"]["speed"])
    th, br = spd.step(8.0, 2.0, 0.05)
    assert th > 0 and br == 0
    th, br = spd.step(2.0, 8.0, 0.05)
    assert th == 0 and br > 0
    assert spd.step(5.0, 5.0, 0.05, hard_brake=True) == (0.0, 1.0)
    assert spd.step(0.0, 0.1, 0.05)[1] > 0                  # holds the car at standstill


# ---------------------------------------------------------------- path planner
def test_offset_path_shifts_left():
    shifted = offset_path(STRAIGHT, 3.5)
    np.testing.assert_allclose(shifted[:, 1], 3.5)


def test_lookahead_point_interpolates():
    p = lookahead_point(STRAIGHT, 7.5)
    np.testing.assert_allclose(p, [7.5, 0.0], atol=1e-9)


def test_planner_ramps_lane_change_offset(cfg):
    pl = PathPlanner(cfg)
    w = world()
    d = Decision(Action.AVOID_OBSTACLE, 7.0, lateral_offset=3.5, use_route=True)
    path = pl.reference_path(w, d, 0.05)
    assert 0 < path[10, 1] < 0.1                            # moves gradually, not in one jump
    for _ in range(100):
        path = pl.reference_path(w, d, 0.05)
    np.testing.assert_allclose(path[10, 1], 3.5, atol=1e-6)


# ---------------------------------------------------------------- decision engine
def test_clear_road_is_lane_following(cfg):
    d = DecisionEngine(cfg).decide(world())
    assert d.action == Action.LANE_FOLLOW
    assert d.target_speed == cfg["planning"]["cruise_speed"]
    assert not d.use_route


def test_pedestrian_in_path_stops_car(cfg):
    d = DecisionEngine(cfg).decide(world([obj("pedestrian", 12.0, 0.5)]))
    assert d.action == Action.EMERGENCY_STOP
    assert d.target_speed == 0.0


def test_pedestrian_on_sidewalk_is_ignored(cfg):
    d = DecisionEngine(cfg).decide(world([obj("pedestrian", 12.0, -6.0)]))
    assert d.action == Action.LANE_FOLLOW


def test_pedestrian_stop_waits_until_clear(cfg):
    eng = DecisionEngine(cfg)
    eng.decide(world([obj("pedestrian", 10.0, 0.0)], speed=0.0))
    d = eng.decide(world([], speed=0.0))                     # just left the corridor
    assert d.action == Action.EMERGENCY_STOP
    for _ in range(25):
        d = eng.decide(world([], speed=0.0))
    assert d.action == Action.LANE_FOLLOW


def test_moving_lead_vehicle_is_followed(cfg):
    eng = DecisionEngine(cfg)
    gap0 = 20.0
    for i in range(80):                                     # lead does 4 m/s, we do 6 m/s
        d = eng.decide(world([obj("car", gap0 + 6.0 - 2.0 * i * 0.05 + 3.8, vx=4.0)], speed=6.0))
    assert d.action == Action.FOLLOW_VEHICLE
    assert d.target_speed < 6.0
    assert eng.avoid_state == "none"


def test_stopped_vehicle_is_overtaken_when_left_lane_free(cfg):
    eng = DecisionEngine(cfg)
    actions = []
    for i in range(80):
        x = 40.0 - 6.0 * i * 0.05                            # stationary car, we approach at 6 m/s
        if x < 15.0:
            break
        actions.append(eng.decide(world([obj("car", x, vx=0.0)], speed=6.0)).action)
    assert Action.AVOID_OBSTACLE in actions
    d = eng.decide(world([obj("car", 15.0, -3.5)], speed=6.0))
    assert d.lateral_offset == 3.5 and d.use_route


def test_stopped_vehicle_not_overtaken_without_left_lane(cfg):
    eng = DecisionEngine(cfg)
    for i in range(80):
        d = eng.decide(world([obj("car", 40.0 - 6.0 * i * 0.05, vx=0.0)], speed=6.0, left_ok=False))
    assert d.action == Action.FOLLOW_VEHICLE
    assert d.target_speed < 1.0


def test_junction_turns_by_route(cfg):
    d = DecisionEngine(cfg).decide(world(junction=True, lane=False))
    assert d.action == Action.TURN and d.use_route
    assert d.target_speed <= cfg["planning"]["junction_speed"]


def test_lost_lane_slows_down(cfg):
    d = DecisionEngine(cfg).decide(world(lane=False))
    assert d.action == Action.SLOW_DOWN and d.use_route
    assert d.target_speed == cfg["planning"]["lane_lost_speed"]


def test_stop_sign_stops_then_continues(cfg):
    eng = DecisionEngine(cfg)
    sign = obj("stop_sign", 4.0 + eng.front, -3.0)
    d = eng.decide(world([sign], speed=0.0))
    assert d.action == Action.STOP_SIGN
    for _ in range(int(cfg["planning"]["stop_sign_wait"] / cfg["dt"]) + 2):
        d = eng.decide(world([sign], speed=0.0))
    assert d.action == Action.LANE_FOLLOW


def test_route_done(cfg):
    d = DecisionEngine(cfg).decide(world(), route_finished=True)
    assert d.action == Action.ROUTE_DONE and d.target_speed == 0.0


def test_stopped_vehicle_not_overtaken_while_car_approaches_from_behind(cfg):
    """A fast car in the left lane behind us (seen by the rear cameras) blocks the overtake."""
    eng = DecisionEngine(cfg)
    for i in range(40):
        behind = obj("car", -25.0 + 8.0 * i * 0.05, 3.5, vx=12.0)
        d = eng.decide(world([obj("car", 16.0, vx=0.0), behind], speed=4.0))
        assert d.action != Action.AVOID_OBSTACLE
    assert "waiting" in d.reason


def test_unknown_speed_car_behind_blocks_lane_change(cfg):
    safe, why = lane_change_safe([obj("car", -30.0, 3.5)], STRAIGHT, 3.5, 6.0, 3.8)
    assert not safe and "unknown" in why
    safe, _ = lane_change_safe([obj("car", -30.0, 3.5, vx=2.0)], STRAIGHT, 3.5, 6.0, 3.8)
    assert safe                                              # slower car behind: no threat


def test_pedestrian_predicted_to_cross_slows_car_early(cfg):
    """Still on the pavement, but walking towards the road -> react before it is in the path."""
    walking = obj("pedestrian", 22.0, -4.5, vx=0.0, vy=1.4)
    standing = obj("pedestrian", 22.0, -4.5, vx=0.0, vy=0.0)
    assert corridor_intrusion(walking, STRAIGHT, 2.1) is not None
    assert corridor_intrusion(standing, STRAIGHT, 2.1) is None
    d = DecisionEngine(cfg).decide(world([walking]))
    assert d.target_speed < cfg["planning"]["cruise_speed"] and "will cross" in d.reason
    d = DecisionEngine(cfg).decide(world([standing]))
    assert d.action == Action.LANE_FOLLOW


def test_car_cutting_in_is_reacted_to(cfg):
    cutting = obj("car", 15.0, 3.5, vx=5.0, vy=-2.0)
    d = DecisionEngine(cfg).decide(world([cutting], speed=8.0))
    assert d.action == Action.FOLLOW_VEHICLE and "cutting in" in d.reason
    parallel = obj("car", 15.0, 3.5, vx=5.0, vy=0.0)
    assert DecisionEngine(cfg).decide(world([parallel], speed=8.0)).action == Action.LANE_FOLLOW
