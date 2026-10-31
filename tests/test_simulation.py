import numpy as np
import pytest

from main import run
from simulation.offline_sim import SCENARIO_NAMES, OfflineSimulator, build_scenario, car
from utils.types import Control


def test_all_scenarios_build():
    for name in SCENARIO_NAMES:
        road, _ = build_scenario(name)
        assert road.length > 100


def test_turn_scenario_turns_ninety_degrees():
    road, _ = build_scenario("turn")
    assert np.isclose(road.yaw[-1], np.pi / 2, atol=0.01)
    assert road.junction.any()


def test_frame_contents(cfg):
    sim = OfflineSimulator(cfg, "obstacle")
    f = sim.reset()
    h, w = cfg["cameras"]["height"], cfg["cameras"]["width"]
    assert f.left.shape == (h, w, 3) and f.right.shape == (h, w, 3)
    assert not np.array_equal(f.left, f.right)
    assert f.route.shape[1] == 2 and f.route[0, 0] < 1.0
    assert f.ground_truth[0].cls == "car"


def test_straight_driving_and_braking(cfg):
    sim = OfflineSimulator(cfg, "lane_keep")
    sim.reset()
    for _ in range(60):
        f = sim.step(Control(throttle=0.6))
    assert f.speed > 4.0
    assert abs(sim.status()["lateral_error"]) < 0.01
    for _ in range(60):
        f = sim.step(Control(brake=1.0))
    assert f.speed == 0.0


def test_collision_is_counted(cfg):
    sim = OfflineSimulator(cfg, "obstacle")
    sim.reset()
    sim.actors = [car(15.0, 0.0)]
    for _ in range(200):
        sim.step(Control(throttle=0.7))
    assert sim.status()["collisions"] == 1


# ---------------------------------------------------------------- closed-loop driving
@pytest.mark.slow
def test_lane_keeping(cfg):
    m = run(cfg, "sim", "curve")
    assert m["collisions"] == 0
    assert m["completion"] > 0.9
    assert m["lane_error_mean_m"] < 0.25
    assert m["lane_detection_rate"] > 0.9


@pytest.mark.slow
def test_pedestrian_stop(cfg):
    m = run(cfg, "sim", "pedestrian")
    assert m["collisions"] == 0
    assert m["stopped_for_pedestrian"]
    assert m["completion"] > 0.9


@pytest.mark.slow
def test_obstacle_avoidance(cfg):
    m = run(cfg, "sim", "obstacle")
    assert m["collisions"] == 0
    assert m["overtakes"] == 1
    assert m["completion"] > 0.9


@pytest.mark.slow
def test_vehicle_following(cfg):
    m = run(cfg, "sim", "follow", max_steps=500)
    assert m["collisions"] == 0
    assert m["actions"].get("FOLLOW_VEHICLE", 0) > 200
    assert m["overtakes"] == 0


@pytest.mark.slow
def test_intersection_turn(cfg):
    m = run(cfg, "sim", "turn")
    assert m["collisions"] == 0
    assert m["actions"].get("TURN", 0) > 50
    assert m["completion"] > 0.9
    assert m["lane_error_mean_m"] < 0.3
