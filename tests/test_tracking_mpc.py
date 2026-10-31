import math

import numpy as np

from control.mpc import MPCSteering
from fusion.camera_fusion import Measurement
from fusion.tracker import MultiObjectTracker
from utils.types import TrackedObject


def meas(x, y, cls="car", sigma=0.3):
    return Measurement(TrackedObject(cls, x, y, float(np.hypot(x, y)), 0.9, "stereo"), np.eye(2) * sigma ** 2)


def line(y):
    return np.stack([np.arange(0, 40.0), np.full(40, float(y))], 1)


def test_tracker_estimates_velocity_of_moving_object():
    rng = np.random.default_rng(0)
    trk = MultiObjectTracker(0.05)
    ego_v = 5.0
    x_rel = 20.0                                              # lead does 8 m/s, we do 5 m/s
    for _ in range(60):
        out = trk.step([meas(x_rel + rng.normal(0, 0.3), rng.normal(0, 0.2))], ego_v, 0.0)
        x_rel += (8.0 - ego_v) * 0.05
    assert len(out) == 1 and out[0].velocity_valid
    assert abs(out[0].vx - 8.0) < 0.6 and abs(out[0].vy) < 0.5
    assert out[0].predicted.shape[1] == 3


def test_tracker_static_object_while_ego_turns():
    """Ego-motion compensation: a parked car stays at zero ground speed while we drive and turn."""
    trk = MultiObjectTracker(0.05)
    ego = np.array([0.0, 0.0, 0.0])
    p_world = np.array([25.0, 3.0])
    v, r, dt = 6.0, 0.2, 0.05
    for _ in range(50):
        c, s = math.cos(ego[2]), math.sin(ego[2])
        d = p_world - ego[:2]
        rel = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])
        out = trk.step([meas(*rel)], v, r)
        ego[0] += v * dt * math.cos(ego[2])
        ego[1] += v * dt * math.sin(ego[2])
        ego[2] += r * dt
    assert out[0].velocity_valid and out[0].speed < 0.6


def test_tracker_keeps_identity_and_drops_lost_tracks():
    trk = MultiObjectTracker(0.05, max_unseen=0.5)
    for _ in range(10):
        out = trk.step([meas(20.0, 0.0), meas(10.0, 3.5, "pedestrian")], 0.0, 0.0)
    ids = {o.cls: o.track_id for o in out}
    for _ in range(5):
        out = trk.step([meas(20.0, 0.0)], 0.0, 0.0)
    assert {o.cls: o.track_id for o in out}["car"] == ids["car"]
    for _ in range(12):
        out = trk.step([meas(20.0, 0.0)], 0.0, 0.0)
    assert [o.cls for o in out] == ["car"]


def test_mpc_steering_signs():
    assert abs(MPCSteering(2.9, 35).step(line(0.0), 8.0, 0.05)) < 0.02
    assert MPCSteering(2.9, 35).step(line(1.5), 8.0, 0.05) < 0      # target to the left -> steer left
    assert MPCSteering(2.9, 35).step(line(-1.5), 8.0, 0.05) > 0


def test_mpc_tracks_a_circle_in_closed_loop():
    """Drive a bicycle model around a 30 m radius bend with MPC; it should stay close to the path."""
    R, L, dt, v = 30.0, 2.9, 0.05, 8.0
    mpc = MPCSteering(L, 35)
    x, y, yaw, wheel = 0.0, 0.0, 0.0, 0.0
    errs = []
    for _ in range(200):
        # Path: circle centred at (0, R) in world coordinates, starting at the origin heading +x.
        a0 = math.atan2(x, R - y)
        angles = a0 + np.linspace(0, 1.5, 60)
        world = np.stack([R * np.sin(angles), R - R * np.cos(angles)], 1)
        c, s = math.cos(yaw), math.sin(yaw)
        d = world - [x, y]
        path = np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], 1)
        steer = mpc.step(path, v, dt)
        cmd = -steer * math.radians(35)
        wheel += (cmd - wheel) * min(1.0, dt / 0.1)
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw += v / L * math.tan(wheel) * dt
        errs.append(abs(math.hypot(x, y - R) - R))
    assert max(errs[40:]) < 0.3
