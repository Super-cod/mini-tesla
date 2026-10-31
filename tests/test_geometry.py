import numpy as np

from simulation.camera_setup import camera_transforms
from simulation.carla_connection import carla_to_rh, carla_yaw_to_rh
from utils.geometry import PinholeCamera, make_stereo_pair, world_to_vehicle


def test_projection_round_trip_on_ground(cfg):
    cam, _ = make_stereo_pair(cfg["cameras"])
    pts = np.array([[8.0, 1.5, 0.0], [20.0, -2.0, 0.0], [35.0, 0.3, 0.0]])
    uv, depth = cam.project(pts)
    assert (depth > 0).all()
    back, ok = cam.pixel_to_ground(uv)
    assert ok.all()
    np.testing.assert_allclose(back, pts[:, :2], atol=1e-6)


def test_point_straight_ahead_projects_left_of_centre_in_right_camera(cfg):
    left, right = make_stereo_pair(cfg["cameras"])
    p = np.array([[20.0, 0.0, 1.6]])
    ul = left.project(p)[0][0, 0]
    ur = right.project(p)[0][0, 0]
    # The left camera sits to the left, so the point appears further right in its image.
    assert ul > left.cx > ur
    disparity = ul - ur
    depth = left.fx * cfg["cameras"]["baseline"] / disparity
    # Stereo gives depth along the (pitched) optical axis.
    assert abs(depth - left.to_camera(p)[0, 2]) < 1e-6


def test_pixel_depth_back_projection():
    cam = PinholeCamera(640, 360, 90.0, np.array([1.5, 0.27, 1.6]), pitch_deg=-8)
    p = np.array([[17.0, -2.0, 0.8]])
    uv, z = cam.project(p)
    q = cam.pixel_depth_to_vehicle(uv[0, 0], uv[0, 1], z[0])
    np.testing.assert_allclose(q, p[0], atol=1e-6)


def test_sky_pixels_do_not_hit_ground():
    cam = PinholeCamera(640, 360, 90.0, np.array([1.5, 0.0, 1.6]), pitch_deg=-8)
    _, ok = cam.pixel_to_ground(np.array([[320, 5]]))
    assert not ok[0]


def test_world_to_vehicle_rotation():
    rel = world_to_vehicle(np.array([[0.0, 10.0]]), np.array([0.0, 0.0]), np.pi / 2)
    np.testing.assert_allclose(rel, [[10.0, 0.0]], atol=1e-9)


def test_carla_conversions(cfg):
    assert carla_to_rh(3.0, 2.0) == (3.0, -2.0)
    assert np.isclose(carla_yaw_to_rh(90.0), -np.pi / 2)
    tfs = camera_transforms(cfg["cameras"], cfg["vehicle"]["wheelbase"])
    # Left camera is at negative y in CARLA's left-handed frame.
    assert tfs["left"][1] < 0 < tfs["right"][1]
    assert np.isclose(tfs["right"][1] - tfs["left"][1], cfg["cameras"]["baseline"])
