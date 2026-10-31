import numpy as np
import pytest
import torch

from fusion.camera_fusion import CameraFusion, fuse_lanes
from models.lane_cnn import LaneCNN
from perception.lane_detection import LaneDetector
from perception.object_detection import GroundTruthDetector
from perception.stereo_depth import DenseStereo, depth_from_disparity, match_detections
from simulation.offline_sim import OfflineSimulator, car, pedestrian
from utils.config import deep_merge, resolve_path
from utils.types import Detection, LaneEstimate


def place(sim, s, d=0.0, dyaw=0.0):
    _, _, yaw = sim.road.pose(s)
    p = sim.road.point(s, d)
    sim.ego = np.array([p[0], p[1], yaw + dyaw])
    sim.s = s
    return sim._frame()


@pytest.mark.parametrize("method", ["classical", "cnn"])
@pytest.mark.parametrize("offset", [-0.8, 0.0, 0.6])
def test_lane_offset_is_measured(cfg, method, offset):
    if method == "cnn" and not resolve_path(cfg["perception"]["lane_weights"]).exists():
        pytest.skip("lane CNN weights not trained")
    cfg = deep_merge(cfg, {"perception": {"lane_method": method}})
    sim = OfflineSimulator(cfg, "lane_keep")
    sim.reset()
    frame = place(sim, 20.0, offset)
    det = LaneDetector(cfg)
    assert det.method == method
    for img, cam in ((frame.left, sim.cam_left), (frame.right, sim.cam_right)):
        lane, mask = det.detect(img, cam)
        assert lane is not None and lane.confidence >= 0.6
        # Car is `offset` to the left of the centre, so the centre is at -offset.
        assert abs(lane.offset - (-offset)) < 0.2
        assert abs(lane.heading) < 0.05
        assert mask.shape == img.shape[:2]


def test_lane_heading_follows_curve(cfg):
    sim = OfflineSimulator(cfg, "curve")
    sim.reset()
    frame = place(sim, 50.0)                     # inside a left curve (radius 50 m)
    lane, _ = LaneDetector(deep_merge(cfg, {"perception": {"lane_method": "classical"}})).detect(frame.left, sim.cam_left)
    assert lane is not None
    assert lane.coeffs[2] > 0.004                # bends to the left: y = c2 x^2 with c2 ~ 1/(2R)


def test_lane_cnn_shapes():
    model = LaneCNN()
    out = model(torch.zeros(2, 3, 128, 256))
    assert out.shape == (2, 1, 128, 256)


def test_fuse_lanes_weights_by_confidence():
    a = LaneEstimate(np.array([1.0, 0, 0]), 1.0, width=3.5)
    b = LaneEstimate(np.array([0.0, 0, 0]), 0.5)
    f = fuse_lanes([a, b, None])
    assert np.isclose(f.coeffs[0], 2 / 3)
    assert fuse_lanes([None, None]) is None


def test_depth_from_disparity():
    assert np.isclose(depth_from_disparity(10.0, 320.0, 0.5), 16.0)
    assert depth_from_disparity(0.0, 320.0, 0.5) == float("inf")


def test_match_detections_pairs_same_object():
    left = [Detection("car", (300, 150, 340, 180)), Detection("pedestrian", (100, 140, 110, 175))]
    right = [Detection("pedestrian", (92, 140, 102, 175)), Detection("car", (290, 150, 330, 180))]
    m = sorted(match_detections(left, right, 360))
    assert m == [(0, 1, 10.0), (1, 0, 8.0)]


def test_match_rejects_negative_disparity():
    left = [Detection("car", (300, 150, 340, 180))]
    right = [Detection("car", (310, 150, 350, 180))]
    assert match_detections(left, right, 360) == []


@pytest.mark.parametrize("dist", [10.0, 20.0, 35.0])
def test_stereo_fusion_locates_car(cfg, dist):
    sim = OfflineSimulator(cfg, "obstacle")
    sim.reset()
    sim.actors = [car(dist + 1.5 + 2.25, 0.0)]         # place its centre roughly `dist` ahead of the cameras
    frame = place(sim, 0.0)
    det = GroundTruthDetector(pixel_noise=0.0)
    fusion = CameraFusion(cfg, sim.cams)
    objs = fusion.locate_objects(frame, det.detect_all(frame, sim.cams))
    assert len(objs) == 1 and objs[0].source == "stereo"
    gt = frame.ground_truth[0]
    # Box-centre disparity measures (close to) the visible rear face of the car.
    assert gt.x - gt.length / 2 - 0.6 < objs[0].x < gt.x + 0.3
    assert abs(objs[0].y - gt.y) < 0.5


def test_mono_fallback_for_object_seen_by_one_camera(cfg):
    sim = OfflineSimulator(cfg, "pedestrian")
    sim.reset()
    sim.actors = [pedestrian(9.0, -4.6, -4.6, None)]
    frame = place(sim, 0.0)
    dets = GroundTruthDetector(pixel_noise=0.0).detect_all(frame, sim.cams)
    dets["front_right"] = []
    objs = CameraFusion(cfg, sim.cams).locate_objects(frame, dets)
    assert len(objs) == 1 and objs[0].source == "mono"
    assert abs(objs[0].x - frame.ground_truth[0].x) < 1.5


def test_dense_stereo_disparity_on_render(cfg):
    sim = OfflineSimulator(cfg, "obstacle")
    sim.reset()
    sim.actors = [car(16.0, 0.0)]
    frame = place(sim, 0.0)
    disp = DenseStereo().disparity(frame.left, frame.right)
    assert disp.shape == frame.left.shape[:2]
    assert (disp > 0).mean() > 0.05


@pytest.mark.parametrize("pos", [(-12.0, 3.5), (-25.0, 3.5), (-1.0, -3.5), (-30.0, 0.0)])
def test_rear_cameras_locate_cars_behind(cfg, pos):
    """Cars behind / beside us are only visible to the rear-facing side cameras."""
    sim = OfflineSimulator(cfg, "obstacle")
    sim.reset()
    s_ego = 60.0
    sim.actors = [car(s_ego + pos[0], pos[1])]
    frame = place(sim, s_ego)
    dets = GroundTruthDetector(pixel_noise=0.0).detect_all(frame, sim.cams)
    assert not dets["front_left"] and not dets["front_right"]
    assert dets["left_repeater"] or dets["right_repeater"]
    objs = CameraFusion(cfg, sim.cams).locate_objects(frame, dets)
    assert len(objs) == 1                                      # one object even if both side cameras see it
    gt = frame.ground_truth[0]
    assert abs(objs[0].y - gt.y) < 1.5
    assert abs(objs[0].x - gt.x) < 0.25 * abs(gt.x) + 2.5
