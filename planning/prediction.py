"""Use the tracker's motion forecasts to look ahead in time.

* corridor_intrusion  - will this object be inside my driving corridor within the next few
                        seconds? (a pedestrian walking towards the road, a car cutting in)
* lane_change_safe    - is the target lane free now *and* for the next few seconds, including
                        cars approaching from behind that only the rear-facing cameras can see?
"""
from __future__ import annotations

import numpy as np

from planning.path_planner import offset_path
from utils.types import TrackedObject


def path_y(path: np.ndarray, x: float | np.ndarray) -> float | np.ndarray:
    if len(path) < 2:
        return np.zeros_like(np.asarray(x, dtype=float))
    return np.interp(x, path[:, 0], path[:, 1])


def corridor_intrusion(obj: TrackedObject, path: np.ndarray, half_width: float,
                       horizon: float = 3.0, x_max: float = 60.0) -> tuple[float, float] | None:
    """Earliest predicted (time, x) at which the object is inside the corridor ahead of the car."""
    if not obj.velocity_valid or obj.predicted is None:
        return None
    for t, x, y in obj.predicted:
        if t > horizon:
            break
        if 0.0 < x < x_max and abs(y - path_y(path, x)) < half_width:
            return float(t), float(x)
    return None


def lane_change_safe(objects: list[TrackedObject], route: np.ndarray, offset: float, ego_speed: float,
                     ego_front: float, horizon: float = 4.0) -> tuple[bool, str]:
    """Check the lane `offset` metres to the left of the route.

    Unsafe if any object is in that lane close to us now, or is predicted to be there within
    `horizon` seconds while we are alongside it (positions relative to where we will be).
    """
    lane = offset_path(route, offset)
    for o in objects:
        if o.cls not in ("car", "cyclist", "pedestrian"):
            continue
        in_lane_now = abs(o.y - path_y(lane, max(o.x, 0.0))) < 2.0
        if in_lane_now and -12.0 < o.x < ego_front + 25.0:
            return False, f"{o.cls} in target lane at {o.x:+.0f} m"
        if not o.velocity_valid or o.predicted is None:
            # Speed not known yet: anything behind us in that lane could be closing fast.
            if in_lane_now and o.cls != "pedestrian" and -40.0 < o.x < 0.0:
                return False, f"{o.cls} behind in target lane, speed unknown"
            continue
        for t, x, y in o.predicted:
            if t > horizon:
                break
            rel = x - ego_speed * t                       # relative to where we will be
            if abs(y - path_y(lane, max(rel, 0.0))) < 2.0 and -6.0 < rel < ego_front + 12.0:
                return False, f"{o.cls} approaching in target lane ({o.vx:.0f} m/s)"
    return True, ""
