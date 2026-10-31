"""Mini Tesla - run the autonomous driving stack.

    python main.py                                   # offline simulator, 'full' scenario
    python main.py --scenario pedestrian --display   # watch it live
    python main.py --all-scenarios                   # run every scenario and print a report
    python main.py --backend carla --display         # CARLA server must be running
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2

from autopilot import Autopilot, draw_overlay
from evaluation.metrics import MetricsRecorder
from simulation.offline_sim import SCENARIO_NAMES
from utils.config import load_config

log = logging.getLogger("mini_tesla")


def make_simulator(cfg: dict, backend: str, scenario: str | None):
    if backend == "sim":
        from simulation.offline_sim import OfflineSimulator
        return OfflineSimulator(cfg, scenario)
    if backend == "carla":
        from simulation.carla_sim import CarlaSimulator
        return CarlaSimulator(cfg)
    raise ValueError(f"unknown backend {backend!r}")


def run(cfg: dict, backend: str = "sim", scenario: str | None = None, max_steps: int | None = None,
        display: bool = False, video: str | None = None, verbose: bool = False) -> dict:
    sim = make_simulator(cfg, backend, scenario)
    pilot = Autopilot(cfg, backend)
    metrics = MetricsRecorder(cfg["dt"])
    writer = None
    max_steps = max_steps or cfg["max_steps"]
    log.info("backend=%s scenario=%s lane=%s detector=%s controller=%s cameras=%s", backend,
             scenario or cfg.get("scenario"), pilot.lanes.method, pilot.detector_kind, pilot.lateral,
             ",".join(pilot.cams))
    try:
        frame = sim.reset()
        if getattr(sim, "max_steer_deg", None):
            pilot.set_max_steer(sim.max_steer_deg)
        last_action = None
        for step in range(max_steps):
            res = pilot.tick(frame)
            status = sim.status()
            metrics.record(res.world, res.decision, status, res.perception_ms)
            if res.decision.action != last_action:
                log.info("t=%6.2fs  %-15s %s", frame.timestamp, res.decision.action.value, res.decision.reason)
                last_action = res.decision.action
            elif verbose and step % 20 == 0:
                log.info("t=%6.2fs  v=%.1f lane=%s", frame.timestamp, frame.speed,
                         None if res.world.lane is None else round(res.world.lane.offset, 2))
            if display or video:
                vis = draw_overlay(frame, res, pilot.cams, cfg["vehicle"])
                if video:
                    if writer is None:
                        Path(video).parent.mkdir(parents=True, exist_ok=True)
                        writer = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), 1.0 / cfg["dt"],
                                                 (vis.shape[1], vis.shape[0]))
                    writer.write(vis)
                if display:
                    cv2.imshow("Mini Tesla", vis)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
            if frame.route_finished and frame.speed < 0.2:
                break
            frame = sim.step(res.control)
    finally:
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()
        sim.close()
    return metrics.summary()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--backend", choices=["sim", "carla"])
    ap.add_argument("--scenario", choices=SCENARIO_NAMES)
    ap.add_argument("--all-scenarios", action="store_true")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--display", action="store_true", help="show the live camera view with overlays")
    ap.add_argument("--video", help="save an annotated .mp4")
    ap.add_argument("--lane-method", choices=["auto", "cnn", "classical"])
    ap.add_argument("--detector", choices=["auto", "yolo", "ground_truth"])
    ap.add_argument("--controller", choices=["mpc", "pid"], help="lateral controller")
    ap.add_argument("--metrics", help="write the metrics summary JSON here")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    overrides: dict = {"perception": {}}
    if args.lane_method:
        overrides["perception"]["lane_method"] = args.lane_method
    if args.detector:
        overrides["perception"]["detector"] = args.detector
    if args.controller:
        overrides["control"] = {"lateral": args.controller}
    cfg = load_config(args.config, overrides)
    backend = args.backend or cfg["backend"]

    scenarios = list(SCENARIO_NAMES) if args.all_scenarios else [args.scenario or cfg["scenario"]]
    if backend == "carla":
        scenarios = ["carla"]
    report = {}
    for sc in scenarios:
        t0 = time.time()
        video = args.video if len(scenarios) == 1 else (
            str(Path(args.video).with_name(f"{Path(args.video).stem}_{sc}.mp4")) if args.video else None)
        report[sc] = run(cfg, backend, None if sc == "carla" else sc, args.steps, args.display, video, args.verbose)
        report[sc]["wall_time_s"] = round(time.time() - t0, 1)
    print(json.dumps(report, indent=2))
    if args.metrics:
        Path(args.metrics).write_text(json.dumps(report, indent=2))
    failed = [k for k, v in report.items() if v.get("collisions", 0) > 0]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
