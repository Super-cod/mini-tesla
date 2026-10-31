"""Train the lane-marking CNN.

Data sources:
  * synthetic (default): images + pixel-perfect masks rendered by the offline simulator with
    random poses, lighting, shadows and traffic.
  * a folder of real/CARLA images:  <dir>/images/*.png and <dir>/masks/*.png
    (scripts/collect_carla_data.py produces this from CARLA's semantic camera).

    python -m models.train_lane_cnn --samples 4000 --epochs 12
    python -m models.train_lane_cnn --data data/carla_lanes --epochs 20
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

from models.lane_cnn import LaneCNN
from simulation.offline_sim import SCENARIO_NAMES, Renderer, build_scenario, car, pedestrian
from utils.config import load_config, resolve_path
from utils.geometry import make_stereo_pair


def synthetic_dataset(n: int, cfg: dict, hw=(128, 256), seed: int = 0):
    rng = np.random.default_rng(seed)
    cams = make_stereo_pair(cfg["cameras"])
    roads = [build_scenario(name)[0] for name in SCENARIO_NAMES]
    imgs = np.empty((n, hw[0], hw[1], 3), np.uint8)
    masks = np.empty((n, hw[0], hw[1]), np.uint8)
    for i in range(n):
        road = roads[rng.integers(len(roads))]
        renderer = Renderer(road)
        s = rng.uniform(0, road.length - 20)
        d = rng.choice([rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0) + road.lane_width])
        x, y, yaw = road.pose(s)
        p = road.point(s, d)
        ego = np.array([p[0], p[1], yaw + rng.normal(0, 0.06)])
        actors = []
        for _ in range(rng.integers(0, 3)):
            sa = s + rng.uniform(6, 45)
            if rng.random() < 0.7:
                actors.append(car(sa, rng.choice([0.0, road.lane_width])))
            else:
                actors.append(pedestrian(sa, rng.uniform(-3, 7), 0, None))
        cam = cams[rng.integers(2)]
        img, mask = renderer.render(cam, ego, s, actors, rng, augment=True, with_mask=True)
        imgs[i] = cv2.resize(img, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)
        masks[i] = cv2.resize(mask, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)
    return imgs, masks


def folder_dataset(root: Path, hw=(128, 256)):
    imgs, masks = [], []
    for ip in sorted((root / "images").glob("*.png")):
        mp = root / "masks" / ip.name
        if not mp.exists():
            continue
        imgs.append(cv2.resize(cv2.imread(str(ip)), (hw[1], hw[0]), interpolation=cv2.INTER_AREA))
        masks.append(cv2.resize(cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE), (hw[1], hw[0]), interpolation=cv2.INTER_AREA))
    if not imgs:
        raise FileNotFoundError(f"no image/mask pairs under {root}")
    return np.stack(imgs), np.stack(masks)


def to_tensors(imgs: np.ndarray, masks: np.ndarray) -> TensorDataset:
    x = torch.from_numpy(imgs[..., ::-1].copy()).permute(0, 3, 1, 2).float().div(255.0)
    y = torch.from_numpy(masks).float().div(255.0)[:, None]
    return TensorDataset(x, y)


def loss_fn(logits, target, pos_weight: float = 8.0):
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=torch.tensor(pos_weight, device=logits.device))
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
    return bce + dice.mean()


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    tp = fp = fn = 0.0
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total += loss_fn(logits, y).item() * len(x)
        pred = (torch.sigmoid(logits) > 0.5).float()
        gt = (y > 0.5).float()
        tp += (pred * gt).sum().item()
        fp += (pred * (1 - gt)).sum().item()
        fn += ((1 - pred) * gt).sum().item()
    n = len(loader.dataset)
    return {"loss": total / n, "iou": tp / max(tp + fp + fn, 1), "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1)}


def train(dataset: Dataset, epochs: int, out: Path, batch: int = 32, lr: float = 2e-3, device=None, log=print):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    n_val = max(1, int(0.1 * len(dataset)))
    train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val], generator=torch.Generator().manual_seed(0))
    tl = DataLoader(train_ds, batch_size=batch, shuffle=True)
    vl = DataLoader(val_ds, batch_size=batch)
    model = LaneCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(tl))
    best = math.inf
    metrics = {}
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            if torch.rand(1).item() < 0.5:                       # horizontal flip augmentation
                x, y = x.flip(3), y.flip(3)
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        metrics = evaluate(model, vl, device)
        log(f"epoch {ep + 1}/{epochs}  val_loss={metrics['loss']:.4f}  IoU={metrics['iou']:.3f}  "
            f"P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  ({time.time() - t0:.1f}s)")
        if metrics["loss"] < best:
            best = metrics["loss"]
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "base": 16, "metrics": metrics}, out)
    return model, metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, help="folder with images/ and masks/ (default: synthetic)")
    ap.add_argument("--samples", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg = load_config()
    hw = tuple(cfg["perception"]["lane_cnn_input"])
    out = args.out or resolve_path(cfg["perception"]["lane_weights"])
    t0 = time.time()
    if args.data:
        imgs, masks = folder_dataset(args.data, hw)
    else:
        imgs, masks = synthetic_dataset(args.samples, cfg, hw)
    print(f"dataset: {len(imgs)} samples in {time.time() - t0:.1f}s")
    train(to_tensors(imgs, masks), args.epochs, out, batch=args.batch)
    print(f"saved best weights to {out}")


if __name__ == "__main__":
    main()
