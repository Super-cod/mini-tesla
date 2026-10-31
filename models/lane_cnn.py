"""Small U-Net that segments lane markings. ~0.3M parameters, runs in a few ms on a laptop GPU."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class LaneCNN(nn.Module):
    """Input: Bx3xHxW image in [0, 1] (RGB). Output: Bx1xHxW logits of 'lane marking'."""

    def __init__(self, base: int = 16):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 6
        self.enc1 = _block(3, c1)
        self.enc2 = _block(c1, c2)
        self.enc3 = _block(c2, c3)
        self.mid = _block(c3, c4)
        self.dec3 = _block(c4 + c3, c3)
        self.dec2 = _block(c3 + c2, c2)
        self.dec1 = _block(c2 + c1, c1)
        self.head = nn.Conv2d(c1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        m = self.mid(F.max_pool2d(e3, 2))
        d3 = self.dec3(torch.cat([F.interpolate(m, size=e3.shape[2:], mode="bilinear", align_corners=False), e3], 1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, size=e2.shape[2:], mode="bilinear", align_corners=False), e2], 1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, size=e1.shape[2:], mode="bilinear", align_corners=False), e1], 1))
        return self.head(d1)


def load_lane_cnn(path, device: str | torch.device = "cpu") -> LaneCNN:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = LaneCNN(base=ckpt.get("base", 16))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()
