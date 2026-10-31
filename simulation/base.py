"""Interface every simulator backend (offline or CARLA) implements."""
from __future__ import annotations

from abc import ABC, abstractmethod

from utils.types import Control, Frame


class Simulator(ABC):
    @abstractmethod
    def reset(self) -> Frame:
        """Spawn the ego vehicle and return the first frame."""

    @abstractmethod
    def step(self, control: Control) -> Frame:
        """Apply a control command, advance one tick and return the new frame."""

    def status(self) -> dict:
        """Backend-side measurements for evaluation (collisions, lane error, progress...)."""
        return {}

    def close(self) -> None:
        pass
