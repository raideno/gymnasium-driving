import collections
import dataclasses
import typing

import numpy as np


@dataclasses.dataclass
class StepData:
    """Single timestep data - flat structure for easy conversion to arrays."""

    timestamp: float
    position_x: float
    position_y: float
    velocity: float
    heading: float
    steering: float
    throttle: float
    brake: float
    reverse: bool


class EpisodeRecorder:
    """Records episode data with optional fixed buffer size."""

    def __init__(self):
        self.steps: collections.deque[StepData] = collections.deque()
        self.terminated = False
        self.truncated = False

    def record(self, step: StepData) -> None:
        self.steps.append(step)

    def reset(self) -> None:
        self.steps.clear()
        self.terminated = False
        self.truncated = False

    def to_arrays(self) -> dict[str, typing.Any]:
        """Convert to numpy arrays for batch processing."""
        if not self.steps:
            return {}

        return {
            "timestamps": np.array([s.timestamp for s in self.steps]),
            "positions": np.array([[s.position_x, s.position_y] for s in self.steps]),
            "velocities": np.array([s.velocity for s in self.steps]),
            "headings": np.array([s.heading for s in self.steps]),
            "steering_angles": np.array([s.steering for s in self.steps]),
            "actions": np.array(
                [
                    [s.steering, s.throttle, s.brake, float(s.reverse)]
                    for s in self.steps
                ]
            ),
            "terminated": self.terminated,
            "truncated": self.truncated,
        }

    def __len__(self) -> int:
        return len(self.steps)
