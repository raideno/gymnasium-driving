from dataclasses import dataclass
from typing import Optional

import gymnasium
import numpy as np

from gymnasium_driving.components.obstacles import Circle, Rectangle


@dataclass(frozen=True)
class SmoothPathProgressRewardConfig:
    truncation_penalty: float = -5.0
    collision_penalty: float = -20.0
    goal_reward: float = 50.0

    heading_weight: float = 0.2
    cte_weight: float = 0.3
    progress_weight: float = 2.0
    alive_bonus: float = 0.05
    no_progress_penalty: float = -0.02

    obstacle_weight: float = 0.8
    obstacle_danger_radius: float = 3.5

    road_half_width: float = 4.0  # meters (half of the 8m road width)

    # --- Smoothness penalties (new) ---
    # Penalizes the change in heading error between steps.
    # This directly discourages heading oscillation.
    heading_rate_weight: float = 0.5

    # Penalizes the change in steering action between steps.
    # This discourages jerky / rapidly reversing steering inputs.
    steering_jerk_weight: float = 0.3


class SmoothPathProgressReward(gymnasium.Wrapper):
    """
    Extends PathProgressReward with two additional smoothness penalties:

    - Heading rate penalty: penalizes large changes in heading error between
      consecutive steps. This directly suppresses the heading oscillation that
      arises when only the absolute heading error is penalized.

    - Steering jerk penalty: penalizes large changes in the steering component
      of the action between consecutive steps. This encourages the agent to
      issue smooth, consistent steering commands rather than rapidly alternating
      left/right inputs.

    Everything else (progress, CTE, obstacle proximity, alive bonus, goal /
    collision / truncation terminals) is identical to PathProgressReward.
    """

    def __init__(
        self,
        environment: gymnasium.Env,
        configuration: Optional[SmoothPathProgressRewardConfig] = None,
    ):
        super().__init__(environment)

        self.env = environment
        self.configuration = configuration or SmoothPathProgressRewardConfig()

        self._prev_goal_distance: Optional[float] = None
        self._prev_path_idx: Optional[int] = None
        self._prev_heading_error: Optional[float] = None
        self._prev_steering: Optional[float] = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        ego_position = np.array(
            [self.env.unwrapped.state["x"], self.env.unwrapped.state["y"]],
            dtype=np.float32,
        )
        goal_pos = np.array(self.env.unwrapped.goal_pos, dtype=np.float32)
        self._prev_goal_distance = float(np.linalg.norm(ego_position - goal_pos))
        self._prev_path_idx = int(self.env.unwrapped.state.get("closest_path_idx", 0))

        # Reset smoothness trackers so the first step incurs no penalty.
        self._prev_heading_error = None
        self._prev_steering = None

        return obs, info

    def step(self, action):
        observation, _reward, terminated, truncated, info = self.env.step(action)

        ego_position = np.array(
            [self.env.unwrapped.state["x"], self.env.unwrapped.state["y"]],
            dtype=np.float32,
        )

        path = self.env.unwrapped.path
        reward = 0.0

        # NOTE: per-step alive bonus
        reward += self.configuration.alive_bonus

        # NOTE: progress reward based on monotonic index along sampled path
        current_path_idx = int(self.env.unwrapped.state.get("closest_path_idx", 0))
        if self._prev_path_idx is None:
            self._prev_path_idx = current_path_idx

        progress_idx = max(0, current_path_idx - self._prev_path_idx)
        reward += self.configuration.progress_weight * float(progress_idx)
        if progress_idx == 0:
            reward += self.configuration.no_progress_penalty

        self._prev_path_idx = current_path_idx

        goal_pos = np.array(self.env.unwrapped.goal_pos, dtype=np.float32)
        goal_distance = float(np.linalg.norm(ego_position - goal_pos))
        self._prev_goal_distance = goal_distance

        if path is not None and len(path) >= 2:
            cte = float(self.env.unwrapped.state["cte"])
            heading_error = float(self.env.unwrapped.state["heading_error"])

            # NOTE: CTE penalty (normalized by road half width)
            reward -= (
                self.configuration.cte_weight
                * (cte / self.configuration.road_half_width) ** 2
            )

            # NOTE: heading error penalty (normalized by pi)
            reward -= self.configuration.heading_weight * (heading_error / np.pi) ** 2

            # NOTE: heading rate penalty — penalizes how much the heading error
            # changed since the last step.  A large delta means the agent is
            # oscillating (overcorrecting left then right repeatedly), so we
            # charge a quadratic cost on the delta normalized by pi.
            if self._prev_heading_error is not None:
                heading_delta = heading_error - self._prev_heading_error
                reward -= (
                    self.configuration.heading_rate_weight
                    * (heading_delta / np.pi) ** 2
                )

            self._prev_heading_error = heading_error

        # NOTE: steering jerk penalty — penalizes large changes in the steering
        # action between consecutive steps.
        # Assumes action is array-like with steering at index 0.
        # Adjust the index (or use action directly if scalar) to match your
        # action space convention.
        try:
            steering = float(action[0])
        except (TypeError, IndexError):
            steering = float(action)

        if self._prev_steering is not None:
            steering_delta = steering - self._prev_steering
            # The steering range is typically [-1, 1], so delta is in [-2, 2].
            # Normalizing by 2 keeps the penalty in [0, 1].
            reward -= (
                self.configuration.steering_jerk_weight * (steering_delta / 2.0) ** 2
            )

        self._prev_steering = steering

        if terminated:
            if goal_distance <= self.env.unwrapped.goal_radius:
                reward += self.configuration.goal_reward
            else:
                reward += self.configuration.collision_penalty

        if truncated:
            reward += self.configuration.truncation_penalty

        # NOTE: obstacle proximity penalty
        obstacle_penalty = 0.0

        for obstacle in self.env.unwrapped.obstacles:
            obstacle_center_distance = float(
                np.linalg.norm(
                    ego_position - np.array(obstacle.center, dtype=np.float32)
                )
            )

            if isinstance(obstacle, Circle):
                obstacle_surface_distance = obstacle_center_distance - obstacle.radius
            elif isinstance(obstacle, Rectangle):
                obstacle_surface_distance = (
                    obstacle_center_distance - max(obstacle.width, obstacle.height) / 2
                )
            else:
                obstacle_surface_distance = obstacle_center_distance - 1.0

            obstacle_surface_distance = max(obstacle_surface_distance, 0.0)

            if obstacle_surface_distance < self.configuration.obstacle_danger_radius:
                normalised = (
                    1.0
                    - obstacle_surface_distance
                    / self.configuration.obstacle_danger_radius
                )
                obstacle_penalty += normalised**2

        reward -= self.configuration.obstacle_weight * obstacle_penalty

        if info is None:
            info = {}

        return observation, reward, terminated, truncated, info
