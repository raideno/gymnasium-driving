import gymnasium
import numpy as np

from dataclasses import dataclass
from typing import Optional

from gymnasium_driving.components.obstacles import Circle, Rectangle

@dataclass(frozen=True)
class PathProgressRewardConfig:
    truncation_penalty: float = -5.0
    collision_penalty: float = -15.0
    goal_reward: float = 50.0

    heading_weight: float = 0.4
    cte_weight: float = 0.6
    progress_weight: float = 1.0
    alive_bonus: float = 0.1

    obstacle_weight: float = 0.8
    obstacle_danger_radius: float = 3.5

    road_half_width: float = 4.0  # meters (half of the 8m road width)

class PathProgressReward(gymnasium.Wrapper):
    """
    Accounts for:
    - Progress towards the goal (based on change in distance to goal).
    - Path following quality (CTE and heading error penalties).
    - Collision penalty (if episode ends without reaching goal).
    - Truncation penalty (if episode is truncated due to time limit).
    - Obstacle proximity penalty (quadratic penalty based on distance to nearby obstacles).
    - Alive bonus (small positive reward each step to encourage efficiency).
    """
    def __init__(
        self,
        environment: gymnasium.Env,
        configuration: Optional[PathProgressRewardConfig] = None,
    ):
        super().__init__(environment)

        self.env = environment
        self.configuration = configuration or PathProgressRewardConfig()
        self._prev_goal_distance: Optional[float] = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        ego_position = np.array(
            [self.env.unwrapped.state["x"], self.env.unwrapped.state["y"]],
            dtype=np.float32,
        )
        goal_pos = np.array(self.env.unwrapped.goal_pos, dtype=np.float32)
        self._prev_goal_distance = float(np.linalg.norm(ego_position - goal_pos))

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

        # NOTE: forward progress reward — positive when getting closer to the goal
        goal_pos = np.array(self.env.unwrapped.goal_pos, dtype=np.float32)
        goal_distance = float(np.linalg.norm(ego_position - goal_pos))

        if self._prev_goal_distance is not None:
            # TODO: compute progress as number of waypoints passed along the path
            # progress based on goal distance is incorrect
            progress = self._prev_goal_distance - goal_distance
            reward += self.configuration.progress_weight * progress

        self._prev_goal_distance = goal_distance

        if path is not None and len(path) >= 2:
            cte = self.env.unwrapped.state["cte"]
            heading_error = self.env.unwrapped.state["heading_error"]

            # NOTE: cte penalty (normalized by road half width)
            reward -= self.configuration.cte_weight * (cte / self.configuration.road_half_width) ** 2

            # NOTE: heading error penalty (normalized by pi)
            reward -= self.configuration.heading_weight * (heading_error / np.pi) ** 2

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
                obstacle_surface_distance = obstacle_center_distance - max(
                    obstacle.width, obstacle.height
                ) / 2
            else:
                obstacle_surface_distance = obstacle_center_distance - 1.0

            obstacle_surface_distance = max(obstacle_surface_distance, 0.0)

            if obstacle_surface_distance < self.configuration.obstacle_danger_radius:
                # 1.0 at surface, 0.0 at danger radius and further
                normalised = 1.0 - obstacle_surface_distance / self.configuration.obstacle_danger_radius
                obstacle_penalty += normalised**2

        reward -= self.configuration.obstacle_weight * obstacle_penalty

        if info is None:
            info = {}

        return observation, reward, terminated, truncated, info
