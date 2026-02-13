import gymnasium
import numpy as np

from gymnasium_driving.wrappers.rewards.path_progress import PathProgressReward
from gymnasium_driving.components.obstacles import Circle, Rectangle


class PathProgressObstaclesReward(PathProgressReward):
    """
    Extends PathProgressReward with an obstacle proximity penalty.

    The agent is penalised proportionally to how close it gets to obstacles,
    giving it a smooth gradient to learn avoidance *before* a collision
    actually terminates the episode.
    """

    OBSTACLE_WEIGHT = 0.8
    # Distance (meters) within which the penalty starts
    OBSTACLE_DANGER_RADIUS = 3.5

    def __init__(
        self,
        environment: gymnasium.Env,
    ):
        super().__init__(environment)

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)

        ego_position = np.array(
            [self.env.unwrapped.state["x"], self.env.unwrapped.state["y"]],
            dtype=np.float32,
        )

        # --- obstacle proximity penalty ---
        obstacle_penalty = 0.0

        for obstacle in self.env.unwrapped.obstacles:
            obs_center = np.array(obstacle.center, dtype=np.float32)
            dist = float(np.linalg.norm(ego_position - obs_center))

            if isinstance(obstacle, Circle):
                surface_dist = dist - obstacle.radius
            elif isinstance(obstacle, Rectangle):
                surface_dist = dist - max(obstacle.width, obstacle.height) / 2
            else:
                surface_dist = dist - 1.0

            surface_dist = max(surface_dist, 0.0)

            if surface_dist < self.OBSTACLE_DANGER_RADIUS:
                # Quadratic penalty: 1.0 at surface, 0.0 at danger radius
                normalised = 1.0 - surface_dist / self.OBSTACLE_DANGER_RADIUS
                obstacle_penalty += normalised ** 2

        reward -= self.OBSTACLE_WEIGHT * obstacle_penalty

        return observation, reward, terminated, truncated, info
