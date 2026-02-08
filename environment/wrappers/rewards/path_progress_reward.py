import gymnasium
import numpy as np

class PathProgressReward(gymnasium.Wrapper):
    """
    Replaces the base environment's reward (always 0.0) with a shaped reward
    that incentivizes:

    1. Forward velocity projected onto the path direction (primary signal)
    2. Staying close to the road center (CTE penalty)
    3. Heading alignment with the path (heading penalty)
    4. Terminal bonuses/penalties for goal/collision
    """

    def __init__(
        self,
        env: gymnasium.Env,
        
        target_velocity: float = 5.0,
        velocity_weight: float = 1.0,
        
        cte_weight: float = 0.3,
        heading_weight: float = 0.2,
        
        goal_reward: float = 100.0,
        
        collision_penalty: float = -10.0,
        truncation_penalty: float = -5.0,
    ):
        super().__init__(env)

        self.target_velocity = target_velocity
        self.velocity_weight = velocity_weight
        self.cte_weight = cte_weight
        self.heading_weight = heading_weight
        self.goal_reward = goal_reward
        self.collision_penalty = collision_penalty
        self.truncation_penalty = truncation_penalty

    def step(self, action):
        observation, _reward, terminated, truncated, info = self.env.step(action)
        
        ego_position = np.array([self.env.unwrapped.state["x"], self.env.unwrapped.state["y"]], dtype=np.float32)
        heading = self.env.unwrapped.state["yaw"]
        velocity = self.env.unwrapped.state["velocity"]

        path = self.env.unwrapped.path
        reward = 0.0

        if path is not None and len(path) >= 2:
            # Find closest point on the reference path
            distances = np.linalg.norm(path - ego_position, axis=1)
            closest_idx = np.argmin(distances)
            cte = distances[closest_idx]

            # Path tangent direction at closest point
            if closest_idx < len(path) - 1:
                path_dir = path[closest_idx + 1] - path[closest_idx]
            else:
                path_dir = path[-1] - path[-2]
            path_heading = np.arctan2(path_dir[1], path_dir[0])
            heading_error = heading - path_heading
            heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

            # ── 1. Forward velocity along path (PRIMARY reward signal) ──
            # Positive when moving forward along path, negative when going backward.
            # Normalized by target_velocity so reward ≈ 1.0 at desired speed.
            velocity_along_path = velocity * np.cos(heading_error)
            reward += self.velocity_weight * (velocity_along_path / self.target_velocity)

            # ── 2. Cross-track error penalty ──
            # Quadratic penalty normalized by approximate road half-width.
            road_half_width = 4.0  # meters (half of the 8m road width)
            reward -= self.cte_weight * (cte / road_half_width) ** 2

            # ── 3. Heading alignment penalty ──
            reward -= self.heading_weight * (heading_error / np.pi) ** 2

        # ── 4. Terminal rewards ──
        goal_pos = np.array(self.env.unwrapped.goal_pos, dtype=np.float32)
        goal_dist = np.linalg.norm(ego_position - goal_pos)

        if terminated:
            if goal_dist <= self.env.unwrapped.goal_radius:
                reward += self.goal_reward
            else:
                reward += self.collision_penalty

        if truncated:
            reward += self.truncation_penalty

        if info is None:
            info = {}

        return observation, reward, terminated, truncated, info
