import gymnasium
import numpy as np

class PathProgressReward(gymnasium.Wrapper):
    TRUNCATION_PENALTY = -5.0
    COLLISION_PENALTY = -10.0
    GOAL_REWARD = 100.0
    
    HEADING_WEIGHT = 0.4
    CTE_WEIGHT = 0.6
    
    def __init__(
        self,
        environment: gymnasium.Env,
    ):
        super().__init__(environment)
        
        self.env = environment

    def step(self, action):
        observation, _reward, terminated, truncated, info = self.env.step(action)
        
        ego_position = np.array([self.env.unwrapped.state["x"], self.env.unwrapped.state["y"]], dtype=np.float32)

        path = self.env.unwrapped.path
        reward = 0.0

        if path is not None and len(path) >= 2:
            cte = self.env.unwrapped.state["cte"]
            heading_error = self.env.unwrapped.state["heading_error"]

            # NOTE: cte penalty
            road_half_width = 4.0  # meters (half of the 8m road width)
            reward -= PathProgressReward.CTE_WEIGHT * (cte / road_half_width) ** 2

            # NOTE: heading penalty
            reward -= PathProgressReward.HEADING_WEIGHT * (heading_error / np.pi) ** 2

        goal_dist = np.linalg.norm(ego_position - np.array(self.env.unwrapped.goal_pos, dtype=np.float32))

        if terminated:
            if goal_dist <= self.env.unwrapped.goal_radius:
                reward += PathProgressReward.GOAL_REWARD
            else:
                reward += PathProgressReward.COLLISION_PENALTY

        if truncated:
            reward += PathProgressReward.TRUNCATION_PENALTY

        if info is None:
            info = {}

        return observation, reward, terminated, truncated, info
