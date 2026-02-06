import typing
import gymnasium

import numpy as np

class SimpleRewardWrapper(gymnasium.RewardWrapper):
    """
    - Goal reaching: +100.0
    - Collision: -100.0
    - Out of bounds: -50.0
    - Off-road (if enforce_road enabled): -100.0
    - Off-road penalty (per step): off_road_penalty
    - Distance penalty (per step): -0.01 * goal_distance
    - Step penalty: -0.1
    """
    
    def __init__(
        self,
        env: gymnasium.Env,
        off_road_penalty: float = 0.1,
    ):
        super().__init__(env)
        self.off_road_penalty = off_road_penalty
    
    def reward(self, reward: float) -> float:
        ego_position = self.env.unwrapped.state[:2]
        
        goal_distance = np.linalg.norm(ego_position - self.env.goal_pos)
        if goal_distance <= self.env.goal_radius:
            return 100.0
        
        # collision
        if self.env._check_collision():
            return -100.0
        
        # out of bounds
        if not self.env._within_world_boundaries():
            return -50.0
        
        # off road
        if self.env.road_network is not None:
            if self.env.road_network.is_off_road(ego_position):
                if self.env.enforce_road:
                    return -100.0
                else:
                    return self.off_road_penalty - 0.01 * goal_distance
        
        # step penalty
        return -0.1 - 0.01 * goal_distance
