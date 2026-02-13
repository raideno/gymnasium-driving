import gymnasium
import numpy as np

from gymnasium_driving.helpers import (
    signed_cte_to_polyline,
    heading_error_to_polyline,
    closest_polyline_index,
)

class DrivingReward(gymnasium.Wrapper):
    def __init__(
        self,
        environment,

        cte_weight: float = 1.0,
        heading_weight: float = 0.5,
        obstacle_weight: float = 2.0,
        speed_weight: float = 0.1,

        target_speed: float = 5.0,
        safe_distance: float = 4.0,

        collision_penalty: float = -100.0,
        goal_reward: float = 100.0,
        **kwargs
    ):
        super().__init__(environment)
        self.env = environment 
        self.cte_weight = cte_weight
        self.heading_weight = heading_weight
        self.obstacle_weight = obstacle_weight
        self.speed_weight = speed_weight
        
        self.target_speed = target_speed
        self.safe_distance = safe_distance
        
        self.collision_penalty = collision_penalty
        self.goal_reward = goal_reward
        
        self.previous_closest_idx = 0
    
    def reset(self, **kwargs):
        self.previous_closest_idx = 0
        return super().reset(**kwargs)
    
    def step(self, action):
        obs, _, terminated, truncated, info = super().step(action)
        
        ego_pos = np.array([self.env.unwrapped.state["x"], self.env.unwrapped.state["y"]], dtype=np.float32)
        ego_heading = self.env.unwrapped.state["yaw"]
        ego_velocity = self.env.unwrapped.state["velocity"]
        
        path = self.env.unwrapped.path
        obstacles = self.env.unwrapped.obstacles
        goal_pos = np.array(self.env.unwrapped.goal_pos, dtype=np.float32)
        goal_radius = self.env.unwrapped.goal_radius
        
        reward = 0.0
        
        self.closest_idx = closest_polyline_index(path, ego_pos)
        
        # NOTE: cte error 
        if path is not None and len(path) >= 2:
            reward -= self.cte_weight * abs(signed_cte_to_polyline(path, ego_pos, idx=self.previous_closest_idx)[0])
       
        # NOTE: heading error 
        if path is not None and len(path) >= 2:
            # cos(0) = 1 (perfect alignment), cos(π) = -1 (opposite direction)
            # heading error \in [-\pi, pi], cosine will smooth the reward
            alignment = np.cos(heading_error_to_polyline(path, ego_heading, self.previous_closest_idx))
            reward += self.heading_weight * alignment
       
        # NOTE: obstacle proximity 
        for obstacle in obstacles:
            distance = np.linalg.norm(ego_pos - np.array(obstacle.center, dtype=np.float32))
            
            if hasattr(obstacle, 'radius'):
                radius = obstacle.radius
            elif hasattr(obstacle, 'width') and hasattr(obstacle, 'height'):
                radius = max(obstacle.width, obstacle.height) / 2
            else:
                radius = 1.0
            
            clearance = distance - radius
            
            if 0 < clearance < self.safe_distance:
                # Smooth penalty that increases as we get closer
                proximity_ratio = 1.0 - (clearance / self.safe_distance)
                penalty = proximity_ratio ** 2  # Quadratic penalty
                reward -= self.obstacle_weight * penalty
       
        # NOTE: speed maintenance 
        normalized_speed_error = abs(ego_velocity - self.target_speed) / max(self.target_speed, 1.0)
        reward -= self.speed_weight * normalized_speed_error 
        
        # NOTE: terminal rewards
        if terminated:
            if np.linalg.norm(ego_pos - goal_pos) <= goal_radius:
                reward += self.goal_reward
            else:
                reward += self.collision_penalty
                    
        # NOTE: step penalty 
        reward += -0.01
        
        info['reward_breakdown'] = {
            'total': reward,
        }
        
        return obs, reward, terminated, truncated, info
