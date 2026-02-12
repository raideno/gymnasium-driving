import gymnasium
import numpy as np

from gymnasium_driving.helpers import (
    signed_cte_to_polyline,
    heading_error_to_polyline,
    closest_polyline_index,
)

class DrivingReward(gymnasium.Wrapper):
    """
    Simple reward function for path following with obstacle avoidance.
    
    Reward = w1·progress + w2·(-|cte|) + w3·cos(heading_error) + w4·(-obstacle_penalty) + w5·(-speed_error) + terminal_rewards
    
    Components:
    1. Progress: Reward for advancing along the path (encourages forward motion)
    2. Cross-track error (CTE): Penalty for deviating from centerline
    3. Heading alignment: Reward for facing the path direction
    4. Obstacle clearance: Penalty for proximity to obstacles
    5. Speed maintenance: Penalty for deviating from target speed
    6. Terminal rewards: Large penalty for collision, reward for goal
    """
    
    def __init__(
        self,
        environment,

        progress_weight: float = 10.0,
        cte_weight: float = 1.0,
        heading_weight: float = 0.5,
        obstacle_weight: float = 2.0,
        speed_weight: float = 0.1,

        target_speed: float = 5.0,
        safe_distance: float = 4.0,

        collision_penalty: float = -100.0,
        goal_reward: float = 100.0,
        off_road_penalty: float = -50.0,
    ):
        super().__init__(environment)
        self.env = environment 
        self.progress_weight = progress_weight
        self.cte_weight = cte_weight
        self.heading_weight = heading_weight
        self.obstacle_weight = obstacle_weight
        self.speed_weight = speed_weight
        
        self.target_speed = target_speed
        self.safe_distance = safe_distance
        
        self.collision_penalty = collision_penalty
        self.goal_reward = goal_reward
        self.off_road_penalty = off_road_penalty
        
        self.previous_progress = 0.0
        self.previous_closest_idx = 0
    
    def reset(self, **kwargs):
        self.previous_progress = 0.0
        self.previous_closest_idx = 0
        return super().reset(**kwargs)
    
    def step(self, action):
        obs, _, terminated, truncated, info = super().step(action)
        
        state = self.env.unwrapped.state
        ego_pos = np.array([state["x"], state["y"]], dtype=np.float32)
        ego_heading = state["yaw"]
        ego_velocity = state["velocity"]
        
        path = self.env.unwrapped.path
        obstacles = self.env.unwrapped.obstacles
        goal_pos = np.array(self.env.unwrapped.goal_pos, dtype=np.float32)
        goal_radius = self.env.unwrapped.goal_radius
        
        reward = 0.0
        
        # ===== 1. PROGRESS REWARD =====
        # Reward for making progress along the path
        if path is not None and len(path) >= 2:
            closest_idx = closest_polyline_index(path, ego_pos)
            
            # Normalized progress along path [0, 1]
            current_progress = closest_idx / max(len(path) - 1, 1)
            
            if self.previous_progress > 0:
                progress_delta = current_progress - self.previous_progress
                # Handle wrap-around (shouldn't happen but be safe)
                if progress_delta < -0.5:
                    progress_delta += 1.0
                reward += self.progress_weight * max(0, progress_delta)
            
            self.previous_progress = current_progress
            self.previous_closest_idx = closest_idx
        
        # ===== 2. CROSS-TRACK ERROR PENALTY =====
        # Penalize deviation from path centerline
        if path is not None and len(path) >= 2:
            cte, _ = signed_cte_to_polyline(path, ego_pos, idx=self.previous_closest_idx)
            reward -= self.cte_weight * abs(cte)
        
        # ===== 3. HEADING ALIGNMENT REWARD =====
        # Reward for facing the correct direction
        if path is not None and len(path) >= 2:
            heading_error = heading_error_to_polyline(
                path, ego_heading, self.previous_closest_idx
            )
            # cos(0) = 1 (perfect alignment), cos(π) = -1 (opposite direction)
            alignment = np.cos(heading_error)
            # reward += self.heading_weight * alignment
            reward += self.heading_weight * (alignment - 1.0)
        
        # ===== 4. OBSTACLE AVOIDANCE PENALTY =====
        # Penalize proximity to obstacles
        for obstacle in obstacles:
            obstacle_pos = np.array(obstacle.center, dtype=np.float32)
            distance = np.linalg.norm(ego_pos - obstacle_pos)
            
            # Get obstacle radius/size
            if hasattr(obstacle, 'radius'):
                radius = obstacle.radius
            elif hasattr(obstacle, 'width') and hasattr(obstacle, 'height'):
                radius = max(obstacle.width, obstacle.height) / 2
            else:
                radius = 1.0
            
            # Effective clearance considering obstacle size
            clearance = distance - radius
            
            if 0 < clearance < self.safe_distance:
                # Smooth penalty that increases as we get closer
                proximity_ratio = 1.0 - (clearance / self.safe_distance)
                penalty = proximity_ratio ** 2  # Quadratic penalty
                reward -= self.obstacle_weight * penalty
        
        # ===== 5. SPEED MAINTENANCE =====
        # Encourage maintaining target speed
        speed_error = abs(ego_velocity - self.target_speed)
        normalized_error = speed_error / max(self.target_speed, 1.0)
        reward -= self.speed_weight * normalized_error
        
        # ===== 6. TERMINAL REWARDS =====
        if terminated:
            # Check if we reached the goal
            goal_distance = np.linalg.norm(ego_pos - goal_pos)
            
            if goal_distance <= goal_radius:
                reward += self.goal_reward
            else:
                # Check if off-road (if road network exists)
                road_network = self.env.unwrapped.road_network
                if road_network is not None and road_network.is_off_road(ego_pos):
                    reward += self.off_road_penalty
                else:
                    # Collision with obstacle
                    reward += self.collision_penalty
                    
        # === step penalty === reward -= 0.01
        reward += -0.01
        
        # Store reward breakdown in info for debugging
        info['reward_breakdown'] = {
            'total': reward,
        }
        
        return obs, reward, terminated, truncated, info
