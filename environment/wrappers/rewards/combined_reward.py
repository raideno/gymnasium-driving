"""
Combined reward wrapper that aggregates multiple reward components.

Allows flexible combination of different reward signals with configurable
weights for multi-objective optimization.
"""

import typing
import numpy as np
import gymnasium

# TODO: make it into a RewardWrapper, why did we use a general Wrapper ?
class CombinedReward(gymnasium.Wrapper):
    """
    Combined reward wrapper that aggregates multiple reward components.
    
    This wrapper combines:
    - Path following (CTE, heading error, progress)
    - Obstacle risk (Gaussian risk field)
    - Boundary keeping
    - Collision avoidance
    - Smoothness
    - Survival bonus
    
    All components can be individually weighted or disabled.
    
    Args:
        env: The environment to wrap
        weights: Dictionary of component weights
        normalize_rewards: Whether to normalize each component
        clip_total: Maximum absolute value for total reward
    """
    
    DEFAULT_WEIGHTS = {
        # Path following
        "cte": 1.0,
        "heading": 0.5,
        "progress": 1.0,
        "velocity": 0.2,
        "goal": 100.0,
        
        # Obstacle avoidance
        "obstacle_risk": 1.0,
        "collision": 100.0,
        "near_miss": 10.0,
        "ttc": 1.0,
        
        # Boundary
        "boundary": 1.0,
        "off_road": 50.0,
        "lane_center": 0.3,
        
        # Smoothness
        "steering_rate": 0.5,
        "jerk": 0.3,
        "lateral_accel": 0.3,
        
        # Survival
        "alive": 0.1,
        "termination": 50.0,
    }
    
    def __init__(
        self,
        env: gymnasium.Env,
        weights: typing.Dict[str, float] | None = None,
        normalize_rewards: bool = False,
        clip_total: float | None = None,
        
        # Path following parameters
        target_velocity: float = 5.0,
        max_cte: float = 4.0,
        
        # Risk field parameters
        sigma_x: float = 5.0,
        sigma_y: float = 3.0,
        safe_distance_threshold: float = 5.0,
        
        # Smoothness parameters
        max_comfortable_lateral_accel: float = 3.0,
        max_comfortable_jerk: float = 2.0,
        max_comfortable_steering_rate: float = 0.5,
        
        # Collision parameters
        critical_distance: float = 2.0,
        min_ttc: float = 2.0,
    ):
        super().__init__(env)
        
        self.weights = {**self.DEFAULT_WEIGHTS, **(weights or {})}
        self.normalize_rewards = normalize_rewards
        self.clip_total = clip_total
        
        # Parameters
        self.target_velocity = target_velocity
        self.max_cte = max_cte
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.safe_distance_threshold = safe_distance_threshold
        self.max_comfortable_lateral_accel = max_comfortable_lateral_accel
        self.max_comfortable_jerk = max_comfortable_jerk
        self.max_comfortable_steering_rate = max_comfortable_steering_rate
        self.critical_distance = critical_distance
        self.min_ttc = min_ttc
        
        # History for derivatives
        self._prev_closest_idx = 0
        self._prev_steering = 0.0
        self._prev_velocity = 0.0
        self._prev_acceleration = 0.0
        self._prev_obstacle_positions: typing.Dict[int, np.ndarray] = {}
        self._steps = 0
    
    # TODO: continuously send a reward of -1
    # TODO: remove the survival reward
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        self._steps += 1
        
        # Compute all reward components
        reward_components = {}
        
        # path following
        path_following_rewards = self._compute_path_following_rewards(observation)
        reward_components.update(path_following_rewards)
        
        # obstacle risk
        obstacle_risk_rewards = self._compute_obstacle_risk_rewards(observation)
        reward_components.update(obstacle_risk_rewards)
        
        # boundary
        boundary_rewards = self._compute_boundary_rewards(observation)
        reward_components.update(boundary_rewards)
        
        # smoothness
        smoothness_rewards = self._compute_smoothness_rewards(action, observation)
        reward_components.update(smoothness_rewards)
        
        # survival
        survival_rewards = self._compute_survival_rewards(observation, terminated, truncated)
        reward_components.update(survival_rewards)
        
        total_reward = 0.0
        weighted_components = {}
        
        for key, value in reward_components.items():
            weight = self.weights.get(key, 0.0)
            weighted_value = weight * value
            weighted_components[f"weighted_{key}"] = weighted_value
            total_reward += weighted_value
        
        if self.clip_total is not None:
            total_reward = np.clip(total_reward, -self.clip_total, self.clip_total)
        
        if info is None:
            info = {}
        info["reward_components"] = reward_components
        info["weighted_components"] = weighted_components
        info["total_reward"] = total_reward
        
        return observation, total_reward, terminated, truncated, info
    
    def _compute_path_following_rewards(self, observation: dict) -> dict:
        """Compute path following reward components."""
        ego_pos = np.array([self.env.unwrapped.state["x"], self.env.unwrapped.state["y"]], dtype=np.float32)
        ego_heading = self.env.unwrapped.state["yaw"]
        ego_velocity = self.env.unwrapped.state["velocity"]
        
        path = self.env.path
        rewards = {
            "cte": 0.0,
            "heading": 0.0,
            "progress": 0.0,
            "velocity": 0.0,
            "goal": 0.0,
        }
        
        if path is None or len(path) < 2:
            return rewards
        
        # NOTE: closest point
        distances = np.linalg.norm(path - ego_pos, axis=1)
        closest_idx = np.argmin(distances)
        closest_point = path[closest_idx]
        
        # NOTE: path direction
        if closest_idx < len(path) - 1:
            path_direction = path[closest_idx + 1] - path[closest_idx]
        else:
            path_direction = path[closest_idx] - path[closest_idx - 1]
        
        path_heading = np.arctan2(path_direction[1], path_direction[0])
        
        # CTE (negative penalty)
        to_ego = ego_pos - closest_point
        cross = np.cross(path_direction, to_ego)
        cte = abs(cross / (np.linalg.norm(path_direction) + 1e-6))
        rewards["cte"] = -(1 - np.exp(-cte / 2.0))
        
        # Heading error (negative penalty)
        heading_error = abs(ego_heading - path_heading)
        heading_error = min(heading_error, 2 * np.pi - heading_error)
        rewards["heading"] = -heading_error / np.pi
        
        # Progress (positive)
        if closest_idx > self._prev_closest_idx:
            rewards["progress"] = (closest_idx - self._prev_closest_idx) * 0.1
        self._prev_closest_idx = closest_idx
        
        # Velocity tracking
        velocity_error = abs(ego_velocity - self.target_velocity)
        rewards["velocity"] = 1 - velocity_error / self.target_velocity
        
        # Goal bonus
        goal_dist = np.linalg.norm(ego_pos - self.env.goal_pos)
        if goal_dist <= self.env.goal_radius:
            rewards["goal"] = 1.0
        
        return rewards
    
    def _compute_obstacle_risk_rewards(self, observation: dict) -> dict:
        """Compute obstacle risk reward components."""
        from ...components.obstacles import Circle, Rectangle
        
        ego_pos = np.array([self.env.unwrapped.state["x"], self.env.unwrapped.state["y"]], dtype=np.float32)
        ego_heading = self.env.unwrapped.state["yaw"]
        ego_velocity = self.env.unwrapped.state["velocity"]
        
        ego_vel_vec = np.array([
            ego_velocity * np.cos(ego_heading),
            ego_velocity * np.sin(ego_heading)
        ])
        
        rewards = {
            "obstacle_risk": 0.0,
            "collision": 0.0,
            "near_miss": 0.0,
            "ttc": 0.0,
        }
        
        min_distance = float('inf')
        min_ttc = float('inf')
        total_risk = 0.0
        
        for i, obstacle in enumerate(self.env.obstacles):
            obs_center = np.array(obstacle.center, dtype=np.float32)
            
            # Obstacle velocity
            if i in self._prev_obstacle_positions:
                obs_vel = (obs_center - self._prev_obstacle_positions[i]) / self.env.unwrapped.DELTA_TIME
            else:
                obs_vel = np.zeros(2)
            self._prev_obstacle_positions[i] = obs_center.copy()
            
            # Distance
            if isinstance(obstacle, Circle):
                radius = obstacle.radius
            elif isinstance(obstacle, Rectangle):
                radius = max(obstacle.width, obstacle.height) / 2
            else:
                radius = 1.0
            
            distance = np.linalg.norm(ego_pos - obs_center) - radius
            min_distance = min(min_distance, max(distance, 0))
            
            # Risk field
            delta = ego_pos - obs_center
            d = np.linalg.norm(delta)
            pseudo_d = max(d - self.safe_distance_threshold, 0.1)
            gaussian = np.exp(-0.5 * ((delta[0]/self.sigma_x)**2 + (delta[1]/self.sigma_y)**2))
            total_risk += gaussian / pseudo_d
            
            # TTC
            rel_pos = obs_center - ego_pos
            rel_vel = obs_vel - ego_vel_vec
            closing = -np.dot(rel_pos, rel_vel) / (np.linalg.norm(rel_pos) + 1e-6)
            if closing > 0:
                ttc = max(distance, 0) / closing
                min_ttc = min(min_ttc, ttc)
        
        # Risk penalty (negative)
        if len(self.env.obstacles) > 0:
            rewards["obstacle_risk"] = -total_risk / len(self.env.obstacles)
        
        # Collision check
        car_corners = self.env._get_car_corners()
        collision = False
        for obstacle in self.env.obstacles:
            for corner in car_corners:
                if obstacle.check_collision(corner):
                    collision = True
                    break
        
        rewards["collision"] = -1.0 if collision else 0.0
        
        # Near miss
        if min_distance < self.critical_distance:
            rewards["near_miss"] = -1.0
        
        # TTC penalty
        if min_ttc < self.min_ttc:
            rewards["ttc"] = -(1 - min_ttc / self.min_ttc)
        
        return rewards
    
    def _compute_boundary_rewards(self, observation: dict) -> dict:
        """Compute boundary reward components."""
        ego_pos = np.array([self.env.unwrapped.state["x"], self.env.unwrapped.state["y"]], dtype=np.float32)
        
        rewards = {
            "boundary": 0.0,
            "off_road": 0.0,
            "lane_center": 0.0,
        }
        
        if self.env.road_network is None:
            return rewards
        
        road = self.env.road_network.roads[0]
        half_width = road.half_width
        
        # Get centerline
        all_points = []
        for segment in road.segments:
            points = segment.get_centerline_points(25)
            all_points.extend(points)
        centerline = np.array(all_points, dtype=np.float32)
        
        if len(centerline) == 0:
            return rewards
        
        distances = np.linalg.norm(centerline - ego_pos, axis=1)
        closest_idx = np.argmin(distances)
        closest_point = centerline[closest_idx]
        
        if closest_idx < len(centerline) - 1:
            road_dir = centerline[closest_idx + 1] - centerline[closest_idx]
        else:
            road_dir = centerline[closest_idx] - centerline[closest_idx - 1]
        
        perp = np.array([-road_dir[1], road_dir[0]])
        perp = perp / (np.linalg.norm(perp) + 1e-6)
        
        lateral = np.dot(ego_pos - closest_point, perp)
        dist_to_edge = half_width - abs(lateral)
        
        # Boundary penalty
        if dist_to_edge < 1.0:
            rewards["boundary"] = -np.exp(-(dist_to_edge ** 2) / 0.5)
        
        # Off-road
        if self.env.road_network.is_off_road(ego_pos):
            rewards["off_road"] = -1.0
        
        # Lane centering (positive for being centered)
        rewards["lane_center"] = 1 - abs(lateral) / half_width
        
        return rewards
    
    def _compute_smoothness_rewards(self, action: np.ndarray, observation: dict) -> dict:
        """Compute smoothness reward components."""
        steering = action[0]

        velocity = self.env.unwrapped.state["velocity"]
        
        dt = self.env.unwrapped.DELTA_TIME
        
        rewards = {
            "steering_rate": 0.0,
            "jerk": 0.0,
            "lateral_accel": 0.0,
        }
        
        # Steering rate
        steering_rate = abs(steering - self._prev_steering) / dt
        if steering_rate > self.max_comfortable_steering_rate:
            excess = steering_rate - self.max_comfortable_steering_rate
            rewards["steering_rate"] = -excess / self.max_comfortable_steering_rate
        
        # Jerk
        accel = (velocity - self._prev_velocity) / dt
        jerk = abs(accel - self._prev_acceleration) / dt
        if jerk > self.max_comfortable_jerk:
            excess = jerk - self.max_comfortable_jerk
            rewards["jerk"] = -excess / self.max_comfortable_jerk
        
        # Lateral acceleration
        wheelbase = getattr(self.env, 'WHEELBASE', 2.5)
        turn_radius = wheelbase / (np.tan(abs(steering)) + 1e-6)
        lateral_accel = velocity ** 2 / turn_radius if turn_radius > 0.1 else 0
        if lateral_accel > self.max_comfortable_lateral_accel:
            excess = lateral_accel - self.max_comfortable_lateral_accel
            rewards["lateral_accel"] = -excess / self.max_comfortable_lateral_accel
        
        self._prev_steering = steering
        self._prev_velocity = velocity
        self._prev_acceleration = accel
        
        return rewards
    
    def _compute_survival_rewards(
        self, observation: dict, terminated: bool, truncated: bool
    ) -> dict:
        """Compute survival reward components."""
        rewards = {
            "alive": 0.0,
            "termination": 0.0,
        }
        
        goal_reached = False
        if terminated:
            ego_pos = np.array([self.env.unwrapped.state["x"], self.env.unwrapped.state["y"]], dtype=np.float32)
            goal_dist = np.linalg.norm(ego_pos - self.env.goal_pos)
            goal_reached = goal_dist <= self.env.goal_radius
        
        if terminated and not goal_reached:
            rewards["termination"] = -1.0
        elif truncated:
            rewards["termination"] = -0.5
        else:
            rewards["alive"] = 1.0
        
        return rewards
    
    def reset(self, **kwargs):
        self._prev_closest_idx = 0
        self._prev_steering = 0.0
        self._prev_velocity = 0.0
        self._prev_acceleration = 0.0
        self._prev_obstacle_positions.clear()
        self._steps = 0
        return self.env.reset(**kwargs)
