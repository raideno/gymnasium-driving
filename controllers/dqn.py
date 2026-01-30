import typing
import gymnasium
import numpy as np

from pathlib import Path

class PathObservationWrapper(gymnasium.ObservationWrapper):
    """
    Wraps the Dict observation space into a flat Box observation suitable for DQN.
    
    Computes path-relative features:
    - velocity (normalized)
    - distance_to_lane_center (lateral error)
    - heading_error (angle to path direction)
    - distance_to_left/right_boundary (normalized)
    - on_road flag
    - lookahead waypoint directions
    - obstacle ray-cast distances in multiple directions
    """
    
    def __init__(
        self,
        env: gymnasium.Env,
        max_velocity: float = 15.0,
        max_boundary_distance: float = 10.0,
        num_lookahead_points: int = 3,
        lookahead_distances: typing.Tuple[float, ...] = (2.0, 5.0, 10.0),
      
        num_obstacle_rays: int = 8,  # Number of rays around the vehicle
        max_obstacle_distance: float = 20.0,  # Max detection range
    ):
        super().__init__(env)
        
        self.max_velocity = max_velocity
        self.max_boundary_distance = max_boundary_distance
        self.num_lookahead_points = num_lookahead_points
        self.lookahead_distances = lookahead_distances
        
        # Obstacle detection
        self.num_obstacle_rays = num_obstacle_rays
        self.max_obstacle_distance = max_obstacle_distance
        # Ray angles relative to vehicle heading (front, front-left, left, etc.)
        self.ray_angles = np.linspace(0, 2 * np.pi, num_obstacle_rays, endpoint=False)
        
        # Observation space:
        # [velocity, dist_lane_center, heading_error, dist_left, dist_right, on_road]
        # + [lookahead_x, lookahead_y] * num_lookahead_points
        # + [obstacle_ray_distance] * num_obstacle_rays
        obs_dim = 6 + 2 * num_lookahead_points + num_obstacle_rays
        
        self.observation_space = gymnasium.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        
        # Cache for path and obstacles
        self._path: typing.Optional[np.ndarray] = None
        self._path_cumulative_dist: typing.Optional[np.ndarray] = None
        self._obstacles: typing.List = []
    
    def set_path(self, path: np.ndarray) -> None:
        """Set the reference path for computing path-relative observations."""
        self._path = path
        if path is not None and len(path) > 1:
            # Precompute cumulative distances along path
            diffs = np.diff(path, axis=0)
            segment_lengths = np.linalg.norm(diffs, axis=1)
            self._path_cumulative_dist = np.concatenate([[0], np.cumsum(segment_lengths)])
        else:
            self._path_cumulative_dist = None
    
    def set_obstacles(self, obstacles: typing.List) -> None:
        """Set the obstacles for computing obstacle-relative observations."""
        self._obstacles = obstacles if obstacles is not None else []
    
    def _compute_obstacle_rays(
        self,
        position: np.ndarray,
        heading: float,
    ) -> np.ndarray:
        """
        Cast rays in multiple directions and return normalized distances to obstacles.
        
        Returns array of shape (num_obstacle_rays,) with values in [0, 1].
        1.0 = no obstacle detected (max distance)
        0.0 = obstacle at vehicle position
        """
        ray_distances = np.ones(self.num_obstacle_rays, dtype=np.float32)
        
        if not self._obstacles:
            return ray_distances
        
        for i, ray_angle in enumerate(self.ray_angles):
            # Compute ray direction in world frame
            world_angle = heading + ray_angle
            ray_dir = np.array([np.cos(world_angle), np.sin(world_angle)])
            
            min_dist = self.max_obstacle_distance
            
            for obstacle in self._obstacles:
                dist = self._ray_obstacle_distance(position, ray_dir, obstacle)
                if dist is not None and dist < min_dist:
                    min_dist = dist
            
            # Normalize to [0, 1] where 1 = max distance (safe), 0 = at obstacle
            ray_distances[i] = np.clip(min_dist / self.max_obstacle_distance, 0, 1)
        
        return ray_distances
    
    def _ray_obstacle_distance(
        self,
        position: np.ndarray,
        ray_dir: np.ndarray,
        obstacle,
    ) -> typing.Optional[float]:
        """
        Compute distance along ray to an obstacle.
        
        Returns None if ray doesn't intersect obstacle within max distance.
        """
        # Handle Circle obstacles
        if hasattr(obstacle, 'center') and hasattr(obstacle, 'radius'):
            return self._ray_circle_intersection(position, ray_dir, obstacle.center, obstacle.radius)
        
        # Handle Rectangle obstacles  
        if hasattr(obstacle, 'corners'):
            return self._ray_polygon_intersection(position, ray_dir, obstacle.corners)
        
        return None
    
    def _ray_circle_intersection(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        center: np.ndarray,
        radius: float,
    ) -> typing.Optional[float]:
        """
        Ray-circle intersection.
        Returns distance to intersection or None.
        """
        center = np.array(center)
        oc = origin - center
        
        a = np.dot(direction, direction)
        b = 2.0 * np.dot(oc, direction)
        c = np.dot(oc, oc) - radius * radius
        
        discriminant = b * b - 4 * a * c
        
        if discriminant < 0:
            return None
        
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)
        
        # Return closest positive intersection
        if t1 > 0:
            return t1
        if t2 > 0:
            return t2
        return None
    
    def _ray_polygon_intersection(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        corners: np.ndarray,
    ) -> typing.Optional[float]:
        """
        Ray-polygon intersection using edge testing.
        Returns distance to closest intersection or None.
        """
        min_t = None
        n = len(corners)
        
        for i in range(n):
            p1 = corners[i]
            p2 = corners[(i + 1) % n]
            
            edge = p2 - p1
            edge_perp = np.array([-edge[1], edge[0]])
            
            denom = np.dot(direction, edge_perp)
            if abs(denom) < 1e-10:
                continue
            
            t = np.dot(p1 - origin, edge_perp) / denom
            
            if t > 0:
                hit_point = origin + t * direction
                # Check if hit point is on edge segment
                edge_param = np.dot(hit_point - p1, edge) / np.dot(edge, edge)
                if 0 <= edge_param <= 1:
                    if min_t is None or t < min_t:
                        min_t = t
        
        return min_t
    
    def _find_closest_path_point(self, position: np.ndarray) -> typing.Tuple[int, float]:
        """Find the closest point on the path and return index and distance."""
        if self._path is None or len(self._path) < 2:
            return 0, 0.0
        
        distances = np.linalg.norm(self._path - position, axis=1)
        closest_idx = np.argmin(distances)
        return closest_idx, distances[closest_idx]
    
    def _get_path_heading(self, idx: int) -> float:
        """Get the path heading at a given index."""
        if self._path is None or len(self._path) < 2:
            return 0.0
        
        if idx < len(self._path) - 1:
            direction = self._path[idx + 1] - self._path[idx]
        else:
            direction = self._path[idx] - self._path[idx - 1]
        
        return np.arctan2(direction[1], direction[0])
    
    def _get_lookahead_points(
        self,
        position: np.ndarray,
        heading: float,
        closest_idx: int,
    ) -> np.ndarray:
        """
        Get lookahead points along the path relative to vehicle frame.
        
        Returns points in vehicle-local coordinates (forward, left).
        """
        lookahead_points = []
        
        if self._path is None or self._path_cumulative_dist is None:
            # Return zeros if no path
            return np.zeros(2 * self.num_lookahead_points, dtype=np.float32)
        
        # Current distance along path
        current_dist = self._path_cumulative_dist[closest_idx]
        
        for lookahead_dist in self.lookahead_distances:
            target_dist = current_dist + lookahead_dist
            
            # Find the point at target distance
            if target_dist >= self._path_cumulative_dist[-1]:
                # Wrap around for loop tracks
                target_dist = target_dist % self._path_cumulative_dist[-1]
            
            # Find segment containing target distance
            idx = np.searchsorted(self._path_cumulative_dist, target_dist) - 1
            idx = max(0, min(idx, len(self._path) - 2))
            
            # Interpolate position
            segment_start_dist = self._path_cumulative_dist[idx]
            segment_length = self._path_cumulative_dist[idx + 1] - segment_start_dist
            if segment_length > 0:
                t = (target_dist - segment_start_dist) / segment_length
            else:
                t = 0.0
            t = np.clip(t, 0, 1)
            
            world_point = self._path[idx] + t * (self._path[idx + 1] - self._path[idx])
            
            # Transform to vehicle frame
            delta = world_point - position
            cos_h, sin_h = np.cos(-heading), np.sin(-heading)
            local_x = delta[0] * cos_h - delta[1] * sin_h  # forward
            local_y = delta[0] * sin_h + delta[1] * cos_h  # left
            
            # Normalize by lookahead distance
            lookahead_points.extend([
                np.clip(local_x / lookahead_dist, -1, 1),
                np.clip(local_y / lookahead_dist, -1, 1),
            ])
        
        return np.array(lookahead_points, dtype=np.float32)
    
    def observation(self, obs: dict) -> np.ndarray:
        """Convert dict observation to flat array with path-relative features."""
        position = obs["position"]
        heading = obs["heading"][0]
        velocity = obs["velocity"][0]
        
        # Normalize velocity to [-1, 1]
        velocity_norm = np.clip(velocity / self.max_velocity, -1, 1)
        
        # Get lane centering info (already in observation)
        dist_lane_center = obs["distance_to_lane_center"][0]
        lane_width = obs["lane_width"][0]
        dist_lane_center_norm = np.clip(dist_lane_center / (lane_width / 2 + 0.1), -1, 1)
        
        # Compute heading error relative to path
        closest_idx, _ = self._find_closest_path_point(position)
        path_heading = self._get_path_heading(closest_idx)
        heading_error = heading - path_heading
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        heading_error_norm = np.clip(heading_error / np.pi, -1, 1)
        
        # Boundary distances (normalized)
        dist_left = obs["distance_to_left_boundary"][0]
        dist_right = obs["distance_to_right_boundary"][0]
        dist_left_norm = np.clip(dist_left / self.max_boundary_distance, 0, 1)
        dist_right_norm = np.clip(dist_right / self.max_boundary_distance, 0, 1)
        
        # On road flag
        on_road = float(obs["on_road"])
        
        # Lookahead points
        lookahead = self._get_lookahead_points(position, heading, closest_idx)
        
        # Obstacle ray distances
        obstacle_rays = self._compute_obstacle_rays(position, heading)
        
        # Combine all features
        flat_obs = np.array([
            velocity_norm,
            dist_lane_center_norm,
            heading_error_norm,
            dist_left_norm,
            dist_right_norm,
            on_road,
        ], dtype=np.float32)
        
        return np.concatenate([flat_obs, lookahead, obstacle_rays])


class DiscreteActionWrapper(gymnasium.ActionWrapper):
    """
    Converts discrete actions to continuous [steering, throttle, brake] (CARLA-compatible).
    
    Creates a grid of (steering, throttle) combinations for DQN.
    """
    
    def __init__(
        self,
        env: gymnasium.Env,
        n_steering: int = 5,
        n_accel: int = 3,
        max_steering: float = np.pi / 4,
        accel_values: typing.Tuple[float, ...] = (0.3, 0.5, 0.7),
    ):
        super().__init__(env)
        
        self.n_steering = n_steering
        self.n_accel = n_accel
        self.max_steering = max_steering
        
        # Create steering bins
        self.steering_bins = np.linspace(-max_steering, max_steering, n_steering)
        
        # Acceleration bins (in [0, 1] range where 0.5 = coast)
        self.accel_bins = np.array(accel_values)
        
        # Build action map: list of (steering, accel) tuples
        self.action_map = [
            (s, a) for s in self.steering_bins for a in self.accel_bins
        ]
        
        # New discrete action space
        self.action_space = gymnasium.spaces.Discrete(len(self.action_map))
    
    def action(self, action: int) -> np.ndarray:
        steering, throttle = self.action_map[action]
        return np.array([steering, throttle, 0.0, 0.0], dtype=np.float32)


class DrivingRewardWrapper(gymnasium.Wrapper):
    """
    Custom reward wrapper for lane-following driving task.
    
    Reward components (inspired by highway-env):
    - Lane centering: rewards staying centered on the lane
    - Speed reward: rewards maintaining target speed
    - Heading alignment: penalizes pointing away from path direction
    - Progress reward: rewards forward progress along path
    - Collision/off-road penalty: large negative terminal reward
    """
    
    def __init__(
        self,
        env: gymnasium.Env,
        target_velocity: float = 5.0,
        lane_centering_cost: float = 4.0,
        speed_reward_weight: float = 0.3,
        heading_penalty_weight: float = 0.1,
        progress_reward_weight: float = 0.2,
        collision_penalty: float = -10.0,
        off_road_penalty: float = -5.0,
        max_velocity: float = 15.0,
    ):
        super().__init__(env)
        
        self.target_velocity = target_velocity
        self.lane_centering_cost = lane_centering_cost
        self.speed_reward_weight = speed_reward_weight
        self.heading_penalty_weight = heading_penalty_weight
        self.progress_reward_weight = progress_reward_weight
        self.collision_penalty = collision_penalty
        self.off_road_penalty = off_road_penalty
        self.max_velocity = max_velocity
        
        # Track previous position for progress reward
        self._prev_position: typing.Optional[np.ndarray] = None
        self._prev_path_distance: float = 0.0
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # Initialize tracking
        if hasattr(obs, '__getitem__') and 'position' in obs:
            self._prev_position = obs['position'].copy()
        else:
            self._prev_position = None
        self._prev_path_distance = 0.0
        
        return obs, info
    
    def step(self, action):
        obs, original_reward, terminated, truncated, info = self.env.step(action)
        
        # Compute custom reward
        reward = self._compute_reward(obs, info, terminated, truncated)
        
        # Update tracking
        if hasattr(obs, '__getitem__') and 'position' in obs:
            self._prev_position = obs['position'].copy()
        
        return obs, reward, terminated, truncated, info
    
    def _compute_reward(
        self,
        obs: dict,
        info: dict,
        terminated: bool,
        truncated: bool,
    ) -> float:
        """Compute multi-component reward."""
        
        # Handle flat observation from wrapper
        if isinstance(obs, np.ndarray):
            # Observation has been flattened - use info or approximate
            return self._compute_reward_from_flat(obs, info, terminated)
        
        # Get raw observation values
        velocity = obs["velocity"][0]
        dist_lane_center = obs["distance_to_lane_center"][0]
        on_road = obs["on_road"]
        
        # 1. Lane centering reward: 1 / (1 + cost * lateral²)
        # This gives 1.0 when perfectly centered, decreasing as we deviate
        lane_centering_reward = 1.0 / (1.0 + self.lane_centering_cost * dist_lane_center ** 2)
        
        # 2. Speed reward: encourage target velocity
        speed_ratio = velocity / self.target_velocity if self.target_velocity > 0 else 0
        speed_reward = self.speed_reward_weight * np.clip(speed_ratio, 0, 1.2)
        
        # 3. Progress reward (forward movement)
        progress_reward = 0.0
        if self._prev_position is not None and 'position' in obs:
            current_pos = obs['position']
            # Approximate forward progress (could be improved with path projection)
            displacement = np.linalg.norm(current_pos - self._prev_position)
            progress_reward = self.progress_reward_weight * displacement
        
        # 4. Heading penalty (computed from path observation wrapper if available)
        heading_penalty = 0.0  # Will be handled in flat obs version
        
        # Combine rewards
        reward = lane_centering_reward + speed_reward + progress_reward - heading_penalty
        
        # 5. Terminal penalties
        if info.get("collision", False):
            reward += self.collision_penalty
        
        if not on_road:
            reward += self.off_road_penalty
        
        return reward
    
    def _compute_reward_from_flat(
        self,
        obs: np.ndarray,
        info: dict,
        terminated: bool,
    ) -> float:
        """Compute reward when observation is already flattened."""
        # obs layout: [velocity_norm, dist_lane_center_norm, heading_error_norm, 
        #              dist_left_norm, dist_right_norm, on_road, ...lookahead...]
        
        velocity_norm = obs[0]  # in [-1, 1], typically [0, 1] for forward
        dist_lane_center_norm = obs[1]  # in [-1, 1]
        heading_error_norm = obs[2]  # in [-1, 1]
        on_road = obs[5]  # 0 or 1
        
        # 1. Lane centering: penalize deviation from center
        # Convert normalized back to approximate meters (assume lane_width ~4m)
        lateral_approx = abs(dist_lane_center_norm) * 2.0
        lane_centering_reward = 1.0 / (1.0 + self.lane_centering_cost * lateral_approx ** 2)
        
        # 2. Speed reward
        velocity_approx = velocity_norm * self.max_velocity
        speed_ratio = velocity_approx / self.target_velocity if self.target_velocity > 0 else 0
        speed_reward = self.speed_reward_weight * np.clip(speed_ratio, 0, 1.2)
        
        # 3. Heading alignment penalty
        heading_penalty = self.heading_penalty_weight * abs(heading_error_norm)
        
        # 4. Small living reward for staying on track
        living_reward = 0.05 if on_road > 0.5 else 0.0
        
        # Combine
        reward = lane_centering_reward + speed_reward + living_reward - heading_penalty
        
        # Terminal penalties
        if info.get("collision", False):
            reward += self.collision_penalty
        
        if on_road < 0.5:
            reward += self.off_road_penalty
        
        return reward


def make_dqn_env(
    env: gymnasium.Env,
    target_velocity: float = 5.0,
    n_steering: int = 5,
    n_accel: int = 3,
) -> gymnasium.Env:
    """
    Wrap an environment for DQN training.
    
    Applies observation, action, and reward wrappers in the correct order.
    """
    # First apply reward wrapper (needs dict observations)
    env = DrivingRewardWrapper(env, target_velocity=target_velocity)
    
    # Then observation wrapper (flattens observations)
    env = PathObservationWrapper(env)
    
    # Finally action wrapper (discretizes actions)
    env = DiscreteActionWrapper(env, n_steering=n_steering, n_accel=n_accel)
    
    return env


class DQNController:
    """
    DQN-based navigation controller compatible with the standard controller interface.
    
    Can train a DQN model on an environment and then use it for inference
    with the same get_action() interface as other controllers.
    """
    
    def __init__(
        self,
        n_steering: int = 5,
        n_accel: int = 3,
        max_steering: float = np.pi / 4,
        accel_values: typing.Tuple[float, ...] = (0.3, 0.5, 0.7),
        target_velocity: float = 5.0,
        # DQN hyperparameters
        learning_rate: float = 1e-4,
        buffer_size: int = 100_000,
        learning_starts: int = 1000,
        batch_size: int = 64,
        gamma: float = 0.99,
        exploration_fraction: float = 0.3,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        train_freq: int = 4,
        target_update_interval: int = 1000,
    ):
        self.n_steering = n_steering
        self.n_accel = n_accel
        self.max_steering = max_steering
        self.accel_values = accel_values
        self.target_velocity = target_velocity
        
        # Build action map
        self.steering_bins = np.linspace(-max_steering, max_steering, n_steering)
        self.accel_bins = np.array(accel_values)
        self.action_map = [
            (s, a) for s in self.steering_bins for a in self.accel_bins
        ]
        
        # DQN hyperparameters
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.gamma = gamma
        self.exploration_fraction = exploration_fraction
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.train_freq = train_freq
        self.target_update_interval = target_update_interval
        
        # Model (set after training or loading)
        self.model = None
        
        # Observation wrapper for inference
        self._obs_wrapper: typing.Optional[PathObservationWrapper] = None
    
    def _create_obs_wrapper(self, env: gymnasium.Env) -> PathObservationWrapper:
        """Create observation wrapper for processing observations during inference."""
        return PathObservationWrapper(env)
    
    def train(
        self,
        env: gymnasium.Env,
        total_timesteps: int = 50_000,
        log_interval: int = 100,
        progress_bar: bool = True,
        verbose: int = 1,
        tb_log_name: str = "dqn_driving",
        reset_num_timesteps: bool = True,
    ) -> "DQNController":
        """
        Train the DQN model on the given environment.
        
        Args:
            env: The base environment (will be wrapped automatically)
            total_timesteps: Total training steps
            log_interval: Log every N episodes
            progress_bar: Show training progress bar
            verbose: Verbosity level (0=none, 1=info, 2=debug)
            tb_log_name: Tensorboard log name
            reset_num_timesteps: Reset timestep counter
        
        Returns:
            self for chaining
        """
        from stable_baselines3 import DQN
        from stable_baselines3.common.monitor import Monitor
        
        # Wrap environment for training
        wrapped_env = make_dqn_env(
            env,
            target_velocity=self.target_velocity,
            n_steering=self.n_steering,
            n_accel=self.n_accel,
        )
        
        # Set path on observation wrapper
        if hasattr(env, 'global_path'):
            obs_wrapper = wrapped_env
            while not isinstance(obs_wrapper, PathObservationWrapper):
                if hasattr(obs_wrapper, 'env'):
                    obs_wrapper = obs_wrapper.env
                else:
                    break
            if isinstance(obs_wrapper, PathObservationWrapper):
                obs_wrapper.set_path(env.global_path)
                # Also set obstacles if available
                if hasattr(env, 'obstacles'):
                    obs_wrapper.set_obstacles(env.obstacles)
        
        # Add monitor for logging
        wrapped_env = Monitor(wrapped_env)
        
        # Create DQN model
        self.model = DQN(
            policy="MlpPolicy",
            env=wrapped_env,
            learning_rate=self.learning_rate,
            buffer_size=self.buffer_size,
            learning_starts=self.learning_starts,
            batch_size=self.batch_size,
            gamma=self.gamma,
            exploration_fraction=self.exploration_fraction,
            exploration_initial_eps=self.exploration_initial_eps,
            exploration_final_eps=self.exploration_final_eps,
            train_freq=self.train_freq,
            target_update_interval=self.target_update_interval,
            verbose=verbose,
        )
        
        # Train
        self.model.learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval,
            progress_bar=progress_bar,
            reset_num_timesteps=reset_num_timesteps,
            tb_log_name=tb_log_name,
        )
        
        # Store observation wrapper for inference
        self._obs_wrapper = self._create_obs_wrapper(env)
        if hasattr(env, 'obstacles'):
            self._obs_wrapper.set_obstacles(env.obstacles)
        
        return self
    
    def get_action(
        self,
        observation: dict,
        path: np.ndarray = None,
        obstacles: typing.List = None,
        road_network = None,
        max_steering: float = np.pi / 4,
        max_acceleration: float = 3.0,
        deterministic: bool = True,
        **kwargs
    ) -> np.ndarray:
        """
        Get action for the given observation.
        
        Interface matches other controllers (clothoids, stanley, purepursuit).
        
        Args:
            observation: Environment observation dict
            path: Reference path waypoints as (N, 2) array
            obstacles: List of obstacles for ray-casting
            road_network: Not used (kept for interface compatibility)
            max_steering: Maximum steering angle (used to clip output)
            max_acceleration: Not used directly
            deterministic: Whether to use deterministic actions
            
        Returns:
            action: [steering, throttle, brake] array (CARLA-compatible)
        """
        if self.model is None:
            raise RuntimeError("Model not trained or loaded. Call train() or load() first.")
        
        # Ensure observation wrapper has path and obstacles
        if self._obs_wrapper is not None:
            if path is not None:
                self._obs_wrapper.set_path(path)
            if obstacles is not None:
                self._obs_wrapper.set_obstacles(obstacles)
        
        # Convert observation to flat format expected by DQN
        if self._obs_wrapper is not None:
            flat_obs = self._obs_wrapper.observation(observation)
        else:
            # Fallback: create minimal observation
            flat_obs = self._create_minimal_obs(observation)
        
        # Get discrete action from model
        discrete_action, _ = self.model.predict(flat_obs, deterministic=deterministic)
        
        # Convert to continuous action
        steering, throttle = self.action_map[int(discrete_action)]
        
        # Clip steering to max
        steering = np.clip(steering, -max_steering, max_steering)
        
        return np.array([steering, throttle, 0.0, 0.0], dtype=np.float32)
    
    def _create_minimal_obs(self, observation: dict) -> np.ndarray:
        """Create minimal flat observation when wrapper isn't available."""
        velocity = observation["velocity"][0] / 15.0
        dist_center = observation["distance_to_lane_center"][0] / 2.0
        dist_left = observation["distance_to_left_boundary"][0] / 10.0
        dist_right = observation["distance_to_right_boundary"][0] / 10.0
        on_road = float(observation["on_road"])
        
        # Pad with zeros for heading error, lookahead (6 values), and obstacle rays (8 values)
        return np.array([
            velocity, dist_center, 0.0, dist_left, dist_right, on_road,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # 3 lookahead points (x, y each)
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # 8 obstacle rays (1.0 = no obstacle)
        ], dtype=np.float32)
    
    def save(self, path: typing.Union[str, Path]) -> None:
        """
        Save the trained model to disk.
        
        Args:
            path: Path to save the model (without extension)
        """
        if self.model is None:
            raise RuntimeError("No model to save. Train first.")
        
        self.model.save(str(path))
        print(f"Model saved to {path}")
    
    def load(self, path: typing.Union[str, Path]) -> "DQNController":
        """
        Load a trained model from disk.
        
        Args:
            path: Path to the saved model (without extension)
            
        Returns:
            self for chaining
        """
        from stable_baselines3 import DQN
        
        self.model = DQN.load(str(path))
        print(f"Model loaded from {path}")
        
        return self
    
    def draw_debug(
        self,
        env,
        observation: dict,
        path: np.ndarray,
    ) -> None:
        """
        Draw debug visualization.
        
        Shows:
        - Obstacle raycasts with hit distances
        - Lookahead points on the path
        - Current action in the discrete action grid
        - Heading error arrow
        - Lane centering indicator
        """
        if not hasattr(env, 'overlay_manager'):
            return
        
        position = observation["position"]
        heading = observation["heading"][0]
        velocity = observation["velocity"][0]
        
        # Colors
        COLOR_RAY_CLEAR = (100, 200, 100)      # Green - no obstacle
        COLOR_RAY_CLOSE = (255, 100, 100)      # Red - obstacle close
        COLOR_RAY_MED = (255, 200, 100)        # Orange - obstacle medium
        COLOR_LOOKAHEAD = (100, 100, 255)      # Blue - lookahead points
        COLOR_LOOKAHEAD_LINE = (150, 150, 255) # Light blue - lookahead lines
        COLOR_HEADING = (255, 150, 0)          # Orange - heading arrow
        COLOR_PATH_HEADING = (0, 200, 200)     # Cyan - path heading
        COLOR_ACTION_GRID_BG = (50, 50, 50)    # Dark gray
        COLOR_ACTION_SELECTED = (50, 255, 50)  # Green - selected action
        COLOR_ACTION_CELL = (100, 100, 100)    # Gray - action cells
        
        # ===== 1. Draw Obstacle Raycasts =====
        if self._obs_wrapper is not None:
            self._obs_wrapper.set_path(path)
            self._obs_wrapper.set_obstacles(env.obstacles if hasattr(env, 'obstacles') else [])
            
            ray_angles = self._obs_wrapper.ray_angles
            max_ray_dist = self._obs_wrapper.max_obstacle_distance
            
            # Compute ray distances
            ray_distances = self._obs_wrapper._compute_obstacle_rays(position, heading)
            
            for i, ray_angle in enumerate(ray_angles):
                world_angle = heading + ray_angle
                ray_dir = np.array([np.cos(world_angle), np.sin(world_angle)])
                
                # Actual distance (denormalized)
                actual_dist = ray_distances[i] * max_ray_dist
                end_point = position + ray_dir * actual_dist
                
                # Color based on distance (normalized value)
                norm_dist = ray_distances[i]
                if norm_dist > 0.7:
                    color = COLOR_RAY_CLEAR
                elif norm_dist > 0.3:
                    color = COLOR_RAY_MED
                else:
                    color = COLOR_RAY_CLOSE
                
                # Draw ray line
                env.overlay_manager.add_line(
                    start=tuple(position),
                    end=tuple(end_point),
                    color=color,
                    width=1,
                )
                
                # Draw hit point if obstacle detected
                if norm_dist < 1.0:
                    env.overlay_manager.add_circle(
                        center=tuple(end_point),
                        radius=0.3,
                        color=color,
                        width=0,
                    )
        
        # ===== 2. Draw Lookahead Points =====
        if path is not None and len(path) > 1 and self._obs_wrapper is not None:
            # Find closest path point
            closest_idx, _ = self._obs_wrapper._find_closest_path_point(position)
            cumulative_dist = self._obs_wrapper._path_cumulative_dist
            
            if cumulative_dist is not None:
                current_dist = cumulative_dist[closest_idx]
                
                for i, lookahead_dist in enumerate(self._obs_wrapper.lookahead_distances):
                    target_dist = current_dist + lookahead_dist
                    
                    # Wrap around for loop tracks
                    if target_dist >= cumulative_dist[-1]:
                        target_dist = target_dist % cumulative_dist[-1]
                    
                    # Find segment containing target distance
                    idx = np.searchsorted(cumulative_dist, target_dist) - 1
                    idx = max(0, min(idx, len(path) - 2))
                    
                    # Interpolate position
                    segment_start_dist = cumulative_dist[idx]
                    segment_length = cumulative_dist[idx + 1] - segment_start_dist
                    if segment_length > 0:
                        t = (target_dist - segment_start_dist) / segment_length
                    else:
                        t = 0.0
                    t = np.clip(t, 0, 1)
                    
                    world_point = path[idx] + t * (path[idx + 1] - path[idx])
                    
                    # Draw line from vehicle to lookahead point
                    env.overlay_manager.add_line(
                        start=tuple(position),
                        end=tuple(world_point),
                        color=COLOR_LOOKAHEAD_LINE,
                        width=1,
                    )
                    
                    # Draw lookahead point
                    point_size = 0.4 + 0.2 * i  # Larger points for further lookahead
                    env.overlay_manager.add_circle(
                        center=tuple(world_point),
                        radius=point_size,
                        color=COLOR_LOOKAHEAD,
                        width=0,
                    )
                    
                    # Label the lookahead distance
                    env.overlay_manager.add_text(
                        position=(world_point[0] + 0.5, world_point[1] + 0.5),
                        text=f"{lookahead_dist:.0f}m",
                        color=COLOR_LOOKAHEAD,
                        font_size=12,
                    )
        
        # ===== 3. Draw Heading Error Visualization =====
        if path is not None and len(path) > 1 and self._obs_wrapper is not None:
            closest_idx, _ = self._obs_wrapper._find_closest_path_point(position)
            path_heading = self._obs_wrapper._get_path_heading(closest_idx)
            
            arrow_length = 3.0
            
            # Draw vehicle heading arrow (orange)
            vehicle_end = position + arrow_length * np.array([np.cos(heading), np.sin(heading)])
            env.overlay_manager.add_arrow(
                start=tuple(position),
                end=tuple(vehicle_end),
                color=COLOR_HEADING,
                width=2,
                head_size=0.5,
            )
            
            # Draw path heading arrow (cyan)
            path_end = position + arrow_length * np.array([np.cos(path_heading), np.sin(path_heading)])
            env.overlay_manager.add_arrow(
                start=tuple(position),
                end=tuple(path_end),
                color=COLOR_PATH_HEADING,
                width=2,
                head_size=0.5,
            )
        
        # ===== 4. Draw Action Grid and Current Action =====
        current_action_idx = None
        current_steering = 0.0
        current_accel = 0.0
        
        if self.model is not None:
            try:
                if self._obs_wrapper is not None:
                    flat_obs = self._obs_wrapper.observation(observation)
                else:
                    flat_obs = self._create_minimal_obs(observation)
                
                discrete_action, _ = self.model.predict(flat_obs, deterministic=True)
                current_action_idx = int(discrete_action)
                current_steering, current_accel = self.action_map[current_action_idx]
            except Exception:
                pass
        
        # Draw action grid visualization near vehicle
        grid_offset_x = 4.0
        grid_offset_y = 3.0
        cell_size = 0.8
        
        grid_origin = (position[0] + grid_offset_x, position[1] + grid_offset_y)
        
        # Draw grid cells
        for action_idx, (steer, accel) in enumerate(self.action_map):
            # Calculate grid position
            steer_idx = action_idx % self.n_steering
            accel_idx = action_idx // self.n_steering
            
            cell_x = grid_origin[0] + steer_idx * cell_size
            cell_y = grid_origin[1] + accel_idx * cell_size
            
            # Determine cell color
            if action_idx == current_action_idx:
                color = COLOR_ACTION_SELECTED
                width = 0  # Filled
            else:
                color = COLOR_ACTION_CELL
                width = 1  # Outline only
            
            # Draw cell as small circle
            env.overlay_manager.add_circle(
                center=(cell_x + cell_size / 2, cell_y + cell_size / 2),
                radius=cell_size / 2.5,
                color=color,
                width=width,
            )
        
        # Draw grid labels
        env.overlay_manager.add_text(
            position=(grid_origin[0] - 0.5, grid_origin[1] + self.n_accel * cell_size + 0.3),
            text="← Steer →",
            color=(0, 0, 0),
            font_size=10,
        )
        env.overlay_manager.add_text(
            position=(grid_origin[0] - 1.5, grid_origin[1] + cell_size),
            text="Accel",
            color=(0, 0, 0),
            font_size=10,
        )
        
        # ===== 5. Draw Status Text =====
        # Action info
        env.overlay_manager.add_text(
            position=(position[0] + 2, position[1] + 6),
            text=f"DQN: steer={np.degrees(current_steering):.1f}° accel={current_accel:.2f}",
            color=(50, 150, 50),
            font_size=14,
        )
        
        # Velocity
        env.overlay_manager.add_text(
            position=(position[0] + 2, position[1] + 7.5),
            text=f"Velocity: {velocity:.1f} m/s ({velocity * 3.6:.1f} km/h)",
            color=(0, 0, 0),
            font_size=14,
        )
        
        # Lane centering info
        dist_lane_center = observation["distance_to_lane_center"][0]
        on_road = observation["on_road"]
        road_status = "ON ROAD" if on_road else "OFF ROAD"
        road_color = (0, 150, 0) if on_road else (200, 0, 0)
        
        env.overlay_manager.add_text(
            position=(position[0] + 2, position[1] + 9),
            text=f"{road_status} | Lane offset: {dist_lane_center:.2f}m",
            color=road_color,
            font_size=14,
        )
        
        # Heading error
        if path is not None and len(path) > 1 and self._obs_wrapper is not None:
            closest_idx, _ = self._obs_wrapper._find_closest_path_point(position)
            path_heading = self._obs_wrapper._get_path_heading(closest_idx)
            heading_error = heading - path_heading
            heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
            
            env.overlay_manager.add_text(
                position=(position[0] + 2, position[1] + 10.5),
                text=f"Heading error: {np.degrees(heading_error):.1f}°",
                color=(100, 100, 100),
                font_size=14,
            )
