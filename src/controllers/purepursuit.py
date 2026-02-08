import numpy as np

class PurePursuitController:
    def __init__(
        self,
        environment,
        lookahead_distance: float = 5.0,
        wheelbase: float = 2.5,
        target_velocity: float = 5.0,
        kp_velocity: float = 1.0,
        max_steering: float = np.pi / 4,
        max_acceleration: float = 3.0,
        **kwargs,
    ):
        self.env = environment
        self.lookahead_distance = lookahead_distance
        self.wheelbase = wheelbase
        self.target_velocity = target_velocity
        self.kp_velocity = kp_velocity
        self.max_steering = max_steering
        self.max_acceleration = max_acceleration
        
        self._lookahead_point_ego: np.ndarray | None = None
        
    def train(self, **kwargs):
        return self
    
    def _find_lookahead_point_ego(self, waypoints_ego: np.ndarray) -> np.ndarray:
        """
        Args:
            waypoints_ego: Waypoints in ego frame, shape (N, 3) with [x, y, curvature]
        Returns:
            Lookahead point [x, y] in ego frame
        """
        # Waypoints are already ordered ahead of the vehicle
        # Find the first waypoint beyond lookahead distance
        cumulative_dist = 0.0
        prev_point = np.array([0.0, 0.0])  # Ego position in ego frame is origin
        
        for i, wp in enumerate(waypoints_ego):
            point = wp[:2]  # [x, y]
            
            if i == 0:
                dist = np.linalg.norm(point)
            else:
                dist = np.linalg.norm(point - prev_point)
            
            cumulative_dist += dist
            
            if cumulative_dist >= self.lookahead_distance:
                return point
            
            prev_point = point
        
        # If no waypoint is far enough, use the last one
        return waypoints_ego[-1, :2]
    
    def _compute_steering_ego(self, lookahead_point_ego: np.ndarray) -> float:
        """
        Compute steering angle from lookahead point in ego frame.
        
        In ego frame, the vehicle is at origin facing +x direction.
        This simplifies the Pure Pursuit formula.
        
        Args:
            lookahead_point_ego: Target point [x, y] in ego frame
            
        Returns:
            Steering angle in radians
        """
        dx, dy = lookahead_point_ego[0], lookahead_point_ego[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        if distance < 0.1:
            return 0.0
        
        # In ego frame, alpha is simply arctan2(dy, dx) since heading is 0
        alpha = np.arctan2(dy, dx)
        
        # Pure Pursuit: delta = arctan(2 * L * sin(alpha) / distance)
        steering = np.arctan2(2.0 * self.wheelbase * np.sin(alpha), distance)
        
        return steering
    
    def _compute_acceleration(self, current_velocity: float) -> float:
        """Proportional velocity controller."""
        velocity_error = self.target_velocity - current_velocity
        return self.kp_velocity * velocity_error
    
    def get_action(
        self,
        observation: dict,
        **kwargs
    ) -> np.ndarray:
        """
        Compute control action from observation.
        
        Args:
            observation: Environment observation dict with:
                - base/velocity: Current velocity
                - path/waypoints: Upcoming waypoints in ego frame (N, 3)
            
        Returns:
            Action array [steering, throttle, 0.0, 0.0]
        """
        velocity = observation["base/velocity"][0]
        waypoints_ego = observation["path/waypoints"]  # Shape: (num_waypoints, 3)
        
        # Find lookahead point in ego frame
        self._lookahead_point_ego = self._find_lookahead_point_ego(waypoints_ego)
        
        # Compute controls
        steering = self._compute_steering_ego(self._lookahead_point_ego)
        acceleration = self._compute_acceleration(velocity)
        
        # Clip to limits
        steering = np.clip(steering, -self.max_steering, self.max_steering)
        acceleration = np.clip(acceleration, 0.0, self.max_acceleration)
        
        # Normalize throttle to [0, 1]
        throttle = acceleration / self.max_acceleration

        return np.array([steering, throttle, 0.0, 0.0], dtype=np.float32), None
        
    def predict(self, observation: dict, **kwargs):
        return self.get_action(observation, **kwargs)
    
    def learn(self, **kwargs):
        return self
    
    def draw_debug(self, env, observation: dict) -> None:
        """Draw debug visualization for the controller."""
        if self._lookahead_point_ego is None:
            return
        
        # Transform lookahead point from ego frame back to world frame for visualization
        position = observation["base/position"]
        heading = observation["base/heading"][0]
        
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        rotation = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
        
        lookahead_world = rotation @ self._lookahead_point_ego + position
        
        env.unwrapped.overlay_manager.add_circle(
            center=tuple(lookahead_world),
            radius=0.5,
            color=(255, 165, 0),
            width=0,
        )
        
        env.unwrapped.overlay_manager.add_line(
            start=tuple(position),
            end=tuple(lookahead_world),
            color=(255, 165, 0),
            width=2,
        )
    