import numpy as np

class PurePursuitController:
    def __init__(
        self,
        lookahead_distance: float = 5.0,
        min_lookahead: float = 2.0,
        max_lookahead: float = 15.0,
        wheelbase: float = 2.5,
        target_velocity: float = 5.0,
        kp_velocity: float = 1.0,
    ):
        self.lookahead_distance = lookahead_distance
        self.wheelbase = wheelbase
        self.target_velocity = target_velocity
        self.kp_velocity = kp_velocity
        
    def find_lookahead_point(
        self, 
        position: np.ndarray, 
        path: np.ndarray,
        velocity: float
    ) -> tuple[np.ndarray, int]:
        """
        Find the lookahead point on the path.
        
        Args:
            position: Current vehicle position (x, y)
            path: Path waypoints as (N, 2) array
            velocity: Current velocity (for adaptive lookahead)
            
        Returns:
            lookahead_point: The target point to steer towards
            lookahead_idx: Index of the lookahead point in the path
        """
        car_to_waypoint_distances = np.linalg.norm(path - position, axis=1)
        closest_idx = np.argmin(car_to_waypoint_distances)
        
        n_points = len(path)
        cumulative_dist = 0.0
        lookahead_idx = closest_idx
        
        for i in range(n_points):
            idx = (closest_idx + i) % n_points
            next_idx = (closest_idx + i + 1) % n_points
            
            segment_dist = np.linalg.norm(path[next_idx] - path[idx])
            cumulative_dist += segment_dist
            
            if cumulative_dist >= self.lookahead_distance:
                lookahead_idx = next_idx
                break
        
        return path[lookahead_idx], lookahead_idx
    
    def compute_steering(
        self,
        position: np.ndarray,
        heading: float,
        lookahead_point: np.ndarray
    ) -> float:
        dx = lookahead_point[0] - position[0]
        dy = lookahead_point[1] - position[1]
        
        distance_to_lookahead = np.sqrt(dx**2 + dy**2)
        if distance_to_lookahead < 0.1:
            return 0.0
        
        # Transform to vehicle coordinate frame
        # Alpha is the angle between vehicle heading and lookahead point
        lookahead_angle = np.arctan2(dy, dx)
        alpha = lookahead_angle - heading
        
        # Normalize to [-pi, pi]
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))
        
        # delta = arctan(2 * L * sin(alpha) / distance_to_lookahead)
        steering = np.arctan2(2.0 * self.wheelbase * np.sin(alpha), distance_to_lookahead)
        
        return steering
    
    def compute_acceleration(self, current_velocity: float) -> float:
        """
        Simple proportional velocity controller.
        
        Args:
            current_velocity: Current vehicle velocity
            
        Returns:
            acceleration: Acceleration command
        """
        velocity_error = self.target_velocity - current_velocity
        return self.kp_velocity * velocity_error
    
    def get_action(
        self,
        observation: dict,
        path: np.ndarray,
        max_steering: float = np.pi / 4,
        max_acceleration: float = 3.0,
        **kwargs
    ) -> np.ndarray:
        """
        Compute control action from observation.
        
        Args:
            observation: Environment observation dict
            path: Path waypoints as (N, 2) array
            max_steering: Maximum steering angle (for clipping)
            max_acceleration: Maximum acceleration (for clipping)
            
        Returns:
            action: [steering, acceleration] array
        """
        position = observation["position"]
        heading = observation["heading"][0]
        velocity = observation["velocity"][0]
        
        lookahead_point, _ = self.find_lookahead_point(position, path, velocity)
        
        steering = self.compute_steering(position, heading, lookahead_point)
        acceleration = self.compute_acceleration(velocity)
        
        steering = np.clip(steering, -max_steering, max_steering)
        acceleration = np.clip(acceleration, -max_acceleration, max_acceleration)
        
        accel_normalized = acceleration / max_acceleration
        # NOTE: convert [-1, 1] to [0, 1]
        accel_action = (accel_normalized + 1.0) / 2.0

        return np.array([steering, accel_action, 0.0, 0.0], dtype=np.float32)
        
    def draw_debug(
        self,
        env,
        observation: dict,
        path: np.ndarray,
    ) -> None:
        position = observation["position"]
        velocity = observation["velocity"][0]
        
        lookahead_point, lookahead_idx = self.find_lookahead_point(
            position, path, velocity
        )
        
        env.overlay_manager.add_circle(
            center=tuple(lookahead_point),
            radius=0.5,
            color=(255, 165, 0),
            width=0,
        )
        
        env.overlay_manager.add_line(
            start=tuple(position),
            end=tuple(lookahead_point),
            color=(255, 165, 0),
            width=2,
        )
    