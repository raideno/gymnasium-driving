import numpy as np

class PIDController:
    """
    Simple PID controller for path following.
    
    Uses PID control for both lateral tracking and velocity control:
    - Lateral control: PID on cross-track error and heading error
    - Velocity control: PID on velocity error
    """
    
    def __init__(
        self,
        # Lateral control gains
        kp_lateral: float = 1.0,
        ki_lateral: float = 0.0,
        kd_lateral: float = 0.5,
        
        # Heading control gains
        kp_heading: float = 2.0,
        ki_heading: float = 0.0,
        kd_heading: float = 0.1,
        
        # Velocity control gains
        kp_velocity: float = 2.0,
        ki_velocity: float = 0.1,
        kd_velocity: float = 0.0,
        
        # Vehicle parameters
        wheelbase: float = 2.5,
        
        # Target velocity
        target_velocity: float = 5.0,
        
        # Lookahead distance for path tracking
        lookahead_distance: float = 5.0,
    ):
        # Store gains
        self.kp_lateral = kp_lateral
        self.ki_lateral = ki_lateral
        self.kd_lateral = kd_lateral
        
        self.kp_heading = kp_heading
        self.ki_heading = ki_heading
        self.kd_heading = kd_heading
        
        self.kp_velocity = kp_velocity
        self.ki_velocity = ki_velocity
        self.kd_velocity = kd_velocity
        
        self.wheelbase = wheelbase
        self.target_velocity = target_velocity
        self.lookahead_distance = lookahead_distance
        
        # Error tracking for integral and derivative terms
        self.lateral_error_integral = 0.0
        self.lateral_error_prev = 0.0
        
        self.heading_error_integral = 0.0
        self.heading_error_prev = 0.0
        
        self.velocity_error_integral = 0.0
        self.velocity_error_prev = 0.0
        
        # Time step (will be updated)
        self.dt = 0.1
        
    def find_closest_point(
        self, 
        position: np.ndarray, 
        path: np.ndarray
    ) -> tuple[int, np.ndarray, float]:
        """
        Find the closest point on the path to the vehicle.
        
        Returns:
            closest_idx: Index of closest point
            closest_point: The closest point coordinates
            cross_track_error: Signed distance from path (positive = left of path)
        """
        # Find closest point
        distances = np.linalg.norm(path - position, axis=1)
        closest_idx = np.argmin(distances)
        closest_point = path[closest_idx]
        
        # Compute cross-track error (signed distance from path)
        # Get path tangent direction at closest point
        next_idx = (closest_idx + 1) % len(path)
        path_tangent = path[next_idx] - closest_point
        path_tangent = path_tangent / (np.linalg.norm(path_tangent) + 1e-6)
        
        # Vector from closest point to vehicle
        to_vehicle = position - closest_point
        
        # Cross product to get signed distance (positive = left of path)
        cross_track_error = np.cross(path_tangent, to_vehicle)
        
        return closest_idx, closest_point, cross_track_error
    
    def find_lookahead_point(
        self,
        position: np.ndarray,
        path: np.ndarray,
        closest_idx: int,
    ) -> tuple[np.ndarray, float]:
        """
        Find a point ahead on the path for heading reference.
        
        Returns:
            lookahead_point: Point on path ahead of vehicle
            path_heading: Heading angle of the path at lookahead point
        """
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
        
        lookahead_point = path[lookahead_idx]
        
        # Compute path heading at lookahead point
        next_idx = (lookahead_idx + 1) % len(path)
        dx = path[next_idx][0] - lookahead_point[0]
        dy = path[next_idx][1] - lookahead_point[1]
        path_heading = np.arctan2(dy, dx)
        
        return lookahead_point, path_heading
    
    def compute_steering(
        self,
        position: np.ndarray,
        heading: float,
        path: np.ndarray,
        dt: float,
    ) -> float:
        """
        Compute steering angle using PID control on cross-track and heading errors.
        """
        # Find closest point and cross-track error
        closest_idx, closest_point, cross_track_error = self.find_closest_point(
            position, path
        )
        
        # Find lookahead point and desired heading
        lookahead_point, path_heading = self.find_lookahead_point(
            position, path, closest_idx
        )
        
        # Compute heading error (normalize to [-pi, pi])
        heading_error = path_heading - heading
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        
        # Update integral terms
        self.lateral_error_integral += cross_track_error * dt
        self.heading_error_integral += heading_error * dt
        
        # Compute derivative terms
        lateral_error_derivative = (cross_track_error - self.lateral_error_prev) / (dt + 1e-6)
        heading_error_derivative = (heading_error - self.heading_error_prev) / (dt + 1e-6)
        
        # PID control for lateral error
        lateral_control = (
            self.kp_lateral * cross_track_error +
            self.ki_lateral * self.lateral_error_integral +
            self.kd_lateral * lateral_error_derivative
        )
        
        # PID control for heading error
        heading_control = (
            self.kp_heading * heading_error +
            self.ki_heading * self.heading_error_integral +
            self.kd_heading * heading_error_derivative
        )
        
        # Combine lateral and heading control
        steering = -(lateral_control + heading_control)
        
        # Update previous errors
        self.lateral_error_prev = cross_track_error
        self.heading_error_prev = heading_error
        
        return steering
    
    def compute_acceleration(
        self,
        current_velocity: float,
        dt: float,
    ) -> float:
        """
        Compute acceleration using PID control on velocity error.
        """
        velocity_error = self.target_velocity - current_velocity
        
        # Update integral term
        self.velocity_error_integral += velocity_error * dt
        
        # Compute derivative term
        velocity_error_derivative = (velocity_error - self.velocity_error_prev) / (dt + 1e-6)
        
        # PID control
        acceleration = (
            self.kp_velocity * velocity_error +
            self.ki_velocity * self.velocity_error_integral +
            self.kd_velocity * velocity_error_derivative
        )
        
        # Update previous error
        self.velocity_error_prev = velocity_error
        
        return acceleration
    
    def reset(self):
        """Reset integral and derivative terms."""
        self.lateral_error_integral = 0.0
        self.lateral_error_prev = 0.0
        self.heading_error_integral = 0.0
        self.heading_error_prev = 0.0
        self.velocity_error_integral = 0.0
        self.velocity_error_prev = 0.0
    
    def get_action(
        self,
        observation: dict,
        path: np.ndarray,
        obstacles: list = None,
        max_steering: float = np.pi / 4,
        max_acceleration: float = 3.0,
        dt: float = 0.1,
    ) -> np.ndarray:
        """
        Compute control action from observation.
        
        Args:
            observation: Environment observation dict
            path: Path waypoints as (N, 2) array
            obstacles: List of obstacles (not used in basic PID)
            max_steering: Maximum steering angle (for clipping)
            max_acceleration: Maximum acceleration (for clipping)
            dt: Time step for derivative/integral calculations
            
        Returns:
            action: [steering, acceleration] array
        """
        position = observation["position"]
        heading = observation["heading"][0]
        velocity = observation["velocity"][0]
        
        # Update time step
        self.dt = dt
        
        # Compute controls
        steering = self.compute_steering(position, heading, path, dt)
        acceleration = self.compute_acceleration(velocity, dt)
        
        # Clip to action bounds
        steering = np.clip(steering, -max_steering, max_steering)
        acceleration = np.clip(acceleration, -max_acceleration, max_acceleration)
        # Convert acceleration to [0, 1] range
        accel_normalized = acceleration / max_acceleration
        accel_action = (accel_normalized + 1.0) / 2.0  # Convert [-1, 1] to [0, 1]

        return np.array([steering, accel_action, 0.0, 0.0], dtype=np.float32)
    
    def draw_debug(
        self,
        env,
        observation: dict,
        path: np.ndarray,
    ) -> None:
        """
        Draw debug visualization on the environment.
        
        Args:
            env: The BicycleCarEnv instance (must have overlay methods)
            observation: Environment observation dict
            path: Path waypoints as (N, 2) array
        """
        position = observation["position"]
        
        # Find closest point and lookahead point
        closest_idx, closest_point, cross_track_error = self.find_closest_point(
            position, path
        )
        lookahead_point, _ = self.find_lookahead_point(position, path, closest_idx)
        
        # Draw closest point
        env.overlay_manager.add_circle(
            center=tuple(closest_point),
            radius=0.3,
            color=(0, 255, 0),  # Green
            width=0,
        )
        
        # Draw lookahead point
        env.overlay_manager.add_circle(
            center=tuple(lookahead_point),
            radius=0.5,
            color=(0, 0, 255),  # Blue
            width=0,
        )
        
        # Draw line from vehicle to lookahead point
        env.overlay_manager.add_line(
            start=tuple(position),
            end=tuple(lookahead_point),
            color=(0, 0, 255),  # Blue
            width=2,
        )
        
        # Draw cross-track error line
        env.overlay_manager.add_line(
            start=tuple(position),
            end=tuple(closest_point),
            color=(255, 0, 0),  # Red
            width=1,
        )
