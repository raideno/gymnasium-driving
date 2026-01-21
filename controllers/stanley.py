"""
Stanley Controller for Autonomous Vehicle Trajectory Tracking.

Implementation based on the paper:
"Autonomous Automobile Trajectory Tracking for Off-Road Driving:
Controller Design, Experimental Validation and Racing"
- Gabriel M. Hoffmann, Claire J. Tomlin, Michael Montemerlo, Sebastian Thrun
- Stanford University, DARPA Grand Challenge 2005

The controller was used on "Stanley", the Stanford Racing Team's entry in the
DARPA Grand Challenge 2005, achieving the fastest completion time and averaging
19.1 mph on desert and mountainous terrain with typical RMS crosstrack error
under 0.1 m.

Key Design Principles:
1. Consider orientation of FRONT WHEELS with respect to trajectory (collocated control)
2. Crosstrack error measured at the guiding (front) wheels, not vehicle body
3. Global asymptotic stability proven for kinematic equations of motion (Theorem 1)
4. Augmented with dynamics compensation for pneumatic tires and steering servo

Key Equations from the paper:
- Kinematic control law (Eq. 5): δ = ψ + arctan(k * e / v)
- Complete steering law (Eq. 9): δ = (ψ - ψ_ss) + arctan(k * e / (k_soft + v)) 
                                    + k_d,yaw * (r_meas - r_traj) 
                                    + k_d,steer * (δ_meas(i) - δ_meas(i+1))
- Steady state yaw (Eq. 8): ψ_ss = k_ag * v * r_traj = m * v * r / (C_y * (1 + a/b))
- Crosstrack error dynamics (Eq. 1): ė = v * sin(ψ - δ)
- Yaw rate (Eq. 2): ψ̇ = -v * sin(δ) / (a + b)
"""

import typing
import dataclasses

import numpy as np


@dataclasses.dataclass
class StanleyDebugInfo:
    """Debug information for visualization."""
    
    # Front wheel position (where crosstrack is measured)
    front_wheel_position: np.ndarray
    
    # Closest point on path to front wheel
    closest_point: np.ndarray
    closest_path_idx: int
    
    # Path tangent direction at closest point
    path_tangent: np.ndarray
    path_heading: float
    
    # Errors
    crosstrack_error: float          # e(t) - lateral distance to path
    heading_error: float             # ψ(t) - vehicle heading relative to path
    
    # Curvature and trajectory info
    path_curvature: float            # ρ of the path at closest point
    trajectory_yaw_rate: float       # r_traj from path curvature
    steady_state_yaw: float          # ψ_ss compensation for curves
    
    # Controller terms (from Equation 9)
    heading_term: float              # (ψ - ψ_ss)
    crosstrack_term: float           # arctan(k * e / (k_soft + v))
    yaw_damping_term: float          # k_d,yaw * (r_meas - r_traj)
    steering_damping_term: float     # k_d,steer * (δ_meas(i) - δ_meas(i+1))
    
    # Final command
    raw_steering: float              # Before saturation
    steering_command: float          # After saturation
    
    # Lookahead point for visualization
    lookahead_point: np.ndarray
    lookahead_distance: float


class StanleyController:
    """
    Stanley Controller for path following.
    
    This implements the complete control law from the Stanford Racing Team paper:
    
    Kinematic Control Law (Equation 5):
        δ(t) = ψ(t) + arctan(k * e(t) / v(t))
        
    where:
        - δ(t): steering angle command
        - ψ(t): vehicle heading relative to closest path segment (heading error)
        - e(t): crosstrack error at front wheels (positive = left of path)
        - v(t): vehicle forward velocity
        - k: crosstrack gain (controls convergence rate)
        
    The control law has proven global asymptotic stability (Theorem 1 in paper).
    
    Complete Steering Law with Dynamics (Equation 9):
        δ(t) = (ψ - ψ_ss) + arctan(k * e / (k_soft + v)) 
               + k_d,yaw * (r_meas - r_traj) 
               + k_d,steer * (δ_meas(i) - δ_meas(i+1))
               
    where:
        - ψ_ss: steady-state yaw for curves (from Eq. 8)
        - k_soft: prevents oversensitivity at low speeds
        - k_d,yaw: yaw rate damping gain
        - r_traj: desired yaw rate from trajectory
        - k_d,steer: steering servo damping gain
        
    The controller uses crosstrack error at the FRONT WHEELS (not vehicle center),
    enabling collocated control of the system.
    """
    
    def __init__(
        self,
        # Core gains from paper
        k: float = 2.5,                    # Crosstrack gain (1/s, convergence rate)
        k_soft: float = 1.0,               # Low-speed softening (m/s)
        
        # Dynamics compensation gains
        k_d_yaw: float = 0.5,              # Yaw rate damping gain
        k_d_steer: float = 0.1,            # Steering servo damping gain
        
        # Vehicle parameters
        wheelbase: float = 2.5,            # Distance from front to rear axle (m)
        front_axle_distance: float = 1.2,  # Distance from CG to front axle (a in paper)
        rear_axle_distance: float = 1.3,   # Distance from CG to rear axle (b in paper)
        
        # Tire and vehicle dynamics (for steady-state yaw calculation)
        vehicle_mass: float = 1500.0,      # kg
        tire_stiffness: float = 145000.0,  # N/rad (C_y, paper found 145 kN/rad for off-road tires)
        
        # Velocity control
        target_velocity: float = 8.0,      # Target velocity (m/s)
        kp_velocity: float = 2.0,          # Proportional gain for velocity
        ki_velocity: float = 0.1,          # Integral gain for velocity
        
        # Lookahead for path (for finding closest segment)
        min_lookahead: float = 2.0,        # Minimum lookahead distance (m)
        max_lookahead: float = 20.0,       # Maximum lookahead distance (m)
        lookahead_ratio: float = 0.5,      # Lookahead = ratio * velocity
        
        # Obstacle avoidance (simple stopping behavior)
        obstacle_stop_distance: float = 5.0,  # Distance to start braking
        obstacle_slow_distance: float = 15.0, # Distance to start slowing
        
        # Control rate
        dt: float = 0.1,                   # Control timestep (s)
    ):
        # Core gains
        self.k = k
        self.k_soft = k_soft
        
        # Dynamics compensation
        self.k_d_yaw = k_d_yaw
        self.k_d_steer = k_d_steer
        
        # Vehicle geometry
        self.wheelbase = wheelbase
        self.a = front_axle_distance  # CG to front axle
        self.b = rear_axle_distance   # CG to rear axle
        
        # Vehicle dynamics
        self.vehicle_mass = vehicle_mass
        self.tire_stiffness = tire_stiffness
        
        # Compute k_ag for steady-state yaw (Equation 8)
        # k_ag = m / (C_y * (1 + a/b))
        self.k_ag = vehicle_mass / (tire_stiffness * (1 + self.a / self.b))
        
        # Velocity control
        self.target_velocity = target_velocity
        self.kp_velocity = kp_velocity
        self.ki_velocity = ki_velocity
        
        # Lookahead
        self.min_lookahead = min_lookahead
        self.max_lookahead = max_lookahead
        self.lookahead_ratio = lookahead_ratio
        
        # Obstacle avoidance
        self.obstacle_stop_distance = obstacle_stop_distance
        self.obstacle_slow_distance = obstacle_slow_distance
        
        # Control timestep
        self.dt = dt
        
        # State for damping terms
        self._prev_steering_meas: float = 0.0
        self._prev_yaw_rate: float = 0.0
        self._velocity_integral: float = 0.0
        
        # Debug info storage
        self.debug_info: typing.Optional[StanleyDebugInfo] = None
        
    def _compute_front_wheel_position(
        self,
        position: np.ndarray,
        heading: float,
    ) -> np.ndarray:
        """
        Compute the position of the front wheels (front axle center).
        
        The Stanley controller measures crosstrack error at the front wheels,
        not at the vehicle center of gravity. This enables collocated control.
        
        Args:
            position: Vehicle CG position (x, y)
            heading: Vehicle heading in radians
            
        Returns:
            Front wheel position (x, y)
        """
        # Front axle is 'a' meters ahead of CG along heading direction
        front_wheel_x = position[0] + self.a * np.cos(heading)
        front_wheel_y = position[1] + self.a * np.sin(heading)
        return np.array([front_wheel_x, front_wheel_y])
    
    def _find_closest_path_point(
        self,
        position: np.ndarray,
        path: np.ndarray,
    ) -> typing.Tuple[np.ndarray, int, np.ndarray, float]:
        """
        Find the closest point on the path and its tangent direction.
        
        Args:
            position: Query position (front wheel position)
            path: Path waypoints as (N, 2) array
            
        Returns:
            closest_point: Closest point on path
            closest_idx: Index of closest waypoint
            tangent: Unit tangent vector at closest point
            path_heading: Heading of path at closest point (radians)
        """
        if len(path) < 2:
            return path[0], 0, np.array([1.0, 0.0]), 0.0
        
        # Find closest waypoint
        distances = np.linalg.norm(path - position, axis=1)
        closest_idx = np.argmin(distances)
        
        # Compute tangent at closest point (forward difference or backward at end)
        if closest_idx < len(path) - 1:
            tangent = path[closest_idx + 1] - path[closest_idx]
        else:
            tangent = path[closest_idx] - path[closest_idx - 1]
        
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm > 1e-6:
            tangent = tangent / tangent_norm
        else:
            tangent = np.array([1.0, 0.0])
        
        path_heading = np.arctan2(tangent[1], tangent[0])
        
        # Project position onto line segment for more accurate closest point
        if closest_idx < len(path) - 1:
            segment_start = path[closest_idx]
            segment_end = path[closest_idx + 1]
            segment_vec = segment_end - segment_start
            segment_len = np.linalg.norm(segment_vec)
            
            if segment_len > 1e-6:
                # Project point onto segment
                t = np.dot(position - segment_start, segment_vec) / (segment_len ** 2)
                t = np.clip(t, 0.0, 1.0)
                closest_point = segment_start + t * segment_vec
            else:
                closest_point = path[closest_idx]
        else:
            closest_point = path[closest_idx]
        
        return closest_point, closest_idx, tangent, path_heading
    
    def _compute_crosstrack_error(
        self,
        front_wheel_pos: np.ndarray,
        closest_point: np.ndarray,
        path_heading: float,
    ) -> float:
        """
        Compute signed crosstrack error.
        
        The crosstrack error e(t) is the lateral distance from the front wheels
        to the nearest point on the path. Positive error means the vehicle is
        to the LEFT of the path.
        
        Args:
            front_wheel_pos: Position of front wheels
            closest_point: Closest point on path
            path_heading: Heading of path at closest point
            
        Returns:
            Signed crosstrack error (positive = left of path)
        """
        # Vector from closest point to front wheel
        error_vec = front_wheel_pos - closest_point
        
        # Normal vector (perpendicular to path, pointing left)
        normal = np.array([-np.sin(path_heading), np.cos(path_heading)])
        
        # Signed crosstrack error (dot product with normal)
        crosstrack_error = np.dot(error_vec, normal)
        
        return crosstrack_error
    
    def _compute_heading_error(
        self,
        vehicle_heading: float,
        path_heading: float,
    ) -> float:
        """
        Compute heading error ψ(t).
        
        This is the angle between the vehicle's heading and the path heading
        at the closest point. Positive means vehicle is pointing left of path.
        
        Args:
            vehicle_heading: Vehicle heading in radians
            path_heading: Path heading at closest point in radians
            
        Returns:
            Heading error in radians, normalized to [-π, π]
        """
        heading_error = vehicle_heading - path_heading
        # Normalize to [-π, π]
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        return heading_error
    
    def _compute_path_curvature(
        self,
        path: np.ndarray,
        idx: int,
    ) -> float:
        """
        Estimate path curvature at a given index using finite differences.
        
        Curvature ρ = dθ/ds where θ is heading and s is arc length.
        
        Args:
            path: Path waypoints as (N, 2) array
            idx: Index to compute curvature at
            
        Returns:
            Estimated curvature (1/m). Positive = turning left.
        """
        n = len(path)
        if n < 3:
            return 0.0
        
        # Use 3-point method for curvature estimation
        # Get indices with wrapping for closed paths
        idx_prev = max(0, idx - 1)
        idx_next = min(n - 1, idx + 1)
        
        if idx_prev == idx or idx_next == idx:
            return 0.0
        
        p_prev = path[idx_prev]
        p_curr = path[idx]
        p_next = path[idx_next]
        
        # Compute headings
        vec_prev = p_curr - p_prev
        vec_next = p_next - p_curr
        
        ds_prev = np.linalg.norm(vec_prev)
        ds_next = np.linalg.norm(vec_next)
        
        if ds_prev < 1e-6 or ds_next < 1e-6:
            return 0.0
        
        heading_prev = np.arctan2(vec_prev[1], vec_prev[0])
        heading_next = np.arctan2(vec_next[1], vec_next[0])
        
        # Change in heading
        dheading = heading_next - heading_prev
        dheading = np.arctan2(np.sin(dheading), np.cos(dheading))
        
        # Arc length
        ds = (ds_prev + ds_next) / 2
        
        # Curvature
        curvature = dheading / ds if ds > 1e-6 else 0.0
        
        return curvature
    
    def _compute_trajectory_yaw_rate(
        self,
        velocity: float,
        curvature: float,
    ) -> float:
        """
        Compute the desired yaw rate from path curvature.
        
        r_traj = v * ρ
        
        Args:
            velocity: Vehicle velocity (m/s)
            curvature: Path curvature (1/m)
            
        Returns:
            Desired yaw rate (rad/s)
        """
        return velocity * curvature
    
    def _compute_steady_state_yaw(
        self,
        velocity: float,
        trajectory_yaw_rate: float,
    ) -> float:
        """
        Compute steady-state yaw offset for curves (Equation 8).
        
        ψ_ss = k_ag * v * r_traj
        
        where k_ag = m / (C_y * (1 + a/b))
        
        This compensates for the vehicle pointing inward on curves to generate
        the required lateral acceleration with front and rear tires.
        
        Args:
            velocity: Vehicle velocity (m/s)
            trajectory_yaw_rate: Desired yaw rate (rad/s)
            
        Returns:
            Steady-state yaw offset (rad)
        """
        return self.k_ag * velocity * trajectory_yaw_rate
    
    def _compute_yaw_rate(
        self,
        velocity: float,
        steering: float,
    ) -> float:
        """
        Estimate current yaw rate from kinematic model (Equation 2).
        
        ψ̇ = v * tan(δ) / (a + b)
        
        Args:
            velocity: Vehicle velocity (m/s)
            steering: Current steering angle (rad)
            
        Returns:
            Estimated yaw rate (rad/s)
        """
        return velocity * np.tan(steering) / self.wheelbase
    
    def compute_steering(
        self,
        position: np.ndarray,
        heading: float,
        velocity: float,
        path: np.ndarray,
        current_steering: float = 0.0,
        max_steering: float = np.pi / 4,
    ) -> float:
        """
        Compute steering command using the Stanley control law.
        
        Implements the complete steering law (Equation 9):
        
        δ(t) = (ψ - ψ_ss) + arctan(k * e / (k_soft + v)) 
               + k_d,yaw * (r_meas - r_traj) 
               + k_d,steer * (δ_meas(i) - δ_meas(i+1))
        
        Args:
            position: Vehicle CG position (x, y)
            heading: Vehicle heading (radians)
            velocity: Vehicle velocity (m/s)
            path: Path waypoints as (N, 2) array
            current_steering: Current measured steering angle (for damping)
            max_steering: Maximum steering angle (for saturation)
            
        Returns:
            Steering command (radians)
        """
        # Step 1: Compute front wheel position (collocated sensing)
        front_wheel_pos = self._compute_front_wheel_position(position, heading)
        
        # Step 2: Find closest point on path and path properties
        closest_point, closest_idx, path_tangent, path_heading = \
            self._find_closest_path_point(front_wheel_pos, path)
        
        # Step 3: Compute errors
        crosstrack_error = self._compute_crosstrack_error(
            front_wheel_pos, closest_point, path_heading
        )
        heading_error = self._compute_heading_error(heading, path_heading)
        
        # Step 4: Compute path curvature and trajectory yaw rate
        path_curvature = self._compute_path_curvature(path, closest_idx)
        trajectory_yaw_rate = self._compute_trajectory_yaw_rate(velocity, path_curvature)
        
        # Step 5: Compute steady-state yaw compensation (Equation 8)
        steady_state_yaw = self._compute_steady_state_yaw(velocity, trajectory_yaw_rate)
        
        # Step 6: Compute current yaw rate from kinematic model
        current_yaw_rate = self._compute_yaw_rate(velocity, current_steering)
        
        # Step 7: Compute control law terms (Equation 9)
        
        # Term 1: Corrected heading error (ψ - ψ_ss)
        heading_term = heading_error - steady_state_yaw
        
        # Term 2: Crosstrack correction arctan(k * e / (k_soft + v))
        # The k_soft term prevents gain from becoming too large at low speeds
        effective_velocity = self.k_soft + abs(velocity)
        crosstrack_term = np.arctan(self.k * crosstrack_error / effective_velocity)
        
        # Term 3: Yaw rate damping k_d,yaw * (r_meas - r_traj)
        # Provides active damping as tire damping diminishes at higher speeds
        yaw_damping_term = self.k_d_yaw * (current_yaw_rate - trajectory_yaw_rate)
        
        # Term 4: Steering servo damping k_d,steer * (δ_meas(i) - δ_meas(i+1))
        # Provides lead control to damp steering wheel response
        steering_damping_term = self.k_d_steer * (self._prev_steering_meas - current_steering)
        
        # Update previous steering for next iteration
        self._prev_steering_meas = current_steering
        
        # Step 8: Sum all terms for complete steering command
        raw_steering = heading_term + crosstrack_term + yaw_damping_term + steering_damping_term
        
        # Step 9: Apply saturation (from Equation 5)
        steering_command = np.clip(raw_steering, -max_steering, max_steering)
        
        # Compute lookahead point for visualization
        lookahead = np.clip(
            self.lookahead_ratio * abs(velocity),
            self.min_lookahead,
            self.max_lookahead
        )
        lookahead_point = self._find_lookahead_point(front_wheel_pos, path, lookahead)
        
        # Store debug info
        self.debug_info = StanleyDebugInfo(
            front_wheel_position=front_wheel_pos,
            closest_point=closest_point,
            closest_path_idx=closest_idx,
            path_tangent=path_tangent,
            path_heading=path_heading,
            crosstrack_error=crosstrack_error,
            heading_error=heading_error,
            path_curvature=path_curvature,
            trajectory_yaw_rate=trajectory_yaw_rate,
            steady_state_yaw=steady_state_yaw,
            heading_term=heading_term,
            crosstrack_term=crosstrack_term,
            yaw_damping_term=yaw_damping_term,
            steering_damping_term=steering_damping_term,
            raw_steering=raw_steering,
            steering_command=steering_command,
            lookahead_point=lookahead_point,
            lookahead_distance=lookahead,
        )
        
        return steering_command
    
    def _find_lookahead_point(
        self,
        position: np.ndarray,
        path: np.ndarray,
        lookahead_distance: float,
    ) -> np.ndarray:
        """
        Find a point on the path at approximately the lookahead distance.
        
        Args:
            position: Current position
            path: Path waypoints
            lookahead_distance: Distance to look ahead
            
        Returns:
            Lookahead point on path
        """
        if len(path) < 2:
            return path[0]
        
        # Find closest point
        distances = np.linalg.norm(path - position, axis=1)
        closest_idx = np.argmin(distances)
        
        # Walk along path to find lookahead point
        cumulative_dist = 0.0
        lookahead_idx = closest_idx
        
        for i in range(len(path) - 1):
            idx = (closest_idx + i) % len(path)
            next_idx = (closest_idx + i + 1) % len(path)
            
            segment_dist = np.linalg.norm(path[next_idx] - path[idx])
            cumulative_dist += segment_dist
            
            if cumulative_dist >= lookahead_distance:
                lookahead_idx = next_idx
                break
        
        return path[lookahead_idx]
    
    def _check_obstacles_ahead(
        self,
        position: np.ndarray,
        heading: float,
        velocity: float,
        obstacles: typing.List,
    ) -> typing.Tuple[float, bool]:
        """
        Check for obstacles ahead and compute speed adjustment.
        
        Args:
            position: Vehicle position
            heading: Vehicle heading
            velocity: Current velocity
            obstacles: List of obstacles
            
        Returns:
            min_obstacle_distance: Distance to nearest obstacle ahead
            should_emergency_brake: Whether to perform emergency braking
        """
        if not obstacles:
            return float('inf'), False
        
        min_distance = float('inf')
        
        # Check in a cone ahead of the vehicle
        for obs in obstacles:
            obs_center = np.array(obs.center)
            to_obs = obs_center - position
            distance = np.linalg.norm(to_obs)
            
            if distance < 0.1:
                continue
            
            # Check if obstacle is ahead (within ±60 degrees)
            to_obs_heading = np.arctan2(to_obs[1], to_obs[0])
            angle_diff = to_obs_heading - heading
            angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
            
            if abs(angle_diff) < np.pi / 3:  # Within ±60 degrees
                # Subtract obstacle radius for distance to surface
                if hasattr(obs, 'radius'):
                    surface_distance = distance - obs.radius
                else:
                    # Rectangle - approximate
                    surface_distance = distance - max(obs.width, obs.height) / 2
                
                min_distance = min(min_distance, surface_distance)
        
        # Emergency braking if too close
        should_emergency_brake = min_distance < self.obstacle_stop_distance
        
        return min_distance, should_emergency_brake
    
    def compute_acceleration(
        self,
        current_velocity: float,
        obstacle_distance: float = float('inf'),
        emergency_brake: bool = False,
    ) -> float:
        """
        Compute acceleration command using PI controller.
        
        From the paper: The brake and throttle are actuated by a switching
        proportional integral (PI) controller.
        
        Args:
            current_velocity: Current vehicle velocity (m/s)
            obstacle_distance: Distance to nearest obstacle (m)
            emergency_brake: Whether to perform emergency braking
            
        Returns:
            Acceleration command (m/s^2)
        """
        if emergency_brake:
            # Emergency braking - maximum deceleration
            return -3.0
        
        # Adjust target velocity based on obstacle distance
        if obstacle_distance < self.obstacle_slow_distance:
            # Reduce speed when approaching obstacles
            speed_factor = (obstacle_distance - self.obstacle_stop_distance) / \
                          (self.obstacle_slow_distance - self.obstacle_stop_distance)
            speed_factor = np.clip(speed_factor, 0.0, 1.0)
            adjusted_target = self.target_velocity * speed_factor
        else:
            adjusted_target = self.target_velocity
        
        # PI velocity control
        velocity_error = adjusted_target - current_velocity
        
        # Proportional term
        p_term = self.kp_velocity * velocity_error
        
        # Integral term (with anti-windup)
        self._velocity_integral += velocity_error * self.dt
        self._velocity_integral = np.clip(self._velocity_integral, -5.0, 5.0)
        i_term = self.ki_velocity * self._velocity_integral
        
        return p_term + i_term
    
    def get_action(
        self,
        observation: dict,
        path: np.ndarray,
        obstacles: typing.List = None,
        max_steering: float = np.pi / 4,
        max_acceleration: float = 3.0,
    ) -> np.ndarray:
        """
        Compute control action from observation.
        
        Args:
            observation: Environment observation dict with:
                - position: (2,) array in meters
                - heading: (1,) array in radians
                - velocity: (1,) array in m/s
            path: Path waypoints as (N, 2) array
            obstacles: List of obstacles (optional)
            max_steering: Maximum steering angle (radians)
            max_acceleration: Maximum acceleration (m/s^2)
            
        Returns:
            action: [steering, acceleration] array
        """
        position = observation["position"]
        heading = observation["heading"][0]
        velocity = observation["velocity"][0]
        
        # Check for obstacles
        obstacles = obstacles or []
        obstacle_distance, emergency_brake = self._check_obstacles_ahead(
            position, heading, velocity, obstacles
        )
        
        # Compute steering
        # Use estimated current steering from previous command
        current_steering = self._prev_steering_meas
        
        steering = self.compute_steering(
            position=position,
            heading=heading,
            velocity=velocity,
            path=path,
            current_steering=current_steering,
            max_steering=max_steering,
        )
        
        # Compute acceleration
        acceleration = self.compute_acceleration(
            current_velocity=velocity,
            obstacle_distance=obstacle_distance,
            emergency_brake=emergency_brake,
        )
        
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
        
        Shows:
        - Front wheel position (where crosstrack is measured)
        - Crosstrack error line
        - Closest point on path
        - Heading error visualization
        - Path tangent and curvature
        - Lookahead point
        - Controller status text
        
        Args:
            env: The BicycleCarEnv instance (must have overlay methods)
            observation: Environment observation dict
            path: Path waypoints as (N, 2) array
        """
        if self.debug_info is None:
            return
        
        info = self.debug_info
        position = observation["position"]
        heading = observation["heading"][0]
        velocity = observation["velocity"][0]
        
        # Colors
        COLOR_FRONT_WHEEL = (0, 200, 200)      # Cyan
        COLOR_CROSSTRACK = (255, 100, 100)     # Light red
        COLOR_CLOSEST = (255, 200, 0)          # Orange
        COLOR_PATH_TANGENT = (100, 255, 100)   # Light green
        COLOR_LOOKAHEAD = (200, 100, 255)      # Purple
        COLOR_HEADING = (100, 100, 255)        # Blue
        COLOR_TEXT = (0, 0, 0)                 # Black
        
        # Draw front wheel position
        env.overlay_manager.add_circle(
            center=tuple(info.front_wheel_position),
            radius=0.3,
            color=COLOR_FRONT_WHEEL,
            width=0,
        )
        
        # Draw closest point on path
        env.overlay_manager.add_circle(
            center=tuple(info.closest_point),
            radius=0.4,
            color=COLOR_CLOSEST,
            width=0,
        )
        
        # Draw crosstrack error line (from closest point to front wheel)
        env.overlay_manager.add_line(
            start=tuple(info.closest_point),
            end=tuple(info.front_wheel_position),
            color=COLOR_CROSSTRACK,
            width=2,
        )
        
        # Draw path tangent at closest point
        tangent_length = 3.0
        tangent_end = info.closest_point + info.path_tangent * tangent_length
        env.overlay_manager.add_arrow(
            start=tuple(info.closest_point),
            end=tuple(tangent_end),
            color=COLOR_PATH_TANGENT,
            width=2,
            head_size=0.5,
        )
        
        # Draw vehicle heading vector
        heading_length = 3.0
        heading_end = info.front_wheel_position + np.array([
            np.cos(heading) * heading_length,
            np.sin(heading) * heading_length
        ])
        env.overlay_manager.add_arrow(
            start=tuple(info.front_wheel_position),
            end=tuple(heading_end),
            color=COLOR_HEADING,
            width=2,
            head_size=0.5,
        )
        
        # Draw lookahead point
        env.overlay_manager.add_circle(
            center=tuple(info.lookahead_point),
            radius=0.5,
            color=COLOR_LOOKAHEAD,
            width=0,
        )
        
        # Draw line to lookahead
        env.overlay_manager.add_line(
            start=tuple(info.front_wheel_position),
            end=tuple(info.lookahead_point),
            color=COLOR_LOOKAHEAD,
            width=1,
        )
        
        # Draw curvature visualization (arc showing turn direction)
        if abs(info.path_curvature) > 0.01:
            # Draw a small arc to indicate curvature direction
            radius = min(5.0, 1.0 / abs(info.path_curvature))
            turn_dir = "LEFT" if info.path_curvature > 0 else "RIGHT"
            curvature_text = f"ρ={info.path_curvature:.3f} ({turn_dir})"
        else:
            curvature_text = "ρ≈0 (STRAIGHT)"
        
        # Draw status text
        text_offset_x = 3
        text_offset_y = 3
        
        # Error information
        error_text = f"e={info.crosstrack_error:.2f}m ψ={np.degrees(info.heading_error):.1f}°"
        env.overlay_manager.add_text(
            position=(position[0] + text_offset_x, position[1] + text_offset_y),
            text=error_text,
            color=COLOR_TEXT,
            font_size=16,
        )
        
        # Control terms
        ctrl_text = (
            f"δ_head={np.degrees(info.heading_term):.1f}° "
            f"δ_cross={np.degrees(info.crosstrack_term):.1f}°"
        )
        env.overlay_manager.add_text(
            position=(position[0] + text_offset_x, position[1] + text_offset_y + 2),
            text=ctrl_text,
            color=COLOR_TEXT,
            font_size=16,
        )
        
        # Curvature and velocity
        curv_text = f"{curvature_text} v={velocity:.1f}m/s"
        env.overlay_manager.add_text(
            position=(position[0] + text_offset_x, position[1] + text_offset_y + 4),
            text=curv_text,
            color=COLOR_TEXT,
            font_size=16,
        )
        
        # Steering command
        steer_text = f"δ_cmd={np.degrees(info.steering_command):.1f}°"
        env.overlay_manager.add_text(
            position=(position[0] + text_offset_x, position[1] + text_offset_y + 6),
            text=steer_text,
            color=(0, 100, 0) if abs(info.steering_command) < 0.3 else (200, 100, 0),
            font_size=16,
        )
        
        # Draw legend
        legend_items = [
            ("● Front Wheel", COLOR_FRONT_WHEEL),
            ("● Closest Point", COLOR_CLOSEST),
            ("— Crosstrack Error", COLOR_CROSSTRACK),
            ("→ Path Tangent", COLOR_PATH_TANGENT),
            ("→ Heading", COLOR_HEADING),
            ("● Lookahead", COLOR_LOOKAHEAD),
        ]
        
        legend_x = position[0] - 15
        legend_y = position[1] + text_offset_y
        
        for i, (label, color) in enumerate(legend_items):
            env.overlay_manager.add_text(
                position=(legend_x, legend_y + i * 1.5),
                text=label,
                color=color,
                font_size=14,
            )
