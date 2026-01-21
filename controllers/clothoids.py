import typing
import dataclasses

import numpy as np

from scipy import integrate

@dataclasses.dataclass
class Tentacle:
    # Tentacle points in world coordinates (N, 2)
    points: np.ndarray
    
    # Curvature parameters
    initial_curvature: float  # ρ0
    curvature_rate: float     # Δρ/Δl
    final_curvature: float    # ρend
    
    # Classification
    is_navigable: bool = True
    distance_to_obstacle: float = float('inf')  # L0 in paper
    
    # Selection criteria (normalized to [0, 1])
    clearance_score: float = 0.0      # V_clearance
    curvature_score: float = 0.0      # V_curvature  
    trajectory_score: float = 0.0     # V_trajectory
    combined_score: float = 0.0       # V_combined
    
    # Index in the tentacle set
    index: int = 0


class ClothoidTentaclesController:
    """
    Clothoid Tentacles path planner and controller.
    
    This implements the complete navigation strategy from the paper:
    - Clothoid tentacles generation
    - Obstacle-based tentacle classification
    - Best tentacle selection using multi-criteria optimization
    - I&I-based trajectory tracking controller
    """
    
    def __init__(
        self,
        # Tentacle generation parameters
        num_tentacles: int = 41,
        t0: float = 7.0,           # Time horizon constant (seconds)
        l0: float = 5.0,           # Length offset constant (meters)
        min_tentacle_length: float = 2.0,  # Minimum length at low speeds
        num_points_per_tentacle: int = 50,
        
        # Vehicle parameters
        wheelbase: float = 2.5,    # Vehicle wheelbase (meters)
        max_lateral_accel: float = 4.0,   # Maximum lateral acceleration (m/s^2)
        max_decel: float = 1.5,    # Maximum comfortable deceleration (m/s^2)
        
        # Classification zone parameters
        base_dc: float = 1.4,      # Base classification zone width
        dc_low_speed_factor: float = 0.2,   # Factor for Vx < 3 m/s
        dc_high_speed_factor: float = 0.6,  # Factor for Vx >= 3 m/s
        
        # Best tentacle selection weights
        weight_clearance: float = 0.1,     # a0 in paper
        weight_curvature: float = 0.2,     # a1 in paper
        weight_trajectory: float = 0.5,    # a2 in paper
        
        # Trajectory criterion parameters
        trajectory_distance_scale: float = 0.3,  # ca in paper (m/rad)
        
        # I&I Controller parameters
        lambda_param: float = 1.0,    # Rate of lateral error convergence
        k_param: float = 2.0,         # Rate of z convergence to zero
        
        # Velocity control
        target_velocity: float = 6.0,  # Target velocity (m/s)
        kp_velocity: float = 2.0,      # Proportional gain for velocity
        
        # Vehicle dynamics parameters (for I&I controller)
        vehicle_mass: float = 1500.0,      # kg
        front_cornering_stiffness: float = 80000.0,  # N/rad
        rear_cornering_stiffness: float = 80000.0,   # N/rad
        front_axle_distance: float = 1.2,  # meters from CG
        rear_axle_distance: float = 1.3,   # meters from CG
    ):
        # Tentacle generation
        self.num_tentacles = num_tentacles
        self.t0 = t0
        self.l0 = l0
        self.min_tentacle_length = min_tentacle_length
        self.num_points = num_points_per_tentacle
        
        # Vehicle parameters
        self.wheelbase = wheelbase
        self.max_lateral_accel = max_lateral_accel
        self.max_decel = max_decel
        
        # Classification
        self.base_dc = base_dc
        self.dc_low_speed_factor = dc_low_speed_factor
        self.dc_high_speed_factor = dc_high_speed_factor
        
        # Selection weights
        self.weight_clearance = weight_clearance
        self.weight_curvature = weight_curvature
        self.weight_trajectory = weight_trajectory
        
        # Trajectory parameters
        self.trajectory_distance_scale = trajectory_distance_scale
        
        # I&I Controller
        self.lambda_param = lambda_param
        self.k_param = k_param
        
        # Velocity control
        self.target_velocity = target_velocity
        self.kp_velocity = kp_velocity
        
        # Vehicle dynamics
        self.vehicle_mass = vehicle_mass
        self.cf = front_cornering_stiffness
        self.cr = rear_cornering_stiffness
        self.lf = front_axle_distance
        self.lr = rear_axle_distance
        
        # State for visualization and debugging
        self.tentacles: typing.List[Tentacle] = []
        self.navigable_tentacles: typing.List[Tentacle] = []
        self.best_tentacle: typing.Optional[Tentacle] = None
        self.classification_radius: float = 0.0
        
        # Store previous state for I&I controller
        self._prev_lateral_error: float = 0.0
        self._prev_time: typing.Optional[float] = None
        
        # Store current steering for tentacle generation
        self._current_steering: float = 0.0
    
    def _compute_tentacle_length(self, velocity: float) -> float:
        """
        Calculate tentacle length based on velocity.
        
        From paper: L_tentacle = t0 * Vx - L0 for Vx > 1 m/s
        """
        if velocity <= 1.0:
            return self.min_tentacle_length
        return self.t0 * velocity - self.l0
    
    def _compute_collision_distance(self, velocity: float) -> float:
        """
        Calculate the collision/braking distance.
        
        From paper: Lc = Vx^2 / ac_max
        This is the minimum distance needed to stop safely.
        """
        if velocity <= 0.1:
            return 1.0  # Minimum collision distance
        return (velocity ** 2) / self.max_decel
    
    def _compute_max_curvature(self, velocity: float) -> float:
        """
        Calculate maximum feasible curvature for stability.
        
        From paper: ρmax = amax / Vx^2
        This ensures the lateral acceleration stays within limits.
        """
        if velocity <= 0.5:
            return 0.5  # Maximum curvature at very low speeds
        return self.max_lateral_accel / (velocity ** 2)
    
    def _compute_initial_curvature(self, steering_angle: float) -> float:
        """
        Calculate initial curvature from current steering angle.
        
        From paper: ρ0 = tan(δ0) / L
        """
        return np.tan(steering_angle) / self.wheelbase
    
    def _compute_classification_width(self, velocity: float) -> float:
        """
        Calculate the classification zone half-width.
        
        From paper:
        dc = 1.4 + 0.2 * Vx / 3          for Vx < 3 m/s
        dc = 1.6 + 0.6 * (Vx - 3) / 15   for Vx >= 3 m/s
        """
        if velocity < 3.0:
            return self.base_dc + self.dc_low_speed_factor * velocity / 3.0
        else:
            return 1.6 + self.dc_high_speed_factor * (velocity - 3.0) / 15.0
    
    def _generate_clothoid_points(
        self,
        initial_curvature: float,
        curvature_rate: float,
        length: float,
        initial_heading: float = 0.0,
        initial_position: typing.Tuple[float, float] = (0.0, 0.0),
    ) -> np.ndarray:
        """
        Generate clothoid curve points using Fresnel integrals.
        
        The clothoid is defined by:
        ρ(s) = ρ0 + (Δρ/Δl) * s
        
        Points are computed by integrating:
        x(s) = x0 + ∫cos(θ(t))dt
        y(s) = y0 + ∫sin(θ(t))dt
        where θ(s) = φ0 + ρ0*s + (Δρ/Δl)*s^2/2
        """
        points = np.zeros((self.num_points, 2))
        arc_lengths = np.linspace(0, length, self.num_points)
        
        x0, y0 = initial_position
        phi0 = initial_heading
        rho0 = initial_curvature
        drho_dl = curvature_rate
        
        for i, s in enumerate(arc_lengths):
            if i == 0:
                points[i] = [x0, y0]
                continue
            
            # Integrate from 0 to s using numerical integration
            def integrand_x(t):
                theta = phi0 + rho0 * t + 0.5 * drho_dl * t * t
                return np.cos(theta)
            
            def integrand_y(t):
                theta = phi0 + rho0 * t + 0.5 * drho_dl * t * t
                return np.sin(theta)
            
            # Use Simpson's rule for numerical integration
            x_integral, _ = integrate.quad(integrand_x, 0, s)
            y_integral, _ = integrate.quad(integrand_y, 0, s)
            
            points[i] = [x0 + x_integral, y0 + y_integral]
        
        return points
    
    def _transform_to_world(
        self,
        local_points: np.ndarray,
        position: np.ndarray,
        heading: float,
    ) -> np.ndarray:
        """Transform points from vehicle-local to world coordinates."""
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        
        # Rotation matrix
        rotation = np.array([
            [cos_h, -sin_h],
            [sin_h, cos_h]
        ])
        
        # Transform each point
        world_points = np.zeros_like(local_points)
        for i, point in enumerate(local_points):
            world_points[i] = position + rotation @ point
        
        return world_points
    
    def generate_tentacles(
        self,
        position: np.ndarray,
        heading: float,
        velocity: float,
        steering_angle: float,
    ) -> typing.List[Tentacle]:
        """
        Generate the full set of clothoid tentacles.
        
        All tentacles:
        - Start from vehicle position with current heading
        - Have the same initial curvature (from current steering)
        - Have varying curvature rates to explore different paths
        - Reach curvatures from -ρmax to +ρmax at distance Lc
        """
        tentacles = []
        
        # Calculate parameters based on current state
        tentacle_length = self._compute_tentacle_length(abs(velocity))
        collision_distance = self._compute_collision_distance(abs(velocity))
        max_curvature = self._compute_max_curvature(max(abs(velocity), 0.5))
        initial_curvature = self._compute_initial_curvature(steering_angle)
        
        # Store classification radius for visualization
        self.classification_radius = self._compute_classification_width(abs(velocity))
        
        # Generate curvature rates for each tentacle
        # At distance Lc, curvatures should span from -ρmax to +ρmax
        # Δρ/Δl = (ρ_target - ρ0) / Lc
        
        # Ensure collision_distance is not too small
        lc = max(collision_distance, 1.0)
        
        for i in range(self.num_tentacles):
            # Linear interpolation of target curvature at Lc
            t = i / (self.num_tentacles - 1) if self.num_tentacles > 1 else 0.5
            target_curvature_at_lc = -max_curvature + 2 * max_curvature * t
            
            # Calculate curvature rate
            curvature_rate = (target_curvature_at_lc - initial_curvature) / lc
            
            # Calculate final curvature at end of tentacle
            final_curvature = initial_curvature + curvature_rate * tentacle_length
            
            # Generate clothoid points in local frame
            local_points = self._generate_clothoid_points(
                initial_curvature=initial_curvature,
                curvature_rate=curvature_rate,
                length=tentacle_length,
                initial_heading=0.0,
                initial_position=(0.0, 0.0),
            )
            
            # Transform to world coordinates
            world_points = self._transform_to_world(
                local_points, position, heading
            )
            
            tentacle = Tentacle(
                points=world_points,
                initial_curvature=initial_curvature,
                curvature_rate=curvature_rate,
                final_curvature=final_curvature,
                index=i,
            )
            tentacles.append(tentacle)
        
        self.tentacles = tentacles
        return tentacles
    
    def classify_tentacles(
        self,
        tentacles: typing.List[Tentacle],
        obstacles: typing.List,
        velocity: float,
    ) -> typing.List[Tentacle]:
        """
        Classify tentacles as navigable or non-navigable.
        
        A tentacle is non-navigable if any obstacle is within the 
        classification zone (dc) and closer than the collision distance (Lc).
        """
        dc = self._compute_classification_width(abs(velocity))
        lc = self._compute_collision_distance(abs(velocity))
        
        navigable = []
        
        for tentacle in tentacles:
            is_navigable = True
            min_obstacle_distance = float('inf')
            
            # Check each point along the tentacle
            cumulative_distance = 0.0
            
            for i in range(len(tentacle.points) - 1):
                p1 = tentacle.points[i]
                p2 = tentacle.points[i + 1]
                segment_length = np.linalg.norm(p2 - p1)
                
                # Check if we're still within collision distance
                if cumulative_distance > lc:
                    break
                
                # Check against all obstacles
                for obstacle in obstacles:
                    obs_center = np.array(obstacle.center)
                    
                    # Get obstacle effective radius
                    if hasattr(obstacle, 'radius'):
                        obs_radius = obstacle.radius
                    else:
                        # Rectangle - use half diagonal as radius
                        obs_radius = np.sqrt(
                            (obstacle.width / 2) ** 2 + 
                            (obstacle.height / 2) ** 2
                        )
                    
                    # Distance from point to obstacle surface
                    dist_to_center = np.linalg.norm(p1 - obs_center)
                    dist_to_surface = dist_to_center - obs_radius
                    
                    # Update minimum distance
                    if dist_to_surface < min_obstacle_distance:
                        min_obstacle_distance = dist_to_surface
                    
                    # Check if obstacle is within classification zone
                    if dist_to_surface < dc and cumulative_distance < lc:
                        is_navigable = False
                
                cumulative_distance += segment_length
            
            tentacle.is_navigable = is_navigable
            tentacle.distance_to_obstacle = max(0.0, min_obstacle_distance)
            
            if is_navigable:
                navigable.append(tentacle)
        
        self.navigable_tentacles = navigable
        return navigable
    
    def compute_clearance_score(
        self,
        tentacle: Tentacle,
        l_half: float = 20.0,
    ) -> float:
        """
        Compute clearance criterion V_clearance.
        
        From paper:
        V_clearance = 0 if tentacle is completely free
        V_clearance = 2 - 2/(1 + e^(-c*L0)) otherwise
        
        where c = ln(1/3) / L_0.5 and L_0.5 is distance where score = 0.5
        """
        if tentacle.distance_to_obstacle >= 1000.0:  # Effectively infinite
            return 0.0
        
        c = np.log(1.0 / 3.0) / l_half
        score = 2.0 - 2.0 / (1.0 + np.exp(-c * tentacle.distance_to_obstacle))
        return score
    
    def compute_curvature_score(
        self,
        tentacle: Tentacle,
        velocity: float,
    ) -> float:
        """
        Compute curvature change criterion V_curvature.
        
        From paper: V_curvature = |Δρ/Δl| / (2 * ρmax / Lc)
        
        This penalizes large curvature changes, promoting smoother paths.
        """
        max_curvature = self._compute_max_curvature(max(abs(velocity), 0.5))
        lc = self._compute_collision_distance(abs(velocity))
        
        max_curvature_rate = 2.0 * max_curvature / max(lc, 0.1)
        
        if max_curvature_rate < 1e-6:
            return 0.0
        
        return abs(tentacle.curvature_rate) / max_curvature_rate
    
    def compute_trajectory_score(
        self,
        tentacle: Tentacle,
        reference_path: np.ndarray,
        velocity: float,
    ) -> float:
        """
        Compute trajectory following criterion V_trajectory.
        
        From paper:
        V_dist = b + ca * α
        V_trajectory = (V_dist - V_min) / (V_max - V_min)
        
        where b is lateral distance to path and α is heading difference.
        """
        lc = self._compute_collision_distance(abs(velocity))
        
        # Find tentacle point at collision distance
        cumulative_dist = 0.0
        eval_point_idx = 0
        
        for i in range(len(tentacle.points) - 1):
            p1 = tentacle.points[i]
            p2 = tentacle.points[i + 1]
            segment_length = np.linalg.norm(p2 - p1)
            cumulative_dist += segment_length
            
            if cumulative_dist >= lc:
                eval_point_idx = i + 1
                break
        
        eval_point_idx = min(eval_point_idx, len(tentacle.points) - 1)
        eval_point = tentacle.points[eval_point_idx]
        
        # Find closest point on reference path
        if len(reference_path) < 2:
            return 0.0
        
        distances = np.linalg.norm(reference_path - eval_point, axis=1)
        closest_idx = np.argmin(distances)
        b = distances[closest_idx]  # Lateral distance
        
        # Calculate heading difference
        # Get tentacle heading at evaluation point
        if eval_point_idx > 0:
            tentacle_direction = (
                tentacle.points[eval_point_idx] - 
                tentacle.points[eval_point_idx - 1]
            )
        else:
            tentacle_direction = (
                tentacle.points[1] - tentacle.points[0]
            )
        tentacle_heading = np.arctan2(
            tentacle_direction[1], tentacle_direction[0]
        )
        
        # Get path heading at closest point
        if closest_idx < len(reference_path) - 1:
            path_direction = (
                reference_path[closest_idx + 1] - 
                reference_path[closest_idx]
            )
        else:
            path_direction = (
                reference_path[closest_idx] - 
                reference_path[closest_idx - 1]
            )
        path_heading = np.arctan2(path_direction[1], path_direction[0])
        
        # Heading difference (normalized to [-π, π])
        alpha = tentacle_heading - path_heading
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))
        
        # Combined distance metric
        v_dist = b + self.trajectory_distance_scale * abs(alpha)
        
        return v_dist
    
    def select_best_tentacle(
        self,
        navigable_tentacles: typing.List[Tentacle],
        reference_path: np.ndarray,
        velocity: float,
    ) -> typing.Optional[Tentacle]:
        """
        Select the best tentacle using multi-criteria optimization.
        
        From paper:
        V_combined = a0*V_clearance + a1*V_curvature + a2*V_trajectory
        
        The tentacle with lowest V_combined is selected.
        """
        if not navigable_tentacles:
            return None
        
        # Compute all scores
        for tentacle in navigable_tentacles:
            tentacle.clearance_score = self.compute_clearance_score(tentacle)
            tentacle.curvature_score = self.compute_curvature_score(
                tentacle, velocity
            )
            tentacle.trajectory_score = self.compute_trajectory_score(
                tentacle, reference_path, velocity
            )
        
        # Normalize trajectory scores to [0, 1]
        traj_scores = [t.trajectory_score for t in navigable_tentacles]
        min_traj = min(traj_scores)
        max_traj = max(traj_scores)
        range_traj = max_traj - min_traj if max_traj > min_traj else 1.0
        
        for tentacle in navigable_tentacles:
            tentacle.trajectory_score = (
                (tentacle.trajectory_score - min_traj) / range_traj
            )
        
        # Compute combined scores
        for tentacle in navigable_tentacles:
            tentacle.combined_score = (
                self.weight_clearance * tentacle.clearance_score +
                self.weight_curvature * tentacle.curvature_score +
                self.weight_trajectory * tentacle.trajectory_score
            )
        
        # Select tentacle with lowest combined score
        best = min(navigable_tentacles, key=lambda t: t.combined_score)
        self.best_tentacle = best
        return best
    
    def select_emergency_tentacle(
        self,
        tentacles: typing.List[Tentacle],
    ) -> typing.Optional[Tentacle]:
        """
        Select tentacle with maximum clearance when all are blocked.
        
        Used for emergency braking scenario.
        """
        if not tentacles:
            return None
        
        # Find tentacle with greatest distance to first obstacle
        best = max(tentacles, key=lambda t: t.distance_to_obstacle)
        self.best_tentacle = best
        return best
    
    def compute_lateral_error(
        self,
        position: np.ndarray,
        heading: float,
        tentacle: Tentacle,
    ) -> typing.Tuple[float, float, float]:
        """
        Compute lateral error and desired curvature for the controller.
        
        Returns:
            lateral_error: Signed lateral distance to tentacle
            lateral_error_rate: Rate of change of lateral error
            desired_curvature: Curvature at closest point on tentacle
        """
        # Find closest point on tentacle
        distances = np.linalg.norm(tentacle.points - position, axis=1)
        closest_idx = np.argmin(distances)
        closest_point = tentacle.points[closest_idx]
        
        # Compute signed lateral error
        # (positive = vehicle is to the left of tentacle)
        if closest_idx < len(tentacle.points) - 1:
            tangent = (
                tentacle.points[closest_idx + 1] - 
                tentacle.points[closest_idx]
            )
        else:
            tangent = (
                tentacle.points[closest_idx] - 
                tentacle.points[closest_idx - 1]
            )
        tangent = tangent / (np.linalg.norm(tangent) + 1e-6)
        
        # Normal vector (pointing left of tangent)
        normal = np.array([-tangent[1], tangent[0]])
        
        # Vector from closest point to vehicle
        to_vehicle = position - closest_point
        
        # Signed lateral error
        lateral_error = np.dot(to_vehicle, normal)
        
        # Compute lateral error rate (approximation)
        # e_dot ≈ Vx * sin(heading - tentacle_heading) + additional terms
        tentacle_heading = np.arctan2(tangent[1], tangent[0])
        heading_error = heading - tentacle_heading
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        
        # Approximate lateral error rate
        # For now, use simple finite difference if available
        dt = 0.1  # Assume 10Hz update rate
        lateral_error_rate = (lateral_error - self._prev_lateral_error) / dt
        self._prev_lateral_error = lateral_error
        
        # Compute desired curvature at this point
        # Using linear interpolation along tentacle
        if len(tentacle.points) > 1:
            arc_length = closest_idx * (
                np.linalg.norm(tentacle.points[1] - tentacle.points[0])
            )
            desired_curvature = (
                tentacle.initial_curvature + 
                tentacle.curvature_rate * arc_length
            )
        else:
            desired_curvature = tentacle.initial_curvature
        
        return lateral_error, lateral_error_rate, desired_curvature
    
    def compute_steering_ii(
        self,
        position: np.ndarray,
        heading: float,
        velocity: float,
        yaw_rate: float,
        sideslip_angle: float,
        tentacle: Tentacle,
    ) -> float:
        """
        Compute steering angle using I&I (Immersion and Invariance) controller.
        
        From paper, equation (12):
        δ_I&I = -m(K+λ)/Cf * ė - mKλ/Cf * e + (Cf+Cr)/Cf * β
                + (LfCf - LrCr)/(Cf*Vx) * ψ̇ + mVx²/(Cf) * ρ
        
        where:
        - m: vehicle mass
        - K, λ: controller gains
        - Cf, Cr: cornering stiffness
        - Lf, Lr: axle distances from CG
        - e, ė: lateral error and its derivative
        - β: sideslip angle
        - ψ̇: yaw rate
        - ρ: desired curvature
        """
        # Get lateral error and desired curvature
        e, e_dot, rho = self.compute_lateral_error(position, heading, tentacle)
        
        # Ensure velocity is not too small (avoid division by zero)
        vx = max(abs(velocity), 0.5)
        
        # I&I control law
        m = self.vehicle_mass
        K = self.k_param
        lam = self.lambda_param
        Cf = self.cf
        Cr = self.cr
        Lf = self.lf
        Lr = self.lr
        
        # Compute steering command
        term1 = -m * (K + lam) / Cf * e_dot
        term2 = -m * K * lam / Cf * e
        term3 = (Cf + Cr) / Cf * sideslip_angle
        term4 = (Lf * Cf - Lr * Cr) / (Cf * vx) * yaw_rate
        term5 = m * vx * vx / Cf * rho
        
        steering = term1 + term2 + term3 + term4 + term5
        
        return steering
    
    def compute_steering_simple(
        self,
        position: np.ndarray,
        heading: float,
        velocity: float,
        tentacle: Tentacle,
    ) -> float:
        """
        Simplified steering controller (pure pursuit-like).
        
        Used when full vehicle dynamics are not available.
        """
        # Find lookahead point on tentacle
        lookahead_distance = max(2.0, abs(velocity) * 0.5)
        
        cumulative_dist = 0.0
        lookahead_idx = 0
        
        for i in range(len(tentacle.points) - 1):
            p1 = tentacle.points[i]
            p2 = tentacle.points[i + 1]
            segment_length = np.linalg.norm(p2 - p1)
            cumulative_dist += segment_length
            
            if cumulative_dist >= lookahead_distance:
                lookahead_idx = i + 1
                break
        
        lookahead_idx = min(lookahead_idx, len(tentacle.points) - 1)
        lookahead_point = tentacle.points[lookahead_idx]
        
        # Vector from vehicle to lookahead point
        dx = lookahead_point[0] - position[0]
        dy = lookahead_point[1] - position[1]
        
        # Distance to lookahead point
        ld = np.sqrt(dx**2 + dy**2)
        if ld < 0.1:
            return 0.0
        
        # Transform to vehicle coordinate frame
        lookahead_angle = np.arctan2(dy, dx)
        alpha = lookahead_angle - heading
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))
        
        # Pure pursuit steering formula
        steering = np.arctan2(2.0 * self.wheelbase * np.sin(alpha), ld)
        
        # Store current steering for next iteration
        self._current_steering = steering
        
        return steering
    
    def compute_acceleration(
        self,
        current_velocity: float,
        has_navigable_path: bool,
        emergency_braking: bool = False,
    ) -> float:
        """
        Compute acceleration command.
        
        Args:
            current_velocity: Current vehicle velocity
            has_navigable_path: Whether a navigable tentacle exists
            emergency_braking: Whether to perform emergency braking
        """
        if emergency_braking or not has_navigable_path:
            # Emergency braking with maximum comfortable deceleration
            return -self.max_decel
        
        # Normal velocity control
        velocity_error = self.target_velocity - current_velocity
        return self.kp_velocity * velocity_error
    
    def get_action(
        self,
        observation: dict,
        path: np.ndarray,
        obstacles: typing.List,
        max_steering: float = np.pi / 4,
        max_acceleration: float = 3.0,
        use_simple_steering: bool = True,
    ) -> np.ndarray:
        """
        Main entry point: compute control action from observation.
        
        Args:
            observation: Environment observation dict with:
                - position: (2,) array
                - heading: (1,) array in radians
                - velocity: (1,) array in m/s
            path: Reference path waypoints as (N, 2) array
            obstacles: List of obstacles from environment
            max_steering: Maximum steering angle
            max_acceleration: Maximum acceleration
            use_simple_steering: Use simplified steering instead of I&I
            
        Returns:
            action: [steering, acceleration] array
        """
        position = observation["position"]
        heading = observation["heading"][0]
        velocity = observation["velocity"][0]
        
        # Use previous steering angle for tentacle generation
        steering_angle = self._current_steering
        
        # Step 1: Generate tentacles
        tentacles = self.generate_tentacles(
            position=position,
            heading=heading,
            velocity=velocity,
            steering_angle=steering_angle,
        )
        
        # Step 2: Classify tentacles based on obstacles
        navigable = self.classify_tentacles(
            tentacles=tentacles,
            obstacles=obstacles,
            velocity=velocity,
        )
        
        # Step 3: Select best tentacle or emergency tentacle
        if navigable:
            best_tentacle = self.select_best_tentacle(
                navigable_tentacles=navigable,
                reference_path=path,
                velocity=velocity,
            )
            emergency_braking = False
        else:
            # No navigable path - select tentacle with most clearance
            best_tentacle = self.select_emergency_tentacle(tentacles)
            emergency_braking = True
        
        # Step 4: Compute control commands
        if best_tentacle is not None:
            if use_simple_steering:
                steering = self.compute_steering_simple(
                    position=position,
                    heading=heading,
                    velocity=velocity,
                    tentacle=best_tentacle,
                )
            else:
                # Use I&I controller (requires more vehicle state)
                yaw_rate = 0.0  # Would need to be computed/observed
                sideslip = 0.0  # Would need to be computed/observed
                steering = self.compute_steering_ii(
                    position=position,
                    heading=heading,
                    velocity=velocity,
                    yaw_rate=yaw_rate,
                    sideslip_angle=sideslip,
                    tentacle=best_tentacle,
                )
        else:
            steering = 0.0
        
        acceleration = self.compute_acceleration(
            current_velocity=velocity,
            has_navigable_path=len(navigable) > 0,
            emergency_braking=emergency_braking,
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
        - All tentacles (gray for non-navigable, light blue for navigable)
        - Classification zones around tentacles
        - Best tentacle (green, thicker)
        - Scores for best tentacle
        """
        position = observation["position"]
        velocity = observation["velocity"][0]
        
        # Colors
        COLOR_NON_NAVIGABLE = (150, 150, 150)     # Gray
        COLOR_NAVIGABLE = (100, 180, 255)         # Light blue
        COLOR_BEST = (50, 255, 50)                # Green
        COLOR_CLASSIFICATION = (255, 200, 100)   # Light orange
        COLOR_EMERGENCY = (255, 100, 100)         # Red
        
        # Draw all tentacles
        for tentacle in self.tentacles:
            if tentacle is self.best_tentacle:
                continue  # Draw best tentacle last
            
            color = COLOR_NAVIGABLE if tentacle.is_navigable else COLOR_NON_NAVIGABLE
            width = 1
            
            # Draw tentacle path
            if len(tentacle.points) >= 2:
                env.overlay_manager.add_path(
                    points=tentacle.points,
                    color=color,
                    width=width,
                    closed=False,
                )
        
        # Draw best tentacle with classification zone
        if self.best_tentacle is not None:
            tentacle = self.best_tentacle
            
            # Determine color based on whether we're in emergency mode
            is_emergency = not tentacle.is_navigable
            color = COLOR_EMERGENCY if is_emergency else COLOR_BEST
            
            # Draw classification zone as circles at key points
            num_zone_circles = 8
            step = max(1, len(tentacle.points) // num_zone_circles)
            for i in range(0, len(tentacle.points), step):
                point = tentacle.points[i]
                env.overlay_manager.add_circle(
                    center=tuple(point),
                    radius=self.classification_radius,
                    color=COLOR_CLASSIFICATION,
                    width=1,
                )
            
            # Draw the tentacle path (thicker)
            if len(tentacle.points) >= 2:
                env.overlay_manager.add_path(
                    points=tentacle.points,
                    color=color,
                    width=3,
                    closed=False,
                )
            
            # Draw endpoint
            end_point = tentacle.points[-1]
            env.overlay_manager.add_circle(
                center=tuple(end_point),
                radius=0.5,
                color=color,
                width=0,
            )
            
            # Draw collision distance marker
            lc = self._compute_collision_distance(abs(velocity))
            cumulative_dist = 0.0
            for i in range(len(tentacle.points) - 1):
                p1 = tentacle.points[i]
                p2 = tentacle.points[i + 1]
                segment_length = np.linalg.norm(p2 - p1)
                cumulative_dist += segment_length
                
                if cumulative_dist >= lc:
                    env.overlay_manager.add_circle(
                        center=tuple(p2),
                        radius=0.3,
                        color=(255, 255, 0),  # Yellow
                        width=0,
                    )
                    break
            
            # Draw score info near vehicle
            score_text = (
                f"Clear:{tentacle.clearance_score:.2f} "
                f"Curv:{tentacle.curvature_score:.2f} "
                f"Traj:{tentacle.trajectory_score:.2f}"
            )
            text_pos = (position[0] + 2, position[1] + 4)
            env.overlay_manager.add_text(
                position=text_pos,
                text=score_text,
                color=(0, 0, 0),
                font_size=16,
            )
            
            # Draw status text
            if is_emergency:
                status_text = "EMERGENCY BRAKING"
                status_color = (255, 0, 0)
            else:
                status_text = f"Navigable: {len(self.navigable_tentacles)}/{len(self.tentacles)}"
                status_color = (0, 100, 0)
            
            status_pos = (position[0] + 2, position[1] + 6)
            env.overlay_manager.add_text(
                position=status_pos,
                text=status_text,
                color=status_color,
                font_size=16,
            )
        
        # Draw tentacle count indicator
        env.overlay_manager.add_text(
            position=(position[0] + 2, position[1] + 2),
            text=f"Tentacles: {len(self.tentacles)}",
            color=(0, 0, 0),
            font_size=16,
        )
