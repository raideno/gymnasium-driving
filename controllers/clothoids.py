import typing
import dataclasses

import numpy as np

from scipy import integrate

@dataclasses.dataclass
class Tentacle:
    # NOTE: [N, 2]
    points: np.ndarray
    
    # p0, Δρ/Δl, ρend
    initial_curvature: float
    curvature_rate: float
    final_curvature: float
    
    is_navigable: bool = True
    distance_to_obstacle: float = float('inf')
    exits_road: bool = False
    
    # V_clearance, V_curvature, V_trajectory, V_combined
    clearance_score: float = 0.0
    curvature_score: float = 0.0
    trajectory_score: float = 0.0
    combined_score: float = 0.0
    
    index: int = 0

class ClothoidTentaclesController:
    def __init__(
        self,
        num_tentacles: int = 41,
        t0: float = 7.0,           # Time horizon constant (seconds)
        l0: float = 5.0,           # Length offset constant (meters)
        min_tentacle_length: float = 2.0,  # Minimum length at low speeds
        num_points_per_tentacle: int = 50,
        
        wheelbase: float = 2.5,
        vehicle_width: float = 1.8,  # Width of the vehicle in meters
        max_lateral_acceleration: float = 4.0,
        max_deceleration: float = 1.5,
        
        # Classification zone parameters
        base_dc: float = 1.4,      # Base classification zone width
        dc_low_speed_factor: float = 0.2,   # Factor for Vx < 3 m/s
        dc_high_speed_factor: float = 0.6,  # Factor for Vx >= 3 m/s
        
        # a0, a1 and a2
        weights: typing.Tuple[float, float, float] = (0.1, 0.2, 0.5),
        
        # Trajectory criterion parameters
        trajectory_distance_scale: float = 0.3,  # ca in paper (m/rad)
        
        target_velocity: float = 6.0,
        kp_velocity: float = 2.0,
    ):
        self.num_tentacles = num_tentacles
        self.t0 = t0
        self.l0 = l0
        self.min_tentacle_length = min_tentacle_length
        self.num_points = num_points_per_tentacle
        
        self.wheelbase = wheelbase
        self.vehicle_width = vehicle_width
        self.max_lateral_acceleration = max_lateral_acceleration
        self.max_deceleration = max_deceleration
        
        self.base_dc = base_dc
        self.dc_low_speed_factor = dc_low_speed_factor
        self.dc_high_speed_factor = dc_high_speed_factor
        
        self.weight_clearance, self.weight_curvature, self.weight_trajectory = weights
        
        # Trajectory parameters
        self.trajectory_distance_scale = trajectory_distance_scale
        
        # Velocity control
        self.target_velocity = target_velocity
        self.kp_velocity = kp_velocity
        
        # State for visualization and debugging
        self.tentacles: typing.List[Tentacle] = []
        self.navigable_tentacles: typing.List[Tentacle] = []
        self.best_tentacle: typing.Optional[Tentacle] = None
        self.classification_radius: float = 0.0
        
        # Store current steering for tentacle generation
        self._current_steering: float = 0.0
    
    def _compute_tentacle_length(self, velocity: float) -> float:
        """
        Equation (7) from paper.
        """
        if velocity > 1.0:
            return self.t0 * velocity - self.l0
        else:
            return self.min_tentacle_length
    
    def _compute_collision_distance(self, velocity: float) -> float:
        """
        Equation (6) from paper.
        """
        # if velocity <= 0.1:
        #     return 1.0
        return (velocity ** 2) / self.max_deceleration
    
    def _compute_max_curvature(self, velocity: float) -> float:
        """
        After equation (5) from paper.
        """
        # if velocity <= 0.5:
        #     return 0.5
        return self.max_lateral_acceleration / (velocity ** 2)
    
    def _compute_initial_curvature(self, steering_angle: float) -> float:
        """
        After equation (5) from paper.
        """
        return np.tan(steering_angle) / self.wheelbase
    
    def _compute_classification_width(self, velocity: float) -> float:
        """
        Before equation (6) from paper.
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
        Equation (1) and (2) from paper.
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
            
            def integrand_x(t):
                theta = phi0 + rho0 * t + 0.5 * drho_dl * t * t
                return np.cos(theta)
            
            def integrand_y(t):
                theta = phi0 + rho0 * t + 0.5 * drho_dl * t * t
                return np.sin(theta)
            
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
        tentacles = []
        
        tentacle_length = self._compute_tentacle_length(abs(velocity))
        collision_distance = self._compute_collision_distance(abs(velocity))
        max_curvature = self._compute_max_curvature(max(abs(velocity), 0.5))
        initial_curvature = self._compute_initial_curvature(steering_angle)
        
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
            
            curvature_rate = (target_curvature_at_lc - initial_curvature) / lc
            
            final_curvature = initial_curvature + curvature_rate * tentacle_length
            
            local_points = self._generate_clothoid_points(
                initial_curvature=initial_curvature,
                curvature_rate=curvature_rate,
                length=tentacle_length,
                initial_heading=0.0,
                initial_position=(0.0, 0.0),
            )
            
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
        road_network=None,
    ) -> typing.List[Tentacle]:
        """
        A tentacle is non-navigable if:
        1. Any obstacle is within the classification zone (dc) and closer than the collision distance (Lc)
        2. It exits the road boundaries (if road_network provided)
        """
        dc = self._compute_classification_width(abs(velocity))
        lc = self._compute_collision_distance(abs(velocity))
        
        navigable = []
        
        for tentacle in tentacles:
            is_navigable = True
            exits_road = False
            min_obstacle_distance = float('inf')
            
            cumulative_distance = 0.0
            
            for i in range(len(tentacle.points) - 1):
                p1 = tentacle.points[i]
                p2 = tentacle.points[i + 1]
                segment_length = np.linalg.norm(p2 - p1)
                
                if cumulative_distance > lc:
                    break
                
                # Check road boundaries - account for vehicle width
                if road_network is not None and not exits_road:
                    # Check centerline
                    if not road_network.contains_point(p1):
                        exits_road = True
                        is_navigable = False
                    else:
                        # Check if vehicle width fits within road
                        # Get direction perpendicular to path segment
                        if i < len(tentacle.points) - 1:
                            tangent = p2 - p1
                            tangent_norm = np.linalg.norm(tangent)
                            if tangent_norm > 1e-6:
                                tangent = tangent / tangent_norm
                                perpendicular = np.array([-tangent[1], tangent[0]])
                                
                                # Check points at vehicle half-width on both sides
                                half_width = self.vehicle_width / 2.0
                                left_point = p1 + perpendicular * half_width
                                right_point = p1 - perpendicular * half_width
                                
                                if not road_network.contains_point(left_point) or \
                                   not road_network.contains_point(right_point):
                                    exits_road = True
                                    is_navigable = False
                
                # Check obstacles
                for obstacle in obstacles:
                    obs_center = np.array(obstacle.center)
                    
                    if hasattr(obstacle, 'radius'):
                        obs_radius = obstacle.radius
                    else:
                        obs_radius = np.sqrt(
                            (obstacle.width / 2) ** 2 + 
                            (obstacle.height / 2) ** 2
                        )
                    
                    dist_to_center = np.linalg.norm(p1 - obs_center)
                    dist_to_surface = dist_to_center - obs_radius
                    
                    if dist_to_surface < min_obstacle_distance:
                        min_obstacle_distance = dist_to_surface
                    
                    if dist_to_surface < dc and cumulative_distance < lc:
                        is_navigable = False
                
                cumulative_distance += segment_length
            
            tentacle.is_navigable = is_navigable
            tentacle.exits_road = exits_road
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
        Equation (7) from paper.
        0 means maximum clearance, no obstacle.
        Higher values mean closer obstacles.
        """
        if tentacle.distance_to_obstacle >= 1000.0:
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
        Equation (7.2) from paper.
        Penalizes large curvature changes, promoting smoother paths.
        0 means no curvature rate (change), 1 means maximum allowed curvature rate.
        """
        max_curvature = self._compute_max_curvature(max(abs(velocity), 0.5))
        lc = self._compute_collision_distance(abs(velocity))
        # max_curvature_rate = 2.0 * max_curvature / max(lc, 0.1)
        # if max_curvature_rate < 1e-6:
        #     return 0.0
        return abs(tentacle.curvature_rate) / (2.0 * max_curvature / max(lc, 0.1))
    
    def compute_trajectory_score(
        self,
        tentacle: Tentacle,
        reference_path: np.ndarray,
        velocity: float,
    ) -> float:
        """
        Equation (8) and (9) from paper.
        
        How well a given tentacle points toward and stays close to the global reference path.
        We pick a point on the tentacle at the collision distance Lc, and we compare it to the reference trajectory using:
        - b: distance to the trajectory (lateral distance to path).
        - alpha: heading difference to the trajectory.
        """
        lc = self._compute_collision_distance(abs(velocity))
        
        # Find tentacle point at collision distance Lc
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
        
        b = distances[closest_idx]
        
        # Calculate heading difference
        ### Get tentacle heading at evaluation point
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
        ### Get path heading at closest point
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
        alpha = tentacle_heading - path_heading
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))
        
        v_dist = b + self.trajectory_distance_scale * abs(alpha)
        
        return v_dist
    
    def select_best_tentacle(
        self,
        navigable_tentacles: typing.List[Tentacle],
        reference_path: np.ndarray,
        velocity: float,
    ) -> typing.Optional[Tentacle]:
        """
        Before equation (7) from paper.
        The tentacle with lowest V_combined is selected.
        """
        if not navigable_tentacles:
            return None
        
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
        
        for tentacle in navigable_tentacles:
            tentacle.combined_score = (
                self.weight_clearance * tentacle.clearance_score +
                self.weight_curvature * tentacle.curvature_score +
                self.weight_trajectory * tentacle.trajectory_score
            )
        
        best = min(navigable_tentacles, key=lambda t: t.combined_score)
        self.best_tentacle = best
        return best
    
    def select_emergency_tentacle(
        self,
        tentacles: typing.List[Tentacle],
    ) -> typing.Optional[Tentacle]:
        if not tentacles:
            return None
        best = max(tentacles, key=lambda t: t.distance_to_obstacle)
        self.best_tentacle = best
        return best
    
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
            return -self.max_deceleration
        
        # Normal velocity control
        velocity_error = self.target_velocity - current_velocity
        return self.kp_velocity * velocity_error
    
    def get_action(
        self,
        observation: dict,
        path: np.ndarray,
        obstacles: typing.List,
        road_network=None,
        max_steering: float = np.pi / 4,
        max_acceleration: float = 3.0,
        **kwargs
    ) -> np.ndarray:
        """
        Args:
            observation: Environment observation dict with:
                - position: (2,) array
                - heading: (1,) array in radians
                - velocity: (1,) array in m/s
            path: Reference path waypoints as (N, 2) array
            obstacles: List of obstacles from environment
            road_network: Optional road network for boundary checking
            max_steering: Maximum steering angle
            max_acceleration: Maximum acceleration
            
        Returns:
            action: [steering, acceleration] array
        """
        position = observation["position"]
        heading = observation["heading"][0]
        velocity = observation["velocity"][0]
        
        steering_angle = self._current_steering
        
        tentacles = self.generate_tentacles(
            position=position,
            heading=heading,
            velocity=velocity,
            steering_angle=steering_angle,
        )
        
        navigable = self.classify_tentacles(
            tentacles=tentacles,
            obstacles=obstacles,
            velocity=velocity,
            road_network=road_network,
        )
        
        if navigable:
            tentacle_to_navigate = self.select_best_tentacle(
                navigable_tentacles=navigable,
                reference_path=path,
                velocity=velocity,
            )
            emergency_braking = False
        else:
            tentacle_to_navigate = self.select_emergency_tentacle(tentacles)
            emergency_braking = True
        
        if tentacle_to_navigate is not None:
            steering = self.compute_steering_simple(
                position=position,
                heading=heading,
                velocity=velocity,
                tentacle=tentacle_to_navigate,
            )
        else:
            steering = 0.0
        
        acceleration = self.compute_acceleration(
            current_velocity=velocity,
            has_navigable_path=len(navigable) > 0,
            emergency_braking=emergency_braking,
        )
        
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
        COLOR_EXITS_ROAD = (255, 150, 0)          # Orange - exits road
        
        # Draw all tentacles
        for tentacle in self.tentacles:
            if tentacle is self.best_tentacle:
                continue  # Draw best tentacle last
            
            # Color based on status
            if tentacle.exits_road:
                color = COLOR_EXITS_ROAD
            elif tentacle.is_navigable:
                color = COLOR_NAVIGABLE
            else:
                color = COLOR_NON_NAVIGABLE
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
