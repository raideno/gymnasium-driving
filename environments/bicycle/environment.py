import typing
import gymnasium

import numpy as np

from .obstacles import *
from .roads import *

from .helpers.renderer import Renderer
from .helpers.overlay import OverlayManager
from .helpers.performance import PerformanceTracker

class BicycleCarEnv(gymnasium.Env):
    """
    A bicycle kinematic model environment with proper meter-based scaling.

    All dimensions are in meters, velocities in m/s.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    DEFAULT_CAR_LENGTH = 4.5
    DEFAULT_CAR_WIDTH = 1.8

    def __init__(
        self,
        render_mode: str | None = None,
        
        # Bicycle model parameters (meters, m/s, m/s^2)
        wheelbase: float = 2.5,
        max_steering: float = np.pi / 4,
        max_velocity: float = 15.0,  # ~54 km/h
        max_acceleration: float = 3.0,
        max_brake_deceleration: float = 6.0,  # ~0.6g braking
        
        # Car dimensions (meters)
        car_length: float | None = None,  # defaults to DEFAULT_CAR_LENGTH
        car_width: float | None = None,   # defaults to DEFAULT_CAR_WIDTH
        
        # Environment parameters (all in meters)
        world_size: typing.Tuple[float, float] | None = None,  # Auto-calculated
        world_padding: float = 10.0,  # Padding around content in meters
        spawn_pos: typing.Tuple[float, float] = (10.0, 10.0),
        spawn_heading: float = 0.0,
        goal_pos: typing.Tuple[float, float] = (90.0, 90.0),
        goal_radius: float = 3.0,
        obstacles: typing.List[typing.Union[Circle, Rectangle]] | None = None,
        road_network: RoadNetwork | None = None,
        enforce_road: bool = False,
        off_road_penalty: float = -10.0,
        solid_road_borders: bool = False,  # If True, road borders are solid obstacles
        
        # Simulation
        dt: float = 0.1,
        
        # Rendering
        screen_size: typing.Tuple[int, int] = (800, 800),
        pixels_per_meter: float | None = None,  # Auto-calculated if None
    ):
        super().__init__()

        self.wheelbase = wheelbase
        self.max_steering = max_steering
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.max_brake_deceleration = max_brake_deceleration
        
        self.car_length = car_length if car_length is not None else self.DEFAULT_CAR_LENGTH
        self.car_width = car_width if car_width is not None else self.DEFAULT_CAR_WIDTH
        
        self.spawn_pos = np.array(spawn_pos, dtype=np.float32)
        self.spawn_heading = spawn_heading
        self.goal_pos = np.array(goal_pos, dtype=np.float32)
        self.goal_radius = goal_radius
        self.dt = dt

        self.obstacles = obstacles if obstacles else []
        self.road_network = road_network
        self.enforce_road = enforce_road
        self.off_road_penalty = off_road_penalty
        self.solid_road_borders = solid_road_borders

        # Calculate world bounds based on content
        self.world_padding = world_padding
        self.world_size, self.world_origin = self._calculate_world_bounds(
            world_size
        )

        self.screen_size = screen_size
        self.render_mode = render_mode

        # Calculate scale (pixels per meter)
        if pixels_per_meter is not None:
            self.pixels_per_meter = pixels_per_meter
        else:
            # Fit world to screen with some margin
            # TODO: move margins into parameters
            scale_x = (screen_size[0] - 40) / self.world_size[0]
            scale_y = (screen_size[1] - 40) / self.world_size[1]
            self.pixels_per_meter = min(scale_x, scale_y)

        # steering, throttle, brake, reverse
        self.action_space = gymnasium.spaces.Box(
            low=np.array([-max_steering, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([max_steering, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.observation_space = gymnasium.spaces.Dict(
            {
                "position": gymnasium.spaces.Box(
                    -np.inf, np.inf, shape=(2,), dtype=np.float32
                ),
                "heading": gymnasium.spaces.Box(
                    -np.pi, np.pi, shape=(1,), dtype=np.float32
                ),
                "velocity": gymnasium.spaces.Box(
                    -max_velocity, max_velocity, shape=(1,), dtype=np.float32
                ),
                "goal": gymnasium.spaces.Box(
                    -np.inf, np.inf, shape=(2,), dtype=np.float32
                ),
                "goal_distance": gymnasium.spaces.Box(
                    0, np.inf, shape=(1,), dtype=np.float32
                ),
                "on_road": gymnasium.spaces.Discrete(2),
                "distance_to_road_center": gymnasium.spaces.Box(
                    0, np.inf, shape=(1,), dtype=np.float32
                ),
                "distance_to_left_boundary": gymnasium.spaces.Box(
                    0, np.inf, shape=(1,), dtype=np.float32
                ),
                "distance_to_right_boundary": gymnasium.spaces.Box(
                    0, np.inf, shape=(1,), dtype=np.float32
                ),
                "lane_number": gymnasium.spaces.Discrete(10),
                "num_lanes": gymnasium.spaces.Discrete(10),
                "lane_width": gymnasium.spaces.Box(
                    0, np.inf, shape=(1,), dtype=np.float32
                ),
                "road_width": gymnasium.spaces.Box(
                    0, np.inf, shape=(1,), dtype=np.float32
                ),
                "distance_to_lane_center": gymnasium.spaces.Box(
                    -np.inf, np.inf, shape=(1,), dtype=np.float32
                ),
                "intersecting_left_boundary": gymnasium.spaces.Discrete(2),
                "intersecting_right_boundary": gymnasium.spaces.Discrete(2),
                "in_single_lane": gymnasium.spaces.Discrete(2),
                "in_multiple_lanes": gymnasium.spaces.Discrete(2),
            }
        )

        self.state = None
        
        self.global_path = None  # Will store waypoints as (N, 2) array
        self._compute_global_path()

        self._episode_data = {
            'actions': [],
            'positions': [],
            'steering_angles': [],
            'on_road': [],
            'velocities': [],
            'headings': [],
            'terminated': False,
            'truncated': False
        }

        self.overlay_manager = OverlayManager()
        self.performance_tracker = PerformanceTracker(show_performance=True)
        self.renderer = Renderer(
            screen_size=screen_size,
            render_mode=render_mode,
            render_fps=self.metadata["render_fps"],
        )

    def _calculate_world_bounds(
        self, explicit_world_size: typing.Tuple[float, float] | None
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        Calculate world bounds based on all content.

        Returns:
            world_size: (width, height) in meters
            world_origin: (min_x, min_y) - bottom-left corner in world coords
        """
        if explicit_world_size is not None:
            return (
                np.array(explicit_world_size, dtype=np.float32),
                np.array([0.0, 0.0], dtype=np.float32),
            )

        # Collect all bounds
        min_x, min_y = float("inf"), float("inf")
        max_x, max_y = float("-inf"), float("-inf")

        # Include spawn position
        min_x = min(min_x, self.spawn_pos[0])
        min_y = min(min_y, self.spawn_pos[1])
        max_x = max(max_x, self.spawn_pos[0])
        max_y = max(max_y, self.spawn_pos[1])

        # Include goal position
        min_x = min(min_x, self.goal_pos[0] - self.goal_radius)
        min_y = min(min_y, self.goal_pos[1] - self.goal_radius)
        max_x = max(max_x, self.goal_pos[0] + self.goal_radius)
        max_y = max(max_y, self.goal_pos[1] + self.goal_radius)

        # Include obstacles
        for obs in self.obstacles:
            bounds = obs.get_bounds()
            min_x = min(min_x, bounds[0])
            min_y = min(min_y, bounds[1])
            max_x = max(max_x, bounds[2])
            max_y = max(max_y, bounds[3])

        # Include roads
        if self.road_network is not None:
            bounds = self.road_network.get_bounds()
            min_x = min(min_x, bounds[0])
            min_y = min(min_y, bounds[1])
            max_x = max(max_x, bounds[2])
            max_y = max(max_y, bounds[3])

        # Apply padding
        min_x -= self.world_padding
        min_y -= self.world_padding
        max_x += self.world_padding
        max_y += self.world_padding

        world_size = np.array([max_x - min_x, max_y - min_y], dtype=np.float32)
        world_origin = np.array([min_x, min_y], dtype=np.float32)

        return world_size, world_origin

    def _world_to_screen(
        self, pos: typing.Union[np.ndarray, typing.Tuple[float, float]]
    ) -> typing.Tuple[int, int]:
        """Convert world coordinates (meters) to screen coordinates (pixels)."""
        if isinstance(pos, tuple):
            pos = np.array(pos, dtype=np.float32)

        # Translate to local coordinates
        local_pos = pos - self.world_origin

        # Scale to pixels
        screen_x = int(local_pos[0] * self.pixels_per_meter) + 20
        screen_y = int(
            self.screen_size[1] - local_pos[1] * self.pixels_per_meter - 20
        )

        return (screen_x, screen_y)

    def _meters_to_pixels(self, meters: float) -> int:
        """Convert a distance in meters to pixels."""
        return max(1, int(meters * self.pixels_per_meter))

    def _compute_global_path(self) -> None:
        """
        Compute a closed-loop global path following the road centerline.
        
        The path starts at spawn, follows the entire road loop in the road's
        direction, and returns to the spawn point. This is ideal for track-based
        scenarios where the vehicle navigates the full circuit.
        """
        if self.road_network is None:
            # Simple straight line from spawn to goal
            self.global_path = np.array([self.spawn_pos, self.goal_pos], dtype=np.float32)
            return
        
        # Extract all centerline points from the road network (preserving order)
        all_centerline_points = []
        
        for road in self.road_network.roads:
            for segment in road.segments:
                # Use more points for smoother paths (increased density for better controller performance)
                num_points = max(256, int(segment.get_length() * 5))
                centerline = segment.get_centerline_points(num_points)
                all_centerline_points.extend(centerline)
        
        if not all_centerline_points:
            # Fallback to straight line
            self.global_path = np.array([self.spawn_pos, self.goal_pos], dtype=np.float32)
            return
        
        centerline_array = np.array(all_centerline_points, dtype=np.float32)
        
        # Find the closest point on centerline to spawn
        spawn_distances = np.linalg.norm(centerline_array - self.spawn_pos, axis=1)
        spawn_idx = np.argmin(spawn_distances)
        
        # Reorder the path to start from spawn_idx and wrap around
        # This creates a full loop starting and ending near spawn
        n_points = len(centerline_array)
        reordered_indices = [(spawn_idx + i) % n_points for i in range(n_points)]
        reordered_path = centerline_array[reordered_indices]
        
        # Build the final path with spawn at start
        path_points = [self.spawn_pos.copy()]
        
        for point in reordered_path:
            # Only add if not too close to the previous point (avoid duplication)
            # Reduced threshold for denser path
            if np.linalg.norm(point - path_points[-1]) > 0.2:
                path_points.append(point)
        
        # Close the loop - return to spawn
        if np.linalg.norm(self.spawn_pos - path_points[-1]) > 0.2:
            path_points.append(self.spawn_pos.copy())
        
        self.global_path = np.array(path_points, dtype=np.float32)
        self.global_path_is_loop = True  # Flag to indicate this is a closed loop

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, typing.Any] | None = None,
    ) -> typing.Tuple[dict[str, typing.Any], dict[str, typing.Any]]:
        super().reset(seed=seed)

        self.state = np.array(
            [
                self.spawn_pos[0],
                self.spawn_pos[1],
                self.spawn_heading,
                0.0,
            ],
            dtype=np.float32,
        )
        
        self.sim_time = 0.0
        for obstacle in self.obstacles:
            obstacle.reset()
        
        # Reset performance tracking
        self.performance_tracker.reset()

        # Clear episode data tracking
        self._episode_data['actions'] = []
        self._episode_data['positions'] = []
        self._episode_data['steering_angles'] = []
        self._episode_data['on_road'] = []
        self._episode_data['velocities'] = []
        self._episode_data['headings'] = []
        self._episode_data['terminated'] = False
        self._episode_data['truncated'] = False

        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(
        self, action: np.ndarray
    ) -> typing.Tuple[
        dict[str, typing.Any], float, bool, bool, dict[str, typing.Any]
    ]:
        # Store data before step for metrics (capture the state before action is applied)
        self._episode_data['positions'].append(self.state[:2].copy())
        self._episode_data['velocities'].append(float(self.state[3]))
        self._episode_data['headings'].append(float(self.state[2]))
        self._episode_data['actions'].append(action.copy())
        self._episode_data['steering_angles'].append(float(action[0]))
        
        steering = np.clip(action[0], -self.max_steering, self.max_steering)
        
        # Throttle: 0 = no acceleration, 1 = full acceleration (CARLA-compatible)
        throttle = np.clip(action[1], 0.0, 1.0)
        
        # Brake: applies deceleration proportional to brake input [0, 1]
        brake = np.clip(action[2], 0.0, 1.0) if len(action) > 2 else 0.0
        
        # Reverse: 0 = forward, non-zero = reverse
        reverse = (action[3] != 0.0) if len(action) > 3 else False
        
        # Apply throttle in the appropriate direction
        if reverse:
            acceleration = -throttle * self.max_acceleration
        else:
            acceleration = throttle * self.max_acceleration

        x, y, theta, v = self.state

        x_new = x + v * np.cos(theta) * self.dt
        y_new = y + v * np.sin(theta) * self.dt
        theta_new = theta + (v / self.wheelbase) * np.tan(steering) * self.dt
        
        v_new = v + acceleration * self.dt
        
        if brake > 0.0 and abs(v_new) > 0.01:
            brake_decel = brake * self.max_brake_deceleration * self.dt
            if v_new > 0:
                v_new = max(0.0, v_new - brake_decel)
            else:
                v_new = min(0.0, v_new + brake_decel)
        
        v_new = np.clip(v_new, -self.max_velocity, self.max_velocity)

        theta_new = np.arctan2(np.sin(theta_new), np.cos(theta_new))

        self.state = np.array([x_new, y_new, theta_new, v_new], dtype=np.float32)
        
        self.sim_time += self.dt
        
        for obstacle in self.obstacles:
            obstacle.update(self.sim_time)
        
        self.performance_tracker.update()

        observation = self._get_observation()
        info = self._get_info()
        
        # TODO: added for tracking
        # Store on_road flag after state update
        self._episode_data['on_road'].append(int(observation["on_road"]))
        # self._episode_on_road_flags.append(int(observation["on_road"]))

        terminated = False
        truncated = False
        reward = 0.0
        
        goal_dist = info["goal_distance"]
        if goal_dist <= self.goal_radius:
            terminated = True
            reward = 100.0
        elif self._check_collision():
            terminated = True
            reward = -100.0
        elif not self._within_world_boundaries():
            truncated = True
            reward = -50.0
        elif self.road_network is not None:
            if self.road_network.is_off_road(self.state[:2]):
                if self.enforce_road:
                    terminated = True
                    reward = -100.0
                else:
                    reward = self.off_road_penalty - 0.01 * goal_dist
            else:
                reward = -0.1 - 0.01 * goal_dist
        # Store episode termination status
        self._episode_data['terminated'] = terminated
        self._episode_data['truncated'] = truncated
        # self._episode_terminated = terminated
        # self._episode_truncated = truncated

        # else:
        #     reward = -0.1 - 0.01 * goal_dist

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> dict[str, typing.Any]:
        goal_dist = np.linalg.norm(self.state[:2] - self.goal_pos)

        on_road = 1
        road_info = {
            "distance_to_road_center": 0.0,
            "distance_to_left_boundary": 0.0,
            "distance_to_right_boundary": 0.0,
            "lane_number": 0,
            "num_lanes": 1,
            "lane_width": 3.5,
            "road_width": 3.5,
            "distance_to_lane_center": 0.0,
            "intersecting_left_boundary": False,
            "intersecting_right_boundary": False,
            "in_single_lane": True,
            "in_multiple_lanes": False,
        }

        if self.road_network is not None:
            comprehensive_info = self.road_network.get_comprehensive_road_info(
                self.state[:2], self.state[2]
            )
            on_road = 1 if comprehensive_info["on_road"] else 0
            road_info.update(comprehensive_info)

        return {
            "position": self.state[:2].copy(),
            "heading": np.array([self.state[2]], dtype=np.float32),
            "velocity": np.array([self.state[3]], dtype=np.float32),
            "goal": self.goal_pos.copy(),
            "goal_distance": np.array([goal_dist], dtype=np.float32),
            "on_road": on_road,
            "distance_to_road_center": np.array(
                [road_info["distance_to_road_center"]], dtype=np.float32
            ),
            "distance_to_left_boundary": np.array(
                [road_info["distance_to_left_boundary"]], dtype=np.float32
            ),
            "distance_to_right_boundary": np.array(
                [road_info["distance_to_right_boundary"]], dtype=np.float32
            ),
            "lane_number": int(road_info["lane_number"]) + 1,
            "num_lanes": int(road_info["num_lanes"]),
            "lane_width": np.array(
                [road_info["lane_width"]], dtype=np.float32
            ),
            "road_width": np.array(
                [road_info["road_width"]], dtype=np.float32
            ),
            "distance_to_lane_center": np.array(
                [road_info["distance_to_lane_center"]], dtype=np.float32
            ),
            "intersecting_left_boundary": int(
                road_info["intersecting_left_boundary"]
            ),
            "intersecting_right_boundary": int(
                road_info["intersecting_right_boundary"]
            ),
            "in_single_lane": int(road_info["in_single_lane"]),
            "in_multiple_lanes": int(road_info["in_multiple_lanes"]),
        }

    def _get_info(self) -> dict[str, typing.Any]:
        on_road = True
        if self.road_network is not None:
            on_road = not self.road_network.is_off_road(self.state[:2])

        return {
            "goal_distance": float(np.linalg.norm(self.state[:2] - self.goal_pos)),
            "collision": self._check_collision(),
            "in_bounds": self._within_world_boundaries(),
            "on_road": on_road,
        }

    def _get_car_corners(self) -> np.ndarray:
        x, y, theta, _ = self.state
        half_length = self.car_length / 2
        half_width = self.car_width / 2
        
        corners_local = np.array([
            # front-right
            [half_length, -half_width],
            # front-left
            [half_length, half_width],
            # rear-left
            [-half_length, half_width],
            # rear-right
            [-half_length, -half_width],
        ])
        
        # Rotation matrix
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, -s], [s, c]])
        
        # Transform to world frame
        pos = np.array([x, y])
        corners_world = np.array([pos + rot @ corner for corner in corners_local])
        
        return corners_world
    
    def _check_collision(self) -> bool:
        corners = self._get_car_corners()
        
        # Check corners against obstacles
        for corner in corners:
            for obs in self.obstacles:
                if obs.check_collision(corner):
                    return True
        
        # Check midpoints of each edge for better collision detection
        for i in range(4):
            midpoint = (corners[i] + corners[(i + 1) % 4]) / 2
            for obs in self.obstacles:
                if obs.check_collision(midpoint):
                    return True
        
        # Check center point
        center = self.state[:2]
        for obs in self.obstacles:
            if obs.check_collision(center):
                return True
        
        if self.solid_road_borders and self.road_network is not None:
            # Check if any car point is off-road
            for corner in corners:
                if self.road_network.is_off_road(corner):
                    return True
            # Check midpoints
            for i in range(4):
                midpoint = (corners[i] + corners[(i + 1) % 4]) / 2
                if self.road_network.is_off_road(midpoint):
                    return True
            # Check center
            if self.road_network.is_off_road(center):
                return True
        
        return False

    def _within_world_boundaries(self) -> bool:
        x, y = self.state[:2]
        min_x, min_y = self.world_origin
        max_x = min_x + self.world_size[0]
        max_y = min_y + self.world_size[1]
        return min_x <= x <= max_x and min_y <= y <= max_y

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            return self._render_frame()
        return None
    
    def _render_frame(self) -> np.ndarray | None:
        return self.renderer.render_frame(
            state=self.state,
            spawn_pos=self.spawn_pos,
            goal_pos=self.goal_pos,
            goal_radius=self.goal_radius,
            obstacles=self.obstacles,
            road_network=self.road_network,
            global_path=self.global_path,
            car_length=self.car_length,
            car_width=self.car_width,
            world_origin=self.world_origin,
            world_size=self.world_size,
            pixels_per_meter=self.pixels_per_meter,
            world_to_screen=self._world_to_screen,
            meters_to_pixels=self._meters_to_pixels,
            get_car_corners=self._get_car_corners,
            sim_time=self.sim_time,
            overlay_manager=self.overlay_manager,
            performance_tracker=self.performance_tracker,
        )
        
    def get_episode_data(self) -> dict[str, typing.Any]:
        """
        Get all tracked data from the current/last episode for metrics computation.
        
        Returns:
            Dictionary containing:
                - 'actions': List of actions taken
                - 'positions': List of vehicle positions (x, y)
                - 'steering_angles': List of steering angles (radians)
                - 'on_road': List of on_road flags (0 or 1)
                - 'velocities': List of velocities (m/s)
                - 'headings': List of heading angles (radians)
                - 'terminated': Whether episode terminated
                - 'truncated': Whether episode was truncated
        """
        return {
            **self._episode_data,
        }

    def close(self) -> None:
        self.renderer.close()
