import typing
import gymnasium

import numpy as np

from .components.obstacles import *

from .components.roads import *

from .components.renderer import Renderer
from .components.overlay import OverlayManager
from .components.performance import PerformanceTracker

class BicycleCarEnv(gymnasium.Env):
    """
    A bicycle kinematic model environment with proper meter-based scaling.

    All dimensions are in meters, velocities in m/s.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    # in meters
    CAR_LENGTH = 4.5
    CAR_WIDTH = 1.8
    WORLD_PADDING = 10.0
    
    # meters, m/s, m/s^2
    WHEELBASE = 2.5
    MAX_STEERING = np.pi / 4
    MAX_VELOCITY = 15.0  # ~54 km/h
    MAX_ACCELERATION = 3.0
    MAX_BRAKE_DECELERATION = 6.0  # ~0.6g braking
    
    SCREEN_SIZE = (800, 800)
    
    def __init__(
        self,
        render_mode: typing.Literal["human", "rgb_array"] | None = None,
        # (position (x, y), heading in radians)
        spawn: typing.Tuple[typing.Tuple[float, float], float] = ((10.0, 10.0), 0.0),
        # (position (x, y), radius in meters)
        goal: typing.Tuple[typing.Tuple[float, float], float] = ((90.0, 90.0), 3.0),
        obstacles: typing.List[typing.Union[Circle, Rectangle]] | None = None,
        road_network: RoadNetwork | None = None,
        # TODO: enforce road should be encoded in the road network itself not here
        enforce_road: bool = False,
        solid_road_borders: bool = False,
        
        # Simulation
        dt: float = 0.1,
    ):
        super().__init__()

        self.spawn_pos, self.spawn_heading = np.array(spawn[0], dtype=np.float32), spawn[1]

        self.goal_pos, self.goal_radius = goal
        
        self.dt = dt

        self.obstacles = obstacles if obstacles else []
        self.road_network = road_network
        self.enforce_road = enforce_road
        self.solid_road_borders = solid_road_borders

        self.world_size, self.world_origin = self._calculate_world_bounds()
        self.world_size = np.array((100, 100))
        self.render_mode = render_mode

        margin_x, margin_y = 40, 40
        scale_x = (BicycleCarEnv.SCREEN_SIZE[0] - margin_x) / self.world_size[0]
        scale_y = (BicycleCarEnv.SCREEN_SIZE[1] - margin_y) / self.world_size[1]
        self.pixels_per_meter = min(scale_x, scale_y)

        # [steering, throttle, brake, reverse]
        self.action_space = gymnasium.spaces.Box(
            low=np.array([-BicycleCarEnv.MAX_STEERING, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([BicycleCarEnv.MAX_STEERING, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.observation_space = gymnasium.spaces.Dict({})

        self.state = None
        
        self.path = self._compute_global_path()  # Will store waypoints as (N, 2) array

        self._episode_data = {
            'actions': [],
            'positions': [],
            'steering_angles': [],
            'velocities': [],
            'headings': [],
            'terminated': False,
            'truncated': False
        }

        self.overlay_manager = OverlayManager()
        self.performance_tracker = PerformanceTracker(show_performance=True)
        self.renderer = Renderer(
            screen_size=BicycleCarEnv.SCREEN_SIZE,
            render_mode=render_mode,
            render_fps=self.metadata["render_fps"],
        )
        
        self.simulation_time = 0.0

    def _calculate_world_bounds(self) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        Calculate world bounds based on all content.

        Returns:
            world_size:
                (width, height) in meters
            world_origin:
                (min_x, min_y)
                Bottom left corner of the rectangular area of the world that will be displayed on screen.
                Everything outside that rectangle gets clipped (not rendered).
        """
        min_x, min_y = float("inf"), float("inf")
        max_x, max_y = float("-inf"), float("-inf")
        
        # spawn position
        min_x, min_y = min(min_x, self.spawn_pos[0]), min(min_y, self.spawn_pos[1])
        max_x, max_y = max(max_x, self.spawn_pos[0]), max(max_y, self.spawn_pos[1])

        # goal position
        min_x, min_y = min(min_x, self.goal_pos[0] - self.goal_radius), min(min_y, self.goal_pos[1] - self.goal_radius)
        max_x, max_y = max(max_x, self.goal_pos[0] + self.goal_radius), max(max_y, self.goal_pos[1] + self.goal_radius)

        # obstacles
        for obstacle in self.obstacles:
            bounds = obstacle.get_bounds()
            
            min_x, min_y = min(min_x, bounds[0]), min(min_y, bounds[1])
            max_x, max_y = max(max_x, bounds[2]), max(max_y, bounds[3])

        # roads
        if self.road_network is not None:
            bounds = self.road_network.get_bounds()
            min_x, min_y = min(min_x, bounds[0]), min(min_y, bounds[1])
            max_x, max_y = max(max_x, bounds[2]), max(max_y, bounds[3])

        # padding
        min_x, min_y = min_x - BicycleCarEnv.WORLD_PADDING, min_y - BicycleCarEnv.WORLD_PADDING
        max_x, max_y = max_x + BicycleCarEnv.WORLD_PADDING, max_y + BicycleCarEnv.WORLD_PADDING

        world_size = np.array([max_x - min_x, max_y - min_y], dtype=np.float32)
        world_origin = np.array([min_x, min_y], dtype=np.float32)

        return world_size, world_origin

    def _world_to_screen(self, position: typing.Union[np.ndarray, typing.Tuple[float, float]]) -> typing.Tuple[int, int]:
        """
        Convert world coordinates (meters) to screen coordinates (pixels).
        Position P (100, 100) in the world isn't (100, 100) in the screen.
        The screen maps an area of the world determined by world_origin. If it is (90, 90).
        To get position of P in screen, we do `local_position = (100, 100) - (90, 90) = (10, 10).
        """
        if isinstance(position, tuple):
            position = np.array(position, dtype=np.float32)

        local_position = position - self.world_origin

        screen_x = int(local_position[0] * self.pixels_per_meter) + 20
        screen_y = int(
            BicycleCarEnv.SCREEN_SIZE[1] - local_position[1] * self.pixels_per_meter - 20
        )

        return (screen_x, screen_y)

    def _meters_to_pixels(self, meters: float) -> int:
        return max(1, int(meters * self.pixels_per_meter))

    def _compute_global_path(self) -> None:
        if self.road_network is None:
            self.path = np.array(
                [self.spawn_pos, self.goal_pos], dtype=np.float32
            )
            return
        
        all_points = []
        for road in self.road_network.roads:
            for segment in road.segments:
                num_points = max(50, int(segment.get_length() * 2))
                all_points.extend(segment.get_centerline_points(num_points))
        
        if not all_points:
            self.path = np.array(
                [self.spawn_pos, self.goal_pos], dtype=np.float32
            )
            return
        
        centerline = np.array(all_points, dtype=np.float32)
        
        # NOTE: find closest point to spawn and reorder path to start there
        spawn_idx = np.argmin(np.linalg.norm(centerline - self.spawn_pos, axis=1))
        self.path = np.vstack([
            centerline[spawn_idx:],
            centerline[:spawn_idx],
            [self.spawn_pos]
        ])
        
        return self.path

    def reset(self, seed: int | None = None, **kwargs) -> typing.Tuple[dict[str, typing.Any], dict[str, typing.Any]]:
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
        
        self.simulation_time = 0.0
        for obstacle in self.obstacles:
            obstacle.reset()
        
        self.performance_tracker.reset()

        self._episode_data['actions'] = []
        self._episode_data['positions'] = []
        self._episode_data['steering_angles'] = []
        self._episode_data['velocities'] = []
        self._episode_data['headings'] = []
        self._episode_data['terminated'] = False
        self._episode_data['truncated'] = False

        observation = self._get_observation()
        info = None

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(
        self,
        action: np.ndarray
    ) -> typing.Tuple[
        dict[str, typing.Any], float, bool, bool, dict[str, typing.Any]
    ]:
        assert len(action) == 4, "Action must be of the form [steering, throttle, brake, reverse]"
        
        self._episode_data['positions'].append(self.state[:2].copy())
        self._episode_data['velocities'].append(float(self.state[3]))
        self._episode_data['headings'].append(float(self.state[2]))
        self._episode_data['actions'].append(action.copy())
        self._episode_data['steering_angles'].append(float(action[0]))
        
        steering = np.clip(action[0], -BicycleCarEnv.MAX_STEERING, BicycleCarEnv.MAX_STEERING)
        throttle = np.clip(action[1], 0.0, 1.0)
        brake = np.clip(action[2], 0.0, 1.0)
        reverse = (action[3] != 0.0)
        
        direction = -1 if reverse else 1
        acceleration = direction * throttle * BicycleCarEnv.MAX_ACCELERATION

        x, y, theta, v = self.state

        x_new = x + v * np.cos(theta) * self.dt
        y_new = y + v * np.sin(theta) * self.dt
        theta_new = theta + (v / BicycleCarEnv.WHEELBASE) * np.tan(steering) * self.dt
        v_new = v + acceleration * self.dt
        
        if brake > 0.0 and abs(v_new) > 0.01:
            brake_decel = brake * BicycleCarEnv.MAX_BRAKE_DECELERATION * self.dt
            if v_new > 0:
                v_new = max(0.0, v_new - brake_decel)
            else:
                v_new = min(0.0, v_new + brake_decel)
        
        v_new = np.clip(v_new, -BicycleCarEnv.MAX_VELOCITY, BicycleCarEnv.MAX_VELOCITY)

        theta_new = np.arctan2(np.sin(theta_new), np.cos(theta_new))

        self.state = np.array([x_new, y_new, theta_new, v_new], dtype=np.float32)
        
        self.simulation_time += self.dt
        
        for obstacle in self.obstacles:
            obstacle.update(self.simulation_time)
        
        self.performance_tracker.update()

        observation = self._get_observation()
        info = None
        
        terminated = False
        truncated = False
        reward = 0.0
        
        goal_dist = np.linalg.norm(self.state[:2] - self.goal_pos)
        if goal_dist <= self.goal_radius:
            terminated = True
        elif self._check_collision():
            terminated = True
        elif not self._within_world_boundaries():
            truncated = True
        elif self.road_network is not None:
            if self.road_network.is_off_road(self.state[:2]):
                if self.enforce_road:
                    terminated = True
        
        self._episode_data['terminated'] = terminated
        self._episode_data['truncated'] = truncated

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> dict[str, typing.Any]:
        return {}
        
    def _get_car_corners(self) -> np.ndarray:
        x, y, theta, _ = self.state
        half_length = BicycleCarEnv.CAR_LENGTH / 2
        half_width = BicycleCarEnv.CAR_WIDTH / 2
        
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
        
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, -s], [s, c]])
        
        position = np.array([x, y])
        corners_world = np.array([position + rot @ corner for corner in corners_local])
        
        return corners_world
    
    def _check_collision(self) -> bool:
        corners = self._get_car_corners()
        
        # corners vs obstacles
        for corner in corners:
            for obs in self.obstacles:
                if obs.check_collision(corner):
                    return True
        
        # edges' midpoints vs obstacles
        for i in range(4):
            midpoint = (corners[i] + corners[(i + 1) % 4]) / 2
            for obs in self.obstacles:
                if obs.check_collision(midpoint):
                    return True
        
        # center point vs obstacles
        center = self.state[:2]
        for obs in self.obstacles:
            if obs.check_collision(center):
                return True
        
        if self.solid_road_borders and self.road_network is not None:
            # corners vs off-road
            for corner in corners:
                if self.road_network.is_off_road(corner):
                    return True
            # edges' midpoints vs off-road
            for i in range(4):
                midpoint = (corners[i] + corners[(i + 1) % 4]) / 2
                if self.road_network.is_off_road(midpoint):
                    return True
            # center point vs off-road
            if self.road_network.is_off_road(center):
                return True
        
        return False

    def _within_world_boundaries(self) -> bool:
        """
        Checks whether the vehicle is withing the rectangular area defined by `world_origin`
        and `world_size`. This area includes everything that could be rendered (spawn, goal, obstacles, etc).
        """
        x, y = self.state[:2]
        min_x, min_y = self.world_origin
        max_x, max_y = min_x + self.world_size[0], min_y + self.world_size[1]
        return min_x <= x <= max_x and min_y <= y <= max_y

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            return self._render_frame()
        return None
    
    def _render_frame(self) -> np.ndarray | None:
	    return self.renderer.render_frame(self)
        
    def get_episode_data(self) -> dict[str, typing.Any]:
        return {
            **self._episode_data,
        }

    def close(self) -> None:
        self.renderer.close()
