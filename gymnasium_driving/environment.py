import typing

import gymnasium
import numpy as np

from gymnasium_driving.components.obstacles import Circle, Rectangle
from gymnasium_driving.components.overlay import OverlayManager
from gymnasium_driving.components.performance import PerformanceTracker
from gymnasium_driving.components.recorder import EpisodeRecorder, StepData
from gymnasium_driving.components.renderer import Renderer
from gymnasium_driving.components.roads import RoadNetwork
from gymnasium_driving.helpers import (
    heading_error_to_polyline,
    signed_cte_to_polyline,
)
from gymnasium_driving.models.ackerman import AckermannModel
from gymnasium_driving.models.bicycle import BicycleModel

ObstacleList = typing.List[typing.Union[Circle, Rectangle]]

RoadNetworkFactory = typing.Callable[["CarEnvironment"], RoadNetwork]
ObstaclesFactory = typing.Callable[["CarEnvironment"], ObstacleList]


class CarState(typing.TypedDict):
    x: float
    y: float
    yaw: float
    velocity: float
    cte: float
    heading_error: float
    closest_path_idx: int


class SpawnGoalInfo(typing.TypedDict):
    spawn_pos: np.ndarray
    spawn_heading: float
    goal_pos: np.ndarray
    goal_radius: float
    path: np.ndarray  # (N, 2) array of waypoints from spawn to goal


PositionsFactory = typing.Callable[["CarEnvironment"], SpawnGoalInfo]


class CarEnvironment(gymnasium.Env):
    """
    A bicycle kinematic model environment with proper meter-based scaling.

    All dimensions are in meters, velocities in m/s.

    road_network_factory, obstacles_factory, and positions_factory are all
    callables invoked on each reset() to generate a fresh road network,
    obstacle set, and spawn/goal configuration respectively.

    The obstacles_factory receives the environment after the road network has
    already been assigned, so it can inspect self.road_network freely.

    The positions_factory receives the environment after both the road network
    and obstacles have been assigned.

    All factories receive the environment instance (self) as their sole
    argument. If you need static values, just wrap them in a lambda:

        road_network_factory=lambda env: my_road_network
        obstacles_factory=lambda env: [Circle(...), Rectangle(...)]
        positions_factory=lambda env: SpawnGoalInfo(...)
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    WORLD_PADDING = 10.0

    CAR_LENGTH = 4.5
    CAR_WIDTH = 1.8
    WHEELBASE = 2.5

    MIN_VELOCITY = -5.0
    MAX_VELOCITY = 5.0

    MIN_ACCELERATION = 0
    MAX_ACCELERATION = 3.0

    MAX_LATERAL_ACCELERATION = 4.0

    MAX_BRAKE_DECELERATION = 1.5

    MIN_STEERING, MAX_STEERING = -np.pi / 4, np.pi / 4
    MIN_THROTTLE, MAX_THROTTLE = 0, 1
    MIN_BRAKE, MAX_BRAKE = 0, 1

    SCREEN_SIZE = (800, 800)

    DELTA_TIME = 0.1

    def __init__(
        self,
        road_network_factory: RoadNetworkFactory,
        obstacles_factory: ObstaclesFactory,
        positions_factory: PositionsFactory,
        render_mode: typing.Literal["rgb_array"] | None = None,
        model: typing.Literal["bicycle", "ackerman"] = "bicycle",
    ):
        super().__init__()

        self.road_network_factory = road_network_factory
        self.obstacles_factory = obstacles_factory
        self.positions_factory = positions_factory

        self.spawn_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.spawn_heading = 0.0
        self.goal_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.goal_radius = 3.0

        # NOTE: will be populated on each reset(); None initialization
        self.road_network: RoadNetwork | None = None
        self.obstacles: ObstacleList = []

        if model == "bicycle":
            self.model = BicycleModel(
                wheelbase=CarEnvironment.WHEELBASE,
                max_steer=CarEnvironment.MAX_STEERING,
                delta_time=CarEnvironment.DELTA_TIME,
            )
        elif model == "ackerman":
            self.model = AckermannModel(
                wheelbase=CarEnvironment.WHEELBASE,
                track_width=CarEnvironment.CAR_WIDTH,
                max_steer=CarEnvironment.MAX_STEERING,
                delta_time=CarEnvironment.DELTA_TIME,
            )
        else:
            raise ValueError(f"Unsupported model type: {model}")

        # NOTE: placeholders until reset()
        self.world_size = np.array([100.0, 100.0], dtype=np.float32)
        self.world_origin = np.array([0.0, 0.0], dtype=np.float32)
        self.pixels_per_meter = 1.0

        self.render_mode = render_mode

        self.action_space = gymnasium.spaces.Box(
            low=np.array(
                [CarEnvironment.MIN_STEERING, 0.0, 0.0, 0.0],
                dtype=np.float32,
            ),
            high=np.array(
                [CarEnvironment.MAX_STEERING, 1.0, 1.0, 1.0],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        self.observation_space = gymnasium.spaces.Dict({})

        self.state: CarState = {
            "x": 0.0,
            "y": 0.0,
            "yaw": 0.0,
            "velocity": 0.0,
            "cte": 0.0,
            "heading_error": 0.0,
            "closest_path_idx": 0,
        }
        self.path: np.ndarray = np.array([], dtype=np.float32)

        self.recorder = EpisodeRecorder()
        self.overlay_manager = OverlayManager()
        self.performance_tracker = PerformanceTracker(show_performance=True)
        self.renderer = Renderer(
            screen_size=CarEnvironment.SCREEN_SIZE,
            render_mode=render_mode,
            render_fps=self.metadata["render_fps"],
        )

        self.simulation_time = 0.0

    def _rebuild_from_factories(self) -> None:
        """
        Invoke all factories in order: road network, positions, then obstacles.
        """
        self.road_network = self.road_network_factory(self)

        info = self.positions_factory(self)
        self.spawn_pos = np.array(info["spawn_pos"], dtype=np.float32)
        self.spawn_heading = float(info["spawn_heading"])
        self.goal_pos = np.array(info["goal_pos"], dtype=np.float32)
        self.goal_radius = float(info["goal_radius"])
        self.path = np.array(info["path"], dtype=np.float32)

        self.obstacles = self.obstacles_factory(self)

    def _get_car_corners_for_pose(
        self,
        position: np.ndarray,
        yaw: float,
    ) -> np.ndarray:
        half_length = CarEnvironment.CAR_LENGTH / 2
        half_width = CarEnvironment.CAR_WIDTH / 2

        corners_local = np.array(
            [
                [half_length, -half_width],
                [half_length, half_width],
                [-half_length, half_width],
                [-half_length, -half_width],
            ],
            dtype=np.float32,
        )

        c, s = np.cos(yaw), np.sin(yaw)
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)

        pos = np.array(position, dtype=np.float32)
        corners_world = np.array(
            [pos + rot @ corner for corner in corners_local],
            dtype=np.float32,
        )
        return corners_world

    def _compute_path_tracking_metrics(
        self,
        ego_pos: np.ndarray,
        ego_yaw: float,
    ) -> typing.Tuple[float, float, int]:
        if self.path is None or len(self.path) < 2:
            return 0.0, 0.0, 0

        cte, closest_idx = signed_cte_to_polyline(self.path, ego_pos)
        heading_error = heading_error_to_polyline(
            self.path,
            ego_yaw,
            closest_idx,
        )
        return cte, heading_error, int(closest_idx)

    def _calculate_world_bounds(
        self,
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        min_x, min_y = float("inf"), float("inf")
        max_x, max_y = float("-inf"), float("-inf")

        min_x, min_y = min(min_x, self.spawn_pos[0]), min(min_y, self.spawn_pos[1])
        max_x, max_y = max(max_x, self.spawn_pos[0]), max(max_y, self.spawn_pos[1])

        min_x, min_y = (
            min(min_x, self.goal_pos[0] - self.goal_radius),
            min(min_y, self.goal_pos[1] - self.goal_radius),
        )
        max_x, max_y = (
            max(max_x, self.goal_pos[0] + self.goal_radius),
            max(max_y, self.goal_pos[1] + self.goal_radius),
        )

        for obstacle in self.obstacles:
            bounds = obstacle.get_bounds()
            min_x, min_y = min(min_x, bounds[0]), min(min_y, bounds[1])
            max_x, max_y = max(max_x, bounds[2]), max(max_y, bounds[3])

        if self.road_network is not None:
            bounds = self.road_network.get_bounds()
            min_x, min_y = min(min_x, bounds[0]), min(min_y, bounds[1])
            max_x, max_y = max(max_x, bounds[2]), max(max_y, bounds[3])

        min_x -= CarEnvironment.WORLD_PADDING
        min_y -= CarEnvironment.WORLD_PADDING
        max_x += CarEnvironment.WORLD_PADDING
        max_y += CarEnvironment.WORLD_PADDING

        world_size = np.array([max_x - min_x, max_y - min_y], dtype=np.float32)
        world_origin = np.array([min_x, min_y], dtype=np.float32)

        return world_size, world_origin

    def _world_to_screen(
        self,
        position: typing.Union[np.ndarray, typing.Tuple[float, float]],
    ) -> typing.Tuple[int, int]:
        if isinstance(position, tuple):
            position = np.array(position, dtype=np.float32)

        local_position = position - self.world_origin

        screen_x = int(local_position[0] * self.pixels_per_meter) + 20
        screen_y = int(
            CarEnvironment.SCREEN_SIZE[1]
            - local_position[1] * self.pixels_per_meter
            - 20
        )

        return (screen_x, screen_y)

    def _meters_to_pixels(self, meters: float) -> int:
        return max(1, int(meters * self.pixels_per_meter))

    def reset(
        self, seed: int | None = None, **kwargs
    ) -> typing.Tuple[dict[str, typing.Any], dict[str, typing.Any]]:
        super().reset(seed=seed)

        self.simulation_time = 0.0

        self._rebuild_from_factories()

        for obstacle in self.obstacles:
            obstacle.reset()

        self.refresh_world_bounds()

        self.state = {
            "x": float(self.spawn_pos[0]),
            "y": float(self.spawn_pos[1]),
            "yaw": self.spawn_heading,
            "velocity": 0.0,
            "cte": 0.0,
            "heading_error": 0.0,
            "closest_path_idx": 0,
        }

        cte, heading_error, closest_idx = self._compute_path_tracking_metrics(
            ego_pos=self.spawn_pos,
            ego_yaw=self.spawn_heading,
        )
        self.state["cte"] = cte
        self.state["heading_error"] = heading_error
        self.state["closest_path_idx"] = closest_idx

        self.performance_tracker.reset()
        self.recorder.reset()

        return {}, {}

    def refresh_world_bounds(self) -> None:
        self.world_size, self.world_origin = self._calculate_world_bounds()

        margin_x, margin_y = 40, 40
        scale_x = (CarEnvironment.SCREEN_SIZE[0] - margin_x) / self.world_size[0]
        scale_y = (CarEnvironment.SCREEN_SIZE[1] - margin_y) / self.world_size[1]
        self.pixels_per_meter = min(scale_x, scale_y)

    def step(
        self, action: np.ndarray
    ) -> typing.Tuple[
        dict[str, typing.Any],
        float,
        bool,
        bool,
        dict[str, typing.Any],
    ]:
        assert len(action) == 4, (
            "Action must be of the form [steering, throttle, brake, reverse]"
        )

        ego_pos = np.array([self.state["x"], self.state["y"]], dtype=np.float32)
        ego_velocity = self.state["velocity"]

        self.recorder.record(
            StepData(
                timestamp=self.simulation_time,
                position_x=self.state["x"],
                position_y=self.state["y"],
                velocity=self.state["velocity"],
                heading=self.state["yaw"],
                steering=float(action[0]),
                throttle=float(action[1]),
                brake=float(action[2]),
                reverse=bool(action[3]),
            )
        )

        steering = np.clip(
            action[0], CarEnvironment.MIN_STEERING, CarEnvironment.MAX_STEERING
        )
        throttle = np.clip(
            action[1], CarEnvironment.MIN_THROTTLE, CarEnvironment.MAX_THROTTLE
        )
        brake = np.clip(action[2], CarEnvironment.MIN_BRAKE, CarEnvironment.MAX_BRAKE)
        reverse = action[3] != 0.0

        direction = -1 if reverse else 1
        acceleration = direction * throttle * CarEnvironment.MAX_ACCELERATION
        acceleration += (
            (-brake * self.MAX_BRAKE_DECELERATION * np.sign(ego_velocity))
            if abs(ego_velocity) > 1e-3
            else 0.0
        )

        self.state = self.model.compute_state(
            x=float(self.state["x"]),
            y=float(self.state["y"]),
            yaw=float(self.state["yaw"]),
            steer=float(steering),
            velocity=float(self.state["velocity"]),
            acceleration=float(acceleration),
        )

        self.state["velocity"] = np.clip(
            self.state["velocity"],
            CarEnvironment.MIN_VELOCITY,
            CarEnvironment.MAX_VELOCITY,
        )

        ego_pos = np.array([self.state["x"], self.state["y"]], dtype=np.float32)
        cte, heading_error, closest_idx = self._compute_path_tracking_metrics(
            ego_pos=ego_pos,
            ego_yaw=float(self.state["yaw"]),
        )
        self.state["cte"] = cte
        self.state["heading_error"] = heading_error
        self.state["closest_path_idx"] = closest_idx

        self.simulation_time += CarEnvironment.DELTA_TIME

        for obstacle in self.obstacles:
            obstacle.update(self.simulation_time)

        self.performance_tracker.update()

        terminated = False
        truncated = False
        reward = 0.0

        goal_dist = np.linalg.norm(ego_pos - self.goal_pos)
        if goal_dist <= self.goal_radius:
            terminated = True
        elif self._check_collision():
            terminated = True
        elif not self._within_world_boundaries():
            truncated = True
        elif self.road_network is not None:
            if self.road_network.is_off_road(ego_pos):
                if self.road_network.enforce_road:
                    terminated = True

        return {}, reward, terminated, truncated, {}

    def _get_car_corners(self) -> np.ndarray:
        return self._get_car_corners_for_pose(
            position=np.array([self.state["x"], self.state["y"]], dtype=np.float32),
            yaw=float(self.state["yaw"]),
        )

    def _check_collision(
        self,
        position: np.ndarray | None = None,
        yaw: float | None = None,
        check_road: bool = True,
        respect_solid_road_borders: bool = True,
    ) -> bool:
        if position is None:
            center = np.array([self.state["x"], self.state["y"]], dtype=np.float32)
        else:
            center = np.array(position, dtype=np.float32)

        pose_yaw = float(self.state["yaw"]) if yaw is None else float(yaw)
        corners = self._get_car_corners_for_pose(center, pose_yaw)

        sample_points = [center, *corners]
        for i in range(4):
            sample_points.append((corners[i] + corners[(i + 1) % 4]) / 2)

        for point in sample_points:
            for obs in self.obstacles:
                if obs.check_collision(point):
                    return True

        should_check_road = (
            check_road
            and self.road_network is not None
            and (not respect_solid_road_borders or self.road_network.solid_road_borders)
        )
        if should_check_road:
            for point in sample_points:
                if self.road_network.is_off_road(point):
                    return True

        return False

    def _within_world_boundaries(self) -> bool:
        x, y = self.state["x"], self.state["y"]
        min_x, min_y = self.world_origin
        max_x, max_y = (
            min_x + self.world_size[0],
            min_y + self.world_size[1],
        )
        return min_x <= x <= max_x and min_y <= y <= max_y

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            return self.renderer.render_frame(self)
        return None

    def close(self) -> None:
        self.renderer.close()


def make_centerline_positions_factory(
    proportion: typing.Union[float, typing.Tuple[float, float]] = (0.7, 0.05),
    noise: typing.Tuple[float, float] = (0.15, 0.5),
) -> PositionsFactory:
    """
    Returns a positions factory that replicates the original random spawn/goal
    sampling behaviour.

    proportion: either a fixed float or a (mean, std) tuple controlling how far
                along the centerline the goal is placed relative to the spawn.
    noise:      (heading_std_rad, lateral_offset_std_meters) applied to the
                sampled spawn pose.
    """
    if isinstance(proportion, (int, float)):
        proportion_mean = float(proportion)
        proportion_var = 0.0
        fixed_proportion = True
    else:
        proportion_mean, proportion_var = proportion
        fixed_proportion = False

    heading_noise, cte_noise = noise

    def _build_centerline(env: "CarEnvironment") -> np.ndarray:
        all_points = []
        for road in env.road_network.roads:
            for segment in road.segments:
                num_points = max(50, int(segment.get_length() * 2))
                all_points.extend(segment.get_centerline_points(num_points))
        return np.array(all_points, dtype=np.float32)

    def factory(env: "CarEnvironment") -> SpawnGoalInfo:
        centerline = _build_centerline(env)
        n = len(centerline)

        if fixed_proportion:
            prop = np.clip(proportion_mean, 0.01, 0.99)
        else:
            prop = np.clip(
                env.np_random.normal(proportion_mean, proportion_var),
                0.01,
                0.99,
            )

        goal_span = int(prop * (n - 1))
        max_spawn_idx = (n - 1) - goal_span

        chosen_spawn_idx = 0
        chosen_spawn_on_path = centerline[0].copy()
        chosen_spawn_pos = chosen_spawn_on_path.copy()
        chosen_spawn_heading = 0.0

        # TODO: probably useless as obstacles are generated after spawn points and thus no obstacles are available at this point, empty
        for _ in range(80):
            if fixed_proportion:
                spawn_idx = 0
            else:
                spawn_idx = int(env.np_random.uniform(0, max_spawn_idx + 1))
                spawn_idx = int(np.clip(spawn_idx, 0, n - 2))

            spawn_on_path = centerline[spawn_idx].copy()

            next_idx = min(spawn_idx + 1, n - 1)
            tangent = centerline[next_idx] - centerline[spawn_idx]
            length = float(np.linalg.norm(tangent))
            if length < 1e-9:
                tangent = np.array([1.0, 0.0], dtype=np.float32)
            else:
                tangent = tangent / length

            normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)

            base_heading = float(np.arctan2(tangent[1], tangent[0]))
            candidate_heading = base_heading + float(
                env.np_random.normal(0.0, heading_noise)
            )
            lateral_offset = float(env.np_random.normal(0.0, cte_noise))
            candidate_pos = (spawn_on_path + lateral_offset * normal).astype(np.float32)

            if not env._check_collision(
                position=candidate_pos,
                yaw=candidate_heading,
                check_road=True,
                respect_solid_road_borders=False,
            ):
                chosen_spawn_idx = spawn_idx
                chosen_spawn_on_path = spawn_on_path
                chosen_spawn_pos = candidate_pos
                chosen_spawn_heading = candidate_heading
                break

        goal_idx = min(chosen_spawn_idx + goal_span, n - 1)
        goal_pos = centerline[goal_idx].copy()

        path = np.array(
            [chosen_spawn_on_path]
            + [centerline[i] for i in range(chosen_spawn_idx + 1, goal_idx + 1)],
            dtype=np.float32,
        )

        return SpawnGoalInfo(
            spawn_pos=chosen_spawn_pos,
            spawn_heading=chosen_spawn_heading,
            goal_pos=goal_pos,
            path=path,
            goal_radius=3.0,
        )

    return factory
