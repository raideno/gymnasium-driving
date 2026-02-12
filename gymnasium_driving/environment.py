import typing
import gymnasium

import numpy as np

from gymnasium_driving.components.obstacles import Circle, Rectangle
from gymnasium_driving.components.roads import RoadNetwork

from gymnasium_driving.components.renderer import Renderer
from gymnasium_driving.components.overlay import OverlayManager
from gymnasium_driving.components.performance import PerformanceTracker
from gymnasium_driving.components.recorder import EpisodeRecorder, StepData

from gymnasium_driving.models.bicycle import BicycleModel
from gymnasium_driving.models.ackerman import AckermannModel


class CarEnvironment(gymnasium.Env):
    """
    A bicycle kinematic model environment with proper meter-based scaling.

    All dimensions are in meters, velocities in m/s.
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

    # NOTE: it prevents the car from sliding out the turn, when lateral acceleration exceeds the friction limit, the tires lose grip and the vehicle slide.
    MAX_LATERAL_ACCELERATION = 4.0  # 4.0

    # when the brake pedal is fully pressed, the car decelerate at 6m/s^2
    MAX_BRAKE_DECELERATION = 1.5  # 6.0

    MIN_STEERING, MAX_STEERING = -np.pi / 4, np.pi / 4
    MIN_THROTTLE, MAX_THROTTLE = 0, 1
    MIN_BRAKE, MAX_BRAKE = 0, 1

    SCREEN_SIZE = (800, 800)

    # time passing when calling .step
    # 0.1second = 100ms per step
    DELTA_TIME = 0.1

    def __init__(
        self,
        render_mode: typing.Literal["rgb_array"] | None = None,
        model: typing.Literal["bicycle", "ackerman"] = "bicycle",
        # (position (x, y), heading in radians)
        spawn: typing.Tuple[typing.Tuple[float, float], float] = (
            (10.0, 10.0),
            0.0,
        ),
        # (position (x, y), radius in meters)
        goal: typing.Tuple[typing.Tuple[float, float], float] = (
            (90.0, 90.0),
            3.0,
        ),
        obstacles: typing.List[typing.Union[Circle, Rectangle]] | None = None,
        road_network: RoadNetwork | None = None,
    ):
        super().__init__()

        self.spawn_pos, self.spawn_heading = (
            np.array(spawn[0], dtype=np.float32),
            spawn[1],
        )

        self.goal_pos, self.goal_radius = goal

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

        self.obstacles = obstacles if obstacles else []
        self.road_network = road_network

        self.world_size, self.world_origin = self._calculate_world_bounds()
        self.render_mode = render_mode

        margin_x, margin_y = 40, 40
        scale_x = (CarEnvironment.SCREEN_SIZE[0] - margin_x) / self.world_size[0]
        scale_y = (CarEnvironment.SCREEN_SIZE[1] - margin_y) / self.world_size[1]
        self.pixels_per_meter = min(scale_x, scale_y)

        # [steering, throttle, brake, reverse]
        self.action_space = gymnasium.spaces.Box(
            low=np.array(
                [CarEnvironment.MIN_STEERING, 0.0, 0.0, 0.0], dtype=np.float32
            ),
            high=np.array(
                [CarEnvironment.MAX_STEERING, 1.0, 1.0, 1.0], dtype=np.float32
            ),
            dtype=np.float32,
        )

        self.observation_space = gymnasium.spaces.Dict({})

        self.state = None

        self._compute_global_path()  # Will store waypoints as (N, 2) array

        self.recorder = EpisodeRecorder()
        self.overlay_manager = OverlayManager()
        self.performance_tracker = PerformanceTracker(show_performance=True)
        self.renderer = Renderer(
            screen_size=CarEnvironment.SCREEN_SIZE,
            render_mode=render_mode,
            render_fps=self.metadata["render_fps"],
        )

        self.simulation_time = 0.0

    def _calculate_world_bounds(
        self,
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
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
        min_x, min_y = min(min_x, self.spawn_pos[0]), min(
            min_y, self.spawn_pos[1]
        )
        max_x, max_y = max(max_x, self.spawn_pos[0]), max(
            max_y, self.spawn_pos[1]
        )

        # goal position
        min_x, min_y = min(min_x, self.goal_pos[0] - self.goal_radius), min(
            min_y, self.goal_pos[1] - self.goal_radius
        )
        max_x, max_y = max(max_x, self.goal_pos[0] + self.goal_radius), max(
            max_y, self.goal_pos[1] + self.goal_radius
        )

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
        min_x, min_y = (
            min_x - CarEnvironment.WORLD_PADDING,
            min_y - CarEnvironment.WORLD_PADDING,
        )
        max_x, max_y = (
            max_x + CarEnvironment.WORLD_PADDING,
            max_y + CarEnvironment.WORLD_PADDING,
        )

        world_size = np.array([max_x - min_x, max_y - min_y], dtype=np.float32)
        world_origin = np.array([min_x, min_y], dtype=np.float32)

        return world_size, world_origin

    def _world_to_screen(
        self, position: typing.Union[np.ndarray, typing.Tuple[float, float]]
    ) -> typing.Tuple[int, int]:
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
            CarEnvironment.SCREEN_SIZE[1]
            - local_position[1] * self.pixels_per_meter
            - 20
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
        spawn_idx = np.argmin(
            np.linalg.norm(centerline - self.spawn_pos, axis=1)
        )
        self.path = np.vstack(
            [centerline[spawn_idx:], centerline[:spawn_idx], [self.spawn_pos]]
        )

        return

    def reset(
        self, seed: int | None = None, **kwargs
    ) -> typing.Tuple[dict[str, typing.Any], dict[str, typing.Any]]:
        super().reset(seed=seed)

        # Recompute path each episode to reflect any randomized road/spawn changes
        self._compute_global_path()

        self.state = {
            "x": self.spawn_pos[0],
            "y": self.spawn_pos[1],
            "yaw": self.spawn_heading,
            "velocity": 0.0,
        }

        self.simulation_time = 0.0
        for obstacle in self.obstacles:
            obstacle.reset()

        self.performance_tracker.reset()

        self.recorder.reset()

        observation = {}
        info = {}

        return observation, info

    def refresh_world_bounds(self) -> None:
        """
        Recompute world bounds and rendering scale after modifying road, spawn, or goal.
        """
        self.world_size, self.world_origin = self._calculate_world_bounds()

        margin_x, margin_y = 40, 40
        scale_x = (CarEnvironment.SCREEN_SIZE[0] - margin_x) / self.world_size[0]
        scale_y = (CarEnvironment.SCREEN_SIZE[1] - margin_y) / self.world_size[1]
        self.pixels_per_meter = min(scale_x, scale_y)

    def step(
        self, action: np.ndarray
    ) -> typing.Tuple[
        dict[str, typing.Any], float, bool, bool, dict[str, typing.Any]
    ]:
        assert len(action) == 4, "Action must be of the form [steering, throttle, brake, reverse]"

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
        brake = np.clip(
            action[2], CarEnvironment.MIN_BRAKE, CarEnvironment.MAX_BRAKE
        )
        reverse = action[3] != 0.0

        direction = -1 if reverse else 1
        acceleration = direction * throttle * CarEnvironment.MAX_ACCELERATION
        # NOTE: braking
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

        # NOTE: enforce velocity limits
        self.state["velocity"] = np.clip(
            self.state["velocity"],
            CarEnvironment.MIN_VELOCITY,
            CarEnvironment.MAX_VELOCITY,
        )

        self.simulation_time += CarEnvironment.DELTA_TIME

        for obstacle in self.obstacles:
            obstacle.update(self.simulation_time)

        self.performance_tracker.update()

        observation = {}
        info = {}

        terminated = False
        truncated = False
        reward = 0.0

        ego_pos = np.array([self.state["x"], self.state["y"]], dtype=np.float32)
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

        return observation, reward, terminated, truncated, info

    def _get_car_corners(self) -> np.ndarray:
        x, y, theta, _ = (
            self.state["x"],
            self.state["y"],
            self.state["yaw"],
            self.state["velocity"],
        )
        half_length = CarEnvironment.CAR_LENGTH / 2
        half_width = CarEnvironment.CAR_WIDTH / 2

        corners_local = np.array(
            [
                # front-right
                [half_length, -half_width],
                # front-left
                [half_length, half_width],
                # rear-left
                [-half_length, half_width],
                # rear-right
                [-half_length, -half_width],
            ]
        )

        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, -s], [s, c]])

        position = np.array([x, y])
        corners_world = np.array(
            [position + rot @ corner for corner in corners_local]
        )

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
        center = np.array(
            [self.state["x"], self.state["y"]], dtype=np.float32
        )
        for obs in self.obstacles:
            if obs.check_collision(center):
                return True

        if (
            self.road_network is not None
            and self.road_network.solid_road_borders
        ):
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
        x, y = self.state["x"], self.state["y"]
        min_x, min_y = self.world_origin
        max_x, max_y = min_x + self.world_size[0], min_y + self.world_size[1]
        return min_x <= x <= max_x and min_y <= y <= max_y

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            return self.renderer.render_frame(self)
        return None

    def close(self) -> None:
        self.renderer.close()