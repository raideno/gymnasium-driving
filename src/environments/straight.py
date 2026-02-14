import typing

import gymnasium as gym
import numpy as np

import gymnasium_driving
import gymnasium_driving.wrappers

from gymnasium_driving.components.obstacles import Circle

class RandomStraightObstacles(gym.Wrapper):
    """
    Places a random number of circular obstacles along a straight road,
    ensuring no two obstacles share the same horizontal (x) position
    and that each obstacle leaves a passable gap on at least one side.
    """

    def __init__(
        self,
        env,
        min_obstacles: int = 2,
        max_obstacles: int = 5,
        min_radius: float = 0.6,
        max_radius: float = 1.2,
        lateral_range: float = 2.5,
        min_x_spacing: float = 8.0,
        exclude_start: float = 8.0,
        exclude_end: float = 8.0,
        min_passage_width: float = 2.5,
        seed: int | None = None,
    ):
        super().__init__(env)

        self.min_obstacles = min_obstacles
        self.max_obstacles = max_obstacles
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.lateral_range = lateral_range
        self.min_x_spacing = min_x_spacing
        self.exclude_start = exclude_start
        self.exclude_end = exclude_end
        self.min_passage_width = min_passage_width
        self.rng = np.random.default_rng(seed)

    def reset(self, **kwargs):
        seed = kwargs.get("seed")
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        obs = super().reset(**kwargs)

        path = self.unwrapped.path
        if path is None or len(path) < 2:
            self.unwrapped.obstacles = []
            return obs

        road_network = getattr(self.unwrapped, "road_network", None)
        half_width = 4.0
        if road_network is not None and len(road_network.roads) > 0:
            half_width = float(road_network.roads[0].half_width)

        car_width = float(getattr(self.unwrapped, "CAR_WIDTH", 1.8))
        required_passage = max(self.min_passage_width, car_width + 0.3)

        # Determine valid x-range for obstacles (along the straight road)
        x_min = float(self.unwrapped.spawn_pos[0]) + self.exclude_start
        x_max = float(self.unwrapped.goal_pos[0]) - self.exclude_end

        if x_max <= x_min:
            self.unwrapped.obstacles = []
            return obs

        num_obstacles = int(self.rng.integers(self.min_obstacles, self.max_obstacles + 1))

        # Generate unique x positions with minimum spacing
        x_positions = []
        for _ in range(num_obstacles * 20):
            if len(x_positions) >= num_obstacles:
                break
            x = float(self.rng.uniform(x_min, x_max))
            if all(abs(x - ex) >= self.min_x_spacing for ex in x_positions):
                x_positions.append(x)

        # Road center y (straight road, constant y)
        center_y = float(self.unwrapped.spawn_pos[1])

        obstacles = []
        for x in x_positions:
            radius = float(self.rng.uniform(self.min_radius, self.max_radius))

            # Sample lateral offset, ensuring passable gap
            max_lateral = min(self.lateral_range, half_width - radius - 0.2)
            if max_lateral < 0:
                continue

            for _ in range(30):
                lateral = float(self.rng.uniform(-max_lateral, max_lateral))
                # Check that at least one side has enough passage
                left_free = (half_width + lateral) - radius
                right_free = (half_width - lateral) - radius
                if max(left_free, right_free) >= required_passage:
                    break
            else:
                continue

            obstacles.append(Circle(center=(x, center_y + lateral), radius=radius))

        self.unwrapped.obstacles = obstacles
        return obs


def make_environment(
    discrete: bool,
    render_mode: typing.Literal["rgb_array", "human"] | None = None,
    target_velocity: float = 2.5,
    n_steering: int = 15,
    min_obstacles: int = 2,
    max_obstacles: int = 5,
    min_radius: float = 0.6,
    max_radius: float = 1.2,
    lateral_range: float = 2.5,
    min_x_spacing: float = 8.0,
    min_passage_width: float = 2.5,
    max_steps: int = 600,
    **kwargs,
):
    """
    Straight road environment with random obstacles for obstacle-avoidance training.
    """
    env = gymnasium_driving.CarEnvironment(
        model="bicycle",
        road_network=gymnasium_driving.components.roads.RoadNetwork(
            roads=[gymnasium_driving.components.roads.Road(
                segments=[
                    gymnasium_driving.components.roads.StraightSegment(
                        start=(10.0, 50.0),
                        heading=0.0,
                        length=100,
                    )
                ],
                width=8.0,
            )],
        ),
        render_mode=render_mode,
        proportion=(0.90, 0.00),
        noise=(0.0, 0.0),
        obstacles=[],
    )

    if discrete:
        env = gymnasium_driving.wrappers.actions.DiscreteSteeringOnlyActionWrapper(
            env,
            target_velocity=target_velocity,
            n_steering=n_steering,
        )
    else:
        env = gymnasium_driving.wrappers.actions.SteeringOnlyActionWrapper(
            env,
            target_velocity=target_velocity,
        )

    env = RandomStraightObstacles(
        env,
        min_obstacles=min_obstacles,
        max_obstacles=max_obstacles,
        min_radius=min_radius,
        max_radius=max_radius,
        lateral_range=lateral_range,
        min_x_spacing=min_x_spacing,
        min_passage_width=min_passage_width,
    )

    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)

    return env
