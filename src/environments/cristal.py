import typing

import gymnasium as gym

import gymnasium_driving
import gymnasium_driving.wrappers

from src.environments.helpers import (
    RandomPathObstacles,
)

def make_environment(
    discrete: bool,
    number_of_obstacles: int,
    render_mode: typing.Literal["rgb_array", "human"] | None = None,
    dimensions: typing.Tuple[float, float] = (80.0, 40.0),
    center: typing.Tuple[float, float] = (50.0, 50.0),
    max_steps: int = 500,
    **kwargs
):
    """
    Randomized road environment with domain randomization of paths and obstacles.
    """
    env = gymnasium_driving.CarEnvironment(
        model="bicycle",
        road_network=gymnasium_driving.components.roads.RoadNetwork(
            roads=[
                gymnasium_driving.components.roads.create_rectangular_track(
                    center=center,
                    length=dimensions[0],
                    height=dimensions[1],
                    turn_radius=8.0,
                    width=8.0,
                )
            ],
        ),
        render_mode=render_mode,
        proportion=(0.90, 0.00),
        noise=(0.0, 0.0),
        obstacles=[],
    )

    if discrete:
        env = gymnasium_driving.wrappers.actions.DiscreteSteeringOnlyActionWrapper(
            env,
            target_velocity=5.0,
            n_steering=10
        )
    else:
        env = gymnasium_driving.wrappers.actions.SteeringOnlyActionWrapper(
            env,
            target_velocity=5.0,
        )

    if number_of_obstacles > 0:
        env = RandomPathObstacles(
            env,
            num_obstacles=number_of_obstacles,
            lateral_offset=3.0,
            min_radius=1,
            max_radius=2.0,
            exclude_start_distance=1.0,
            exclude_goal_distance=1.0,
        )

    # 1000 steps x 0.1 s = 100 s
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)

    return env