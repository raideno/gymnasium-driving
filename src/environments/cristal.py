import typing

import gymnasium as gym

import gymnasium_driving
import gymnasium_driving.wrappers

from src.environments.helpers.random import (
    RandomPathObstacles,
    RandomSpawn,
    RandomRoadNetwork,
    RandomGoal,
)

def make_environment(
    discrete: bool,
    random_obstacles: bool = True,
    number_of_obstacles: int = 4,
    random_spawn: bool = True,
    random_goal: bool = True,
    render_mode: typing.Literal["rgb_array", "human"] | None = None,
    dimensions: typing.Tuple[float, float] = (80.0, 40.0),
    center: typing.Tuple[float, float] = (50.0, 50.0),
    wrappers: typing.List[
        typing.Literal[
            "WithRoadInfo", "WithBaseInfo", "WithPathInfo", "WithObstaclesInfo"
        ]
    ] = [
        "WithRoadInfo",
        "WithBaseInfo",
        "WithPathInfo",
        "WithObstaclesInfo",
    ],
    base_with_position: bool = False,
    road_length_range: typing.Tuple[float, float] = (55.0, 90.0),
    road_height_range: typing.Tuple[float, float] = (35.0, 70.0),
    road_turn_radius_range: typing.Tuple[float, float] = (6.0, 14.0),
    road_width_range: typing.Tuple[float, float] = (6.0, 10.0),
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
        spawn=((50.0, 30.0), 0.0),
        goal=((10.0, 50.0), 2.0),
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

    for wrapper_name in wrappers:
        if wrapper_name == "WithRoadInfo":
            env = gymnasium_driving.wrappers.observations.WithRoadInfo(env)
        elif wrapper_name == "WithBaseInfo":
            env = gymnasium_driving.wrappers.observations.WithBaseInfo(
                env, with_position=base_with_position
            )
        elif wrapper_name == "WithPathInfo":
            env = gymnasium_driving.wrappers.observations.WithPathInfo(env)
        elif wrapper_name == "WithObstaclesInfo":
            env = gymnasium_driving.wrappers.observations.WithObstaclesInfo(env)
        else:
            raise ValueError(f"Unknown wrapper name: {wrapper_name}")

    if random_obstacles:
        env = RandomPathObstacles(
            env,
            num_obstacles=number_of_obstacles,
            lateral_offset=3.0,
            min_radius=1,
            max_radius=2.0,
            exclude_start_distance=1.0,
            exclude_goal_distance=1.0,
        )

    if random_spawn:
        env = RandomSpawn(
            env,
            path_fraction=0.6,
            lateral_noise=0.4,
            heading_noise=0.2,
        )

    env = RandomRoadNetwork(
        env,
        center=center,
        length_range=road_length_range,
        height_range=road_height_range,
        turn_radius_range=road_turn_radius_range,
        width_range=road_width_range,
    )

    if random_goal:
        env = RandomGoal(
            env,
            min_progress=0.35,
            max_progress=0.9,
            goal_radius=2.5,
        )

    # 1000 steps x 0.1 s = 100 s
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)

    return env