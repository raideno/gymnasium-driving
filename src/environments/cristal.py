import typing

import gymnasium as gym

import gymnasium_driving
import gymnasium_driving.wrappers

from src.environments.random import RandomPathObstacles, RandomSpawn

def make_environment(
    discrete: bool,
    random_obstacles: bool = False,
    number_of_obstacles: int = 3,
    random_spawn: bool = True,
    render_mode: typing.Literal["rgb_array", "human"] | None = None,
    dimensions: typing.Tuple[int, int] = (80.0, 40.0),
    wrappers: typing.List[typing.Literal["WithRoadInfo", "WithBaseInfo", "WithPathInfo", "WithObstaclesInfo"]] = [
        "WithRoadInfo",
        "WithBaseInfo",
        "WithPathInfo",
        "WithObstaclesInfo",
    ],
):
    """
    Create the bicycle environment with the correct wrapper stack.

    Wrapper order (inside -> outside):
        BicycleCarEnv
            - PathProgressReward      (reward shaping)
            - Action wrapper          (discrete or continuous)
            - WithBaseInfo            (heading, velocity — NO absolute pos)
            - WithPathInfo            (waypoints + CTE in ego frame)
            - WithObstaclesInfo       (obstacles in ego frame)
            - RandomSpawn             (spawn randomization)
            - RandomPathObstacles     (obstacle randomization)
            - TimeLimit
    """
    env = gymnasium_driving.CarEnvironment(
        model="bicycle",
        road_network=gymnasium_driving.components.roads.RoadNetwork(
            roads=[
                gymnasium_driving.components.roads.create_rectangular_track(
                    center=(50.0, 50.0),
                    length=dimensions[0],
                    height=dimensions[1],
                    turn_radius=8.0,
                    width=8.0,
                )
            ]
        ),
        render_mode=render_mode,
        spawn=((50.0, 30.0), 0.0),
        goal=((10.0, 50.0), 2.0),
        obstacles=[
            gymnasium_driving.components.obstacles.Circle(
                center=(90, 50), radius=1.0
            ),
        ],
    )

    if discrete:
        env = gymnasium_driving.wrappers.actions.DiscreteActionWrapper(env)
    else:
        env = gymnasium_driving.wrappers.actions.ContinuousActionWrapper(env)

    for wrapper_name in wrappers:
        if wrapper_name == "WithRoadInfo":
            env = gymnasium_driving.wrappers.observations.WithRoadInfo(env)
        elif wrapper_name == "WithBaseInfo":
            env = gymnasium_driving.wrappers.observations.WithBaseInfo(env)
        elif wrapper_name == "WithPathInfo":
            env = gymnasium_driving.wrappers.observations.WithPathInfo(env)
        elif wrapper_name == "WithObstaclesInfo":
            env = gymnasium_driving.wrappers.observations.WithObstaclesInfo(env)
        else:
            raise ValueError(f"Unknown wrapper name: {wrapper_name}")

    if random_spawn:
        env = RandomSpawn(
            env,
            path_fraction=0.5,
            lateral_noise=0.1,
            heading_noise=0.15,
        )

    if random_obstacles:
        env = RandomPathObstacles(
            env,
            num_obstacles=number_of_obstacles
        )

    # 1000 steps x 0.1 s = 100 s
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)

    return env