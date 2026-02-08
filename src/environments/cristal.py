import typing

import gymnasium as gym

import gymnasium_driving
import gymnasium_driving.wrappers

from src.environments.random import RandomPathObstacles
    
def make_environment(
    discrete: bool,
    random_obstacles: bool = False,
    render_mode: typing.Literal["rgb_array", "human"] | None = None
):
    """
    Create the bicycle environment with the correct wrapper stack.

    Wrapper order (inside → outside):
        BicycleCarEnv
            - PathProgressReward      (reward shaping)
            - Action wrapper           (discrete or continuous)
            
            - WithBaseInfo (position, heading, velocity)
            - WithPathInfo (waypoints + CTE in ego frame)
            - TimeLimit
    """
    env = gymnasium_driving.CarEnvironment(
        model="bicycle",
        road_network=gymnasium_driving.components.roads.RoadNetwork(roads=[
            gymnasium_driving.components.roads.create_rectangular_track(
                center=(50.0, 50.0),
                length=80.0,
                height=40.0,
                turn_radius=8.0,
                width=8.0,
            )
        ]),
        # TODO: disable rendering
        render_mode=render_mode,
        spawn=((50.0, 30.0), 0.0),
        goal=((10.0, 50.0), 2.0),
        obstacles=[
            gymnasium_driving.components.obstacles.Circle(center=(90, 50), radius=1.0),
        ],
    )

    env = gymnasium_driving.wrappers.rewards.PathProgressReward(
        env,
        target_velocity=5.0,
        velocity_weight=1.0,
        cte_weight=0.3,
        heading_weight=0.2,
    )

    if discrete:
        env = gymnasium_driving.wrappers.actions.DiscreteActionWrapper(env)
    else:
        env = gymnasium_driving.wrappers.actions.ContinuousActionWrapper(env)

    env = gymnasium_driving.wrappers.observations.WithBaseInfo(env)
    env = gymnasium_driving.wrappers.observations.WithPathInfo(env)
    env = gymnasium_driving.wrappers.observations.WithObstaclesInfo(env)

    if random_obstacles:
        env = RandomPathObstacles(env, num_obstacles=1)

    # 1000 steps × 0.1 s = 100 s ≈ 2 laps at 5 m/s
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)

    return env