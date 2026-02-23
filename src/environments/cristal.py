import typing

import gymnasium as gym

import gymnasium_driving
import gymnasium_driving.wrappers

def make_environment(
    discrete: bool,
    render_mode: typing.Literal["rgb_array", "human"] | None = None,
    target_velocity: float = 2.5,
    n_steering: int = 15,
    max_steps: int = 600,
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
                    center=(50, 50),
                    length=80,
                    height=40,
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
            target_velocity=target_velocity,
            n_steering=n_steering,
        )
    else:
        env = gymnasium_driving.wrappers.actions.SteeringOnlyActionWrapper(
            env,
            target_velocity=target_velocity,
        )

    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)

    return env