import typing

import gymnasium
import numpy

import gymnasium_driving
import gymnasium_driving.components
import gymnasium_driving.components.roads
import gymnasium_driving.environment
import gymnasium_driving.wrappers


def make_environment(
    discrete: bool,
    render_mode: typing.Literal["rgb_array"] | None = None,
    target_velocity: float = 2.5,
    n_steering: int = 15,
    max_steps: int = 600,
    **kwargs,
):
    """
    Randomized road environment with domain randomization of paths and obstacles.
    """
    env = gymnasium_driving.CarEnvironment(
        model="bicycle",
        road_network_factory=lambda e: gymnasium_driving.components.roads.RoadNetwork(
            roads=[
                gymnasium_driving.components.roads.create_rectangular_track(
                    center=(numpy.random.randint(40, 60), numpy.random.randint(40, 60)),
                    length=numpy.random.randint(80, 120),
                    height=numpy.random.randint(80, 120),
                    turn_radius=numpy.random.randint(6, 10),
                    width=numpy.random.randint(6, 10),
                )
            ],
        ),
        render_mode=render_mode,
        positions_factory=gymnasium_driving.environment.make_centerline_positions_factory(),
        obstacles_factory=lambda e: [],
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

    env = gymnasium.wrappers.TimeLimit(env, max_episode_steps=max_steps)

    return env
