import typing

import gymnasium

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
    Straight road environment with random obstacles for obstacle-avoidance training.
    """
    env = gymnasium_driving.CarEnvironment(
        model="bicycle",
        road_network_factory=lambda e: gymnasium_driving.components.roads.RoadNetwork(
            roads=[
                gymnasium_driving.components.roads.Road(
                    segments=[
                        gymnasium_driving.components.roads.StraightSegment(
                            start=(10.0, 50.0),
                            heading=0.0,
                            length=100,
                        )
                    ],
                    width=8.0,
                )
            ],
        ),
        positions_factory=gymnasium_driving.environment.make_centerline_positions_factory(),
        render_mode=render_mode,
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
