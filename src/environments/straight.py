import typing

import gymnasium

import gymnasium_driving
import gymnasium_driving.components
import gymnasium_driving.components.roads
import gymnasium_driving.environment
import gymnasium_driving.factories
import gymnasium_driving.wrappers


def make_environment(
    discrete: bool,
    render_mode: typing.Literal["rgb_array"] | None = None,
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
        positions_factory=gymnasium_driving.factories.make_centerline_positions_factory(),
        obstacles_factory=gymnasium_driving.factories.make_random_obstacles_factory(
            num_obstacles=1, min_spacing_m=20.0
        ),
        render_mode=render_mode,
    )

    if discrete:
        env = gymnasium_driving.wrappers.actions.DiscretizeActionWrapper(
            env, n_steering=5, n_throttle=3, n_brake=2
        )

    env = gymnasium.wrappers.TimeLimit(env, max_episode_steps=max_steps)

    return env
