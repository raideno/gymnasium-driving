import gymnasium
import numpy


class WithBaseInfo(gymnasium.ObservationWrapper):
    """
    A gymnasium.ObservationWrapper that augments the
    gymnasium_driving.environment.CarEnvironment observation dictionary
    with basic robot/base state.

    Added observation keys:

    - "base/heading" (Box, shape=(1,))
        Contains [yaw] in radians

    - "base/velocity" (Box, shape=(1,))
        Contains [velocity]

    All values are returned as numpy.float32.

    Requirements:
        The wrapped environment must expose:

        - env.unwrapped.state with keys:
            - "x"
            - "y"
            - "yaw"
            - "velocity"

        - env.unwrapped.MAX_VELOCITY
    """

    def __init__(self, environment: gymnasium.Env):
        super().__init__(environment)

        self.env = environment

        new_spaces = dict(self.observation_space.spaces)

        new_spaces["base/heading"] = gymnasium.spaces.Box(
            -numpy.pi, numpy.pi, shape=(1,), dtype=numpy.float32
        )

        new_spaces["base/velocity"] = gymnasium.spaces.Box(
            -self.env.unwrapped.MAX_VELOCITY,
            self.env.unwrapped.MAX_VELOCITY,
            shape=(1,),
            dtype=numpy.float32,
        )

        self.observation_space = gymnasium.spaces.Dict(new_spaces)

    def observation(self, observation: dict) -> dict:
        observation["base/heading"] = numpy.array(
            [self.env.unwrapped.state["yaw"]],
            dtype=numpy.float32,
        )
        observation["base/velocity"] = numpy.array(
            [self.env.unwrapped.state["velocity"]],
            dtype=numpy.float32,
        )

        return observation
