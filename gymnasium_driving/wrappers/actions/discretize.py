import gymnasium
import numpy

import gymnasium_driving


class DiscretizeActionWrapper(gymnasium.ActionWrapper):
    def __init__(
        self,
        environment: gymnasium.Env,
        n_steering: int = 5,
        n_throttle: int = 3,
        n_brake: int = 2,
    ):
        super().__init__(environment)

        if n_steering < 2:
            raise ValueError("n_steering must be >= 2")
        if n_throttle < 1:
            raise ValueError("n_throttle must be >= 1")
        if n_brake < 1:
            raise ValueError("n_brake must be >= 1")

        self.n_steering = int(n_steering)
        self.n_throttle = int(n_throttle)
        self.n_brake = int(n_brake)

        max_steer = float(gymnasium_driving.CarEnvironment.MAX_STEERING)

        self.steering_levels = numpy.linspace(
            -max_steer,
            max_steer,
            self.n_steering,
            dtype=numpy.float32,
        )

        self.throttle_levels = numpy.linspace(
            0.0,
            1.0,
            self.n_throttle,
            dtype=numpy.float32,
        )

        self.brake_levels = numpy.linspace(
            0.0,
            1.0,
            self.n_brake,
            dtype=numpy.float32,
        )

        self._total_actions = self.n_steering * self.n_throttle * self.n_brake

        self.action_space = gymnasium.spaces.Discrete(self._total_actions)

    def action(self, action: int) -> numpy.ndarray:
        idx = int(action)

        if idx < 0 or idx >= self._total_actions:
            raise ValueError(
                f"Action index {idx} out of bounds [0, {self._total_actions - 1}]"
            )

        steering_idx = idx // (self.n_throttle * self.n_brake)
        remainder = idx % (self.n_throttle * self.n_brake)

        throttle_idx = remainder // self.n_brake
        brake_idx = remainder % self.n_brake

        steering = float(self.steering_levels[steering_idx])
        throttle = float(self.throttle_levels[throttle_idx])
        brake = float(self.brake_levels[brake_idx])

        # NOTE: reverse disabled
        reverse = 0.0

        return numpy.array(
            [steering, throttle, brake, reverse],
            dtype=numpy.float32,
        )
