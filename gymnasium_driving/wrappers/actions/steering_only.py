import gymnasium

import numpy as np

class SteeringOnlyActionWrapper(gymnasium.ActionWrapper):
    def __init__(
        self,
        environment: gymnasium.Env,
        target_velocity: float,
    ):
        super().__init__(environment)
        
        self.env = environment

        max_velocity = float(self.env.unwrapped.MAX_VELOCITY)
        if target_velocity < 0.0:
            raise ValueError("target_velocity must be non-negative because reverse is disabled")
        if target_velocity > max_velocity:
            raise ValueError(
                f"target_velocity ({target_velocity}) cannot exceed env MAX_VELOCITY ({max_velocity})"
            )

        self.target_velocity = float(target_velocity)

        self.action_space = gymnasium.spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def action(self, action: np.ndarray) -> np.ndarray:
        steering_norm = float(np.asarray(action).reshape(-1)[0])
        steering = np.clip(
            steering_norm,
            -1.0,
            1.0,
        ) * float(self.env.unwrapped.MAX_STEERING)

        current_velocity = self.env.unwrapped.state["velocity"]

        velocity_error = self.target_velocity - current_velocity

        if velocity_error > 0.1: # some tolerance
            throttle_gain = 0.5
            throttle = np.clip(throttle_gain * velocity_error, 0.0, 1.0)
        else:
            throttle = 0.0

        brake = 0.0
        reverse = 0.0

        return np.array([steering, throttle, brake, reverse], dtype=np.float32)


class DiscreteSteeringOnlyActionWrapper(gymnasium.ActionWrapper):
    def __init__(
        self,
        environment: gymnasium.Env,
        target_velocity: float,
        n_steering: int = 5,
    ):
        super().__init__(environment)

        self.env = environment

        max_velocity = float(self.env.unwrapped.MAX_VELOCITY)
        if target_velocity < 0.0:
            raise ValueError("target_velocity must be non-negative because reverse is disabled")
        if target_velocity > max_velocity:
            raise ValueError(
                f"target_velocity ({target_velocity}) cannot exceed env MAX_VELOCITY ({max_velocity})"
            )
        if n_steering < 2:
            raise ValueError("n_steering must be >= 2")

        self.target_velocity = float(target_velocity)
        self.n_steering = int(n_steering)

        max_steer = float(self.env.unwrapped.MAX_STEERING)
        self.steering_levels = np.linspace(-max_steer, max_steer, self.n_steering)
        self.action_space = gymnasium.spaces.Discrete(self.n_steering)

    def action(self, action: int) -> np.ndarray:
        steering_idx = int(action)
        steering = float(self.steering_levels[steering_idx])

        current_velocity = self.env.unwrapped.state["velocity"]
        velocity_error = self.target_velocity - current_velocity

        if velocity_error > 0.1:  # some tolerance
            throttle_gain = 0.5
            throttle = np.clip(throttle_gain * velocity_error, 0.0, 1.0)
        else:
            throttle = 0.0

        brake = 0.0
        reverse = 0.0

        return np.array([steering, throttle, brake, reverse], dtype=np.float32)
