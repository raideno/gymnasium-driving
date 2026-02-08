"""
Continuous action wrapper with a symmetric 2D action space.

PPO's Gaussian policy initializes with mean ≈ 0.  When the action space
is [steering, throttle, brake, reverse] with throttle ∈ [0, 1],
the initial mean (0) equals minimum throttle → the car never accelerates.
Additionally, throttle and brake are sampled independently so the car
simultaneously accelerates and brakes.

This wrapper solves both problems by:
  1. Reducing the action space to 2 dimensions: [steering, acceleration]
  2. Making the space symmetric around 0: both dims ∈ [-1, 1]
  3. Mapping positive acceleration → throttle, negative → brake (mutually exclusive)
"""

import gymnasium
import numpy as np

class ContinuousActionWrapper(gymnasium.ActionWrapper):
    """
    Maps a symmetric 2D action space to the environment's 4D space.

    Agent outputs:
        action[0]  steering_norm  ∈ [-1, 1]  →  [-MAX_STEERING, MAX_STEERING]
        action[1]  accel_norm     ∈ [-1, 1]  →  positive = throttle, negative = brake

    This is much better for Gaussian policies because:
    - The space is symmetric around 0  (initial mean is sensible)
    - Only 2 dimensions instead of 4   (easier to explore)
    - Throttle and brake are mutually exclusive by construction
    """

    def __init__(self, env: gymnasium.Env):
        super().__init__(env)

        self.action_space = gymnasium.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
        )

    def action(self, action: np.ndarray) -> np.ndarray:
        steering = float(action[0]) * self.env.unwrapped.MAX_STEERING
        accel = float(action[1])

        if accel >= 0:
            throttle = accel
            brake = 0.0
        else:
            throttle = 0.0
            brake = -accel

        return np.array([steering, throttle, brake, 0.0], dtype=np.float32)
