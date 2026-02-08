import gymnasium

import numpy as np

class DiscreteActionWrapper(gymnasium.ActionWrapper):
    """
    Discretizes the continuous action space into a simple discrete set.
    
    Actions:
        0: Do nothing (coast)
        1: Accelerate forward
        2: Brake
        3: Steer left + accelerate
        4: Steer right + accelerate
        5: Steer left + coast
        6: Steer right + coast
        7: Hard left + brake
        8: Hard right + brake
    """
    
    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        
        self.action_space = gymnasium.spaces.Discrete(9)
        
        self.action_map = {
            0: np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),  # coast
            1: np.array([0.0, 0.7, 0.0, 0.0], dtype=np.float32),  # accelerate
            2: np.array([0.0, 0.0, 0.8, 0.0], dtype=np.float32),  # brake
            3: np.array([-self.env.unwrapped.MAX_STEERING * 0.5, 0.6, 0.0, 0.0], dtype=np.float32),  # left + accel
            4: np.array([self.env.unwrapped.MAX_STEERING * 0.5, 0.6, 0.0, 0.0], dtype=np.float32),  # right + accel
            5: np.array([-self.env.unwrapped.MAX_STEERING * 0.3, 0.0, 0.0, 0.0], dtype=np.float32),  # left + coast
            6: np.array([self.env.unwrapped.MAX_STEERING * 0.3, 0.0, 0.0, 0.0], dtype=np.float32),  # right + coast
            7: np.array([-self.env.unwrapped.MAX_STEERING, 0.0, 0.5, 0.0], dtype=np.float32),  # hard left + brake
            8: np.array([self.env.unwrapped.MAX_STEERING, 0.0, 0.5, 0.0], dtype=np.float32),  # hard right + brake
        }
    
    def action(self, action) -> np.ndarray:
        return self.action_map[int(action)]
    
class CoarseDiscreteActionWrapper(gymnasium.ActionWrapper):
    """
    Simplified discrete action space with only essential actions.
    Useful for faster learning with fewer actions to explore.
    
    Actions:
        0: Coast (no steering, no throttle, no brake)
        1: Accelerate straight
        2: Brake
        3: Steer left
        4: Steer right
    """
    
    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        
        self.action_space = gymnasium.spaces.Discrete(5)
        
        self.action_map = {
            0: np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),  # coast
            1: np.array([0.0, 0.5, 0.0, 0.0], dtype=np.float32),  # accelerate
            2: np.array([0.0, 0.0, 0.6, 0.0], dtype=np.float32),  # brake
            3: np.array([-self.env.unwrapped.MAX_STEERING * 0.4, 0.4, 0.0, 0.0], dtype=np.float32),  # left
            4: np.array([self.env.unwrapped.MAX_STEERING * 0.4, 0.4, 0.0, 0.0], dtype=np.float32),  # right
        }
    
    def action(self, action) -> np.ndarray:
        return self.action_map[int(action)]

class MultiDiscreteActionWrapper(gymnasium.ActionWrapper):
    """
    Discretizes each action component separately.
    
    Action space: MultiDiscrete([n_steering, n_throttle, n_brake])
    """
    
    def __init__(
        self,
        env: gymnasium.Env,
        n_steering: int = 5,
        n_throttle: int = 3,
        n_brake: int = 3,
    ):
        super().__init__(env)
        
        self.n_steering = n_steering
        self.n_throttle = n_throttle
        self.n_brake = n_brake
        
        self.action_space = gymnasium.spaces.MultiDiscrete(
            [n_steering, n_throttle, n_brake]
        )
        
        max_steer = self.env.unwrapped.MAX_STEERING
        
        self.steering_levels = np.linspace(-max_steer, max_steer, n_steering)
        self.throttle_levels = np.linspace(0.0, 1.0, n_throttle)
        self.brake_levels = np.linspace(0.0, 1.0, n_brake)
    
    def action(self, action: np.ndarray) -> np.ndarray:
        steering_idx, throttle_idx, brake_idx = action
        
        steering = self.steering_levels[steering_idx]
        throttle = self.throttle_levels[throttle_idx]
        brake = self.brake_levels[brake_idx]
        
        return np.array([steering, throttle, brake, 0.0], dtype=np.float32)

class FlattenMultiDiscreteWrapper(gymnasium.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # NOTE: total number of combinations
        # self.nvec = env.unwrapped.action_space.nvec
        self.nvec = env.action_space.nvec
        
        self.total_actions = np.prod(self.nvec)
        self.action_space = gymnasium.spaces.Discrete(self.total_actions)
    
    def action(self, action):
        # NOTE: flat action -> MultiDiscrete
        result = []
        for dim in reversed(self.nvec):
            result.append(action % dim)
            action //= dim
        return np.array(list(reversed(result)), dtype=np.int64)
    
    def reverse_action(self, action):
        # MultiDiscrete action -> flat (for logging/debugging)
        flat = 0
        multiplier = 1
        for a, dim in zip(reversed(action), reversed(self.nvec)):
            flat += a * multiplier
            multiplier *= dim
        return flat
