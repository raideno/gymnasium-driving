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
    

