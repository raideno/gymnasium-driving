import gymnasium

import numpy as np

class WithBaseInfo(gymnasium.ObservationWrapper):
    """
    [position x, position y, heading theta, velocity v]
    """
    
    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        
        new_spaces = dict(self.observation_space.spaces)
        
        new_spaces["base/position"] = gymnasium.spaces.Box(
            -np.inf, np.inf, shape=(2,), dtype=np.float32
        )
        
        new_spaces["base/heading"] = gymnasium.spaces.Box(
            -np.pi, np.pi, shape=(1,), dtype=np.float32
        )
        
        new_spaces["base/velocity"] = gymnasium.spaces.Box(
            -self.env.unwrapped.MAX_VELOCITY,
            self.env.unwrapped.MAX_VELOCITY,
            shape=(1,),
            dtype=np.float32,
        )
        
        self.observation_space = gymnasium.spaces.Dict(new_spaces)
    
    def observation(self, observation: dict) -> dict:
        state = self.env.unwrapped.state
        
        observation["base/position"] = state[:2].copy()
        observation["base/heading"] = np.array([state[2]], dtype=np.float32)
        observation["base/velocity"] = np.array([state[3]], dtype=np.float32)
        
        return observation
