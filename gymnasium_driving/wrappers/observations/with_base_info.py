import gymnasium

import numpy as np

class WithBaseInfo(gymnasium.ObservationWrapper):
    """
    [position x, position y, heading theta, velocity v]
    """
    
    def __init__(
        self,
        environment: gymnasium.Env,
        with_position: bool = True
    ):
        super().__init__(environment)
        
        self.env = environment
        self.with_position = with_position
        
        new_spaces = dict(self.observation_space.spaces)
        
        if self.with_position:
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
        if self.with_position:
            observation["base/position"] = np.array([self.env.unwrapped.state["x"], self.env.unwrapped.state["y"]], dtype=np.float32)
        observation["base/heading"] = np.array([self.env.unwrapped.state["yaw"]], dtype=np.float32)
        observation["base/velocity"] = np.array([self.env.unwrapped.state["velocity"]], dtype=np.float32)
        
        return observation
