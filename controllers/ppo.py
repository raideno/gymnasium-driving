import stable_baselines3

import environments
import environments.bicycle as bicycle

class PPOController:
    """
    Requires a Discrete Observation Space.
    """
    def __init__(
        self,
        env,
        model_kwargs={}
    ):
        self.env = env
        
        self.model = stable_baselines3.PPO(
            "MultiInputPolicy",
            self.env,
            **model_kwargs
        )
        
    def train(
        self,
        **kwargs
    ):
        self.model.learn(
            **kwargs,
        )
        return self
    
    def get_action(
        self,
        observation,
        **kwargs
    ):
        return self.model.predict(
            observation,
            deterministic=True
        )
        