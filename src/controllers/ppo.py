import stable_baselines3

class PPOController:
    """
    Requires a Discrete Observation Space.
    """
    def __init__(
        self,
        environment,
        **kwargs,
    ):
        self.env = environment
        
        self.model = stable_baselines3.PPO(
            "MultiInputPolicy",
            self.env,
            **kwargs.get("model_kwargs", {})
        )
        
    def learn(
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
        
    def predict(
        self,
        observation,
        **kwargs
    ):
        return self.model.predict(
            observation,
            deterministic=True
        )
        
    def draw_debug(
        self,
        observation,
        **kwargs
    ):
        pass