import sb3_contrib

class TRPOController:
    """
    Requires a Discrete Observation Space.
    """
    def __init__(
        self,
        environment,
        **kwargs,
    ):
        self.env = environment
        
        self.model = sb3_contrib.TRPO(
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