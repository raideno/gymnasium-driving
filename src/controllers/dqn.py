import omegaconf
import stable_baselines3

class DQNController:
    """
    Requires a Discrete Observation Space.
    """
    def __init__(
        self,
        environment,
        **kwargs
    ):
        self.env = environment
        
        # TODO: why can't we use MlpPolicy when using dict observation space ?
        # MlpPolicy accepts only one input / one dimension observations ?
        # self.model = stable_baselines3.DQN("MlpPolicy", self.env)
        
        model_kwargs = kwargs.get("model_kwargs", {})
        if isinstance(model_kwargs, omegaconf.DictConfig):
            model_kwargs = omegaconf.OmegaConf.to_container(model_kwargs, resolve=True)
        
        self.model = stable_baselines3.DQN(
            "MultiInputPolicy",
            self.env,
            **model_kwargs
        )
        
    def learn(
        self,
        **kwargs
    ):
        self.model.learn(
            # total_timesteps=total_timesteps,
            # log_interval=log_interval
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