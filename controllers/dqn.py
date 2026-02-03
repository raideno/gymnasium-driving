import stable_baselines3

import environments
import environments.bicycle as bicycle

class DQNController:
    def __init__(
        self,
        env
    ):
        self.env = env
        
        # base.position (x, y), base.heading (theta), base.velocity (v)
        # TODO: moved to outside
        # self.env = environments.bicycle.wrappers.observations.WithBaseInfo(self.env)
        # self.env = environments.bicycle.wrappers.actions.MultiDiscreteActionWrapper(self.env)
        # self.env = environments.bicycle.wrappers.actions.FlattenMultiDiscreteWrapper(self.env)
        
        # TODO: why can't we use MlpPolicy when using dict observation space ?
        # MlpPolicy accepts only one input / one dimension observations ?
        # self.model = stable_baselines3.DQN("MlpPolicy", self.env)
        
        self.model = stable_baselines3.DQN(
            "MultiInputPolicy",
            self.env,
            verbose=1
        )
        
    def train(
        self,
        total_timesteps,
        log_interval,
    ):
        self.model.learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval,
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
        