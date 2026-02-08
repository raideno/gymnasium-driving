import stable_baselines3

class CustomCallback(stable_baselines3.common.callbacks.BaseCallback):
    def __init__(self):
        super().__init__()
        
        self.episode_rewards = []
        self.episode_lengths = []
        
        self.current_episode_reward = 0
        
    def _on_step(self) -> bool:
        self.current_episode_reward += self.locals["rewards"][0]
        
        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.locals["infos"][0].get("episode", {}).get("l", 0))
            
            self.current_episode_reward = 0
            
        self.logger.record("custom/key", -1)
        
        return True
