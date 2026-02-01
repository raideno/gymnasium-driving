import gymnasium

class WithRoadInfo(gymnasium.ObservationWrapper):
    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        
    def observation(self, observation: dict) -> dict:
        pass