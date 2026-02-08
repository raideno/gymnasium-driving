from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from ..evaluate import evaluate
from premade_environment import make_environment

def main():
    total_timesteps = 1_000_000
    env = Monitor(make_environment(discrete=True))
    
    print("[observation_space]:", env.observation_space)
    print("[action_space]:", env.action_space)

    model = DQN(
        "MultiInputPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=10_000,
        batch_size=64,
        gamma=0.99,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        verbose=1,
    )

    print("[total_timesteps]:", total_timesteps)
    
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
   
    evaluate(model, env)

if __name__ == "__main__":
    main()
