from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from evaluate import evaluate
from premade_environment import make_environment

def main():
    total_timesteps = 1_000_000
    env = Monitor(make_environment(discrete=True))
    
    print("[observation_space]:", env.observation_space)
    print("[action_space]:", env.action_space)
    
    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,   # encourage exploration of all 9 actions
        verbose=1,
    )

    print("[total_timesteps]:", total_timesteps)
    
    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    evaluate(model, env)

if __name__ == "__main__":
    main()
  