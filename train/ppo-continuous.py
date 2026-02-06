import sys
import os
import time

import numpy as np
import gymnasium as gym

import matplotlib.pyplot as plt

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor

import environments.bicycle as bicycle

from train.environment import make_environment
from train.evaluate import evaluate

def main(total_timesteps: int = 500_000):
    """PPO with ContinuousActionWrapper (2D symmetric [steer, accel])."""
    print("=" * 60)
    print(" PPO  +  Continuous 2-D actions  +  PathProgressReward")
    print("=" * 60)

    env = Monitor(make_environment(discrete=False))
    print(f"  observation_space : {env.observation_space}")
    print(f"  action_space      : {env.action_space}")

    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.0,
        verbose=1,
    )

    print(f"\n  Training for {total_timesteps:,} timesteps …\n")
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    print("\n  Training complete ✓")

    evaluate(model, env)
    return model, env

if __name__ == "__main__":
    main()
