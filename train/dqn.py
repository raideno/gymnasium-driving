import sys
import os
import time

import numpy as np
import gymnasium as gym

import matplotlib.pyplot as plt

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor

import environments.bicycle as bicycle

from train.evaluate import evaluate
from train.environment import make_environment

def main():
    total_timesteps = 1_000_000
   
    # NOTE: Monitor is a stable baseline thing to monitor training nicely in the terminal
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
