import sys
import os
import time

import numpy as np
import gymnasium as gym

import matplotlib.pyplot as plt

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor

import environments.bicycle as bicycle

def make_environment(discrete: bool = True):
    """
    Create the bicycle environment with the correct wrapper stack.

    Wrapper order (inside → outside):
        BicycleCarEnv
        → PathProgressReward      (reward shaping)
        → Action wrapper           (discrete or continuous)
        → WithBaseInfo             (heading + velocity observations)
        → WithPathInfo             (waypoints + CTE in ego frame)
        → TimeLimit
    """
    env = bicycle.BicycleCarEnv(
        road_network=bicycle.RoadNetwork(roads=[
            bicycle.create_rectangular_track(
                center=(50.0, 50.0),
                length=80.0,
                height=40.0,
                turn_radius=8.0,
                width=8.0,
            )
        ]),
        render_mode="rgb_array",
        spawn=((50.0, 30.0), 0.0),
        goal=((10.0, 50.0), 2.0),
        obstacles=[
            bicycle.Circle(center=(90, 50), radius=1.0),
        ],
        solid_road_borders=True,
        dt=0.1,
    )

    env = bicycle.wrappers.rewards.PathProgressReward(
        env,
        target_velocity=5.0,
        velocity_weight=1.0,
        cte_weight=0.3,
        heading_weight=0.2,
    )

    if discrete:
        env = bicycle.wrappers.actions.DiscreteActionWrapper(env)
    else:
        env = bicycle.wrappers.actions.ContinuousActionWrapper(env)

    env = bicycle.wrappers.observations.WithBaseInfo(env)
    env = bicycle.wrappers.observations.WithPathInfo(env)

    # 1000 steps × 0.1 s = 100 s ≈ 2 laps at 5 m/s
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)

    return env