import environments
import environments.bicycle as bicycle

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

# rgb_array, human
def build_cristal_environment(render_mode: str):
    return environments.bicycle.BicycleCarEnv(
        road_network=bicycle.RoadNetwork(roads=[
            bicycle.create_rectangular_track(
                center=(50.0, 50.0),
                length=80.0,
                width=40.0,
                turn_radius=8.0,
                lane_config=bicycle.lane_config_from_width(8.0, num_lanes=1),
            )
        ]),
        render_mode=render_mode,
        wheelbase=1.75,
        max_steering=np.pi / 4,
        max_velocity=15.0,
        world_size=(100.0, 100.0),
        spawn_pos=(50.0, 30.0),
        goal_pos=(10.0, 50.0),
        goal_radius=2.0,
        obstacles=[
            bicycle.Circle(center=(87.50, 50), radius=3.0),
        ],
        solid_road_borders=True,
        # 0.1second = 100ms per step
        dt=0.1
    )