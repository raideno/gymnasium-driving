import time
import helpers

import environments
import environments.bicycle as bicycle

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

def build_cristal_environment():
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
        render_mode="rgb_array",
        wheelbase=1.75,
        max_steering=np.pi / 4,
        max_velocity=15.0,
        world_size=(100.0, 100.0),
        spawn_pos=(50.0, 30.0),
        goal_pos=(10.0, 50.0),
        goal_radius=2.0,
        obstacles=[
            # Static obstacle (default behavior)
            bicycle.Circle(center=(87.50, 50), radius=3.0),
            
            # Moving obstacle: oscillates left-right on the track
            bicycle.Circle(
                center=(50.0, 70.0), 
                radius=2.5,
                motion=bicycle.oscillate(axis='y', amplitude=15.0, frequency=0.1)
            ),
            
            # Moving obstacle: circular motion
            # bicycle.Circle(
            #     center=(14.0, 50.0), 
            #     radius=2.0,
            #     motion=bicycle.circular(radius=5.0, angular_speed=0.5)
            # ),
            
            # Moving obstacle: follows waypoints
            # bicycle.Circle(
            #     center=(50.0, 30.0),
            #     radius=2.0,
            #     motion=bicycle.waypoints(
            #         points=[(30.0, 30.0), (70.0, 30.0), (70.0, 70.0), (30.0, 70.0)],
            #         speed=5.0,
            #         loop=True
            #     )
            # ),
        ],
        solid_road_borders=True,
        # 0.1second = 100ms per step
        dt=0.1
    )