# Gymnasium Driving Environment Usage

This document explains how to instantiate, run, and customize `gymnasium_driving.CarEnvironment`.

## Installation

To install the environment and use it in your own project:

```bash
pip install git+https://github.com/raideno/gymnasium-driving-environment.git
```

## Core API

`CarEnvironment` is created with three factories:

- `road_network_factory(env) -> RoadNetwork`
- `positions_factory(env) -> SpawnGoalInfo`
- `obstacles_factory(env) -> list[Circle | Rectangle]`

Constructor (from `gymnasium_driving/environment.py`):

```python
CarEnvironment(
    road_network_factory=...,
    obstacles_factory=...,
    positions_factory=...,
    render_mode="rgb_array" | None,
    model="bicycle" | "ackerman",
)
```

Units are metric (`m`, `m/s`, `rad`).

## Minimal example

```python
import gymnasium_driving
import gymnasium_driving.components.roads as roads
import gymnasium_driving.factories as factories
import gymnasium_driving.wrappers as wrappers

env = gymnasium_driving.CarEnvironment(
    model="bicycle",
    road_network_factory=lambda e: roads.RoadNetwork(
        roads=[
            roads.create_rectangular_track(
                center=(50.0, 50.0),
                length=100.0,
                height=100.0,
                turn_radius=8.0,
                width=8.0,
            )
        ]
    ),
    positions_factory=factories.make_centerline_positions_factory(),
    obstacles_factory=factories.make_empty_obstacles_factory(),
    render_mode="rgb_array",
)

# Add reward and observation wrappers (recommended)
env = wrappers.rewards.Reward(env)
env = wrappers.observations.WithPathInfo(env)
env = wrappers.observations.WithBaseInfo(env)

observation, info = env.reset(seed=0)
terminated = truncated = False

while not (terminated or truncated):
    action = env.action_space.sample()  # [steering, throttle, brake, reverse]
    observation, reward, terminated, truncated, info = env.step(action)

frame = env.render()  # RGB array when render_mode="rgb_array"
env.close()
```

## Action and episode behavior

- Native action space is continuous `Box(4,)`: `[steering, throttle, brake, reverse]`
- `step()` returns `info` with `cte` and `heading_error`
- Episode ends when:
  - goal is reached (`terminated`)
  - collision occurs (`terminated`)
  - out of world bounds (`truncated`)
  - optionally off-road when `RoadNetwork.enforce_road=True`

Base environment returns empty observation `{}` and reward `0.0`; wrappers supply learning signals.

## Built-in wrappers

Observation wrappers (`gymnasium_driving/wrappers/observations`):

- `WithPathInfo`: adds `path/waypoints`, `path/info`
- `WithObstaclesInfo`: adds `obstacles/instances`, `obstacles/num_obstacles_detected`
- `WithBaseInfo`: adds base position/heading/velocity
- `WithRoadInfo`: adds road boundary distances

Action wrapper:

- `DiscretizeActionWrapper`: converts to discrete action index

Reward wrapper:

- `Reward`: progress/path tracking + smoothness + obstacle penalties

## Customization points

Roads (`gymnasium_driving/components/roads.py`):

- Use `RoadBuilder` for procedural road sequences (`straight`, `turn_left`, `turn_right`)
- Use helpers like `create_rectangular_track` or `create_oval_track`
- Control constraints with `RoadNetwork(enforce_road=..., solid_road_borders=...)`

Obstacles (`gymnasium_driving/components/obstacles.py`):

- Shapes: `Circle`, `Rectangle`
- Optional motion functions: `static`, `linear`, `circular`, `oscillate`, `figure_eight`, `waypoints`

Factories (`gymnasium_driving/factories`):

- `make_centerline_positions_factory(...)`
- `make_empty_obstacles_factory()`
- `make_random_obstacles_factory(num_obstacles, min_spacing_m)`

## Existing environment recipes

The project already contains two ready-to-use environment builders:

- `src/environments/straight.py`
- `src/environments/cristal.py`

These are wired into Hydra configs under `configurations/environment/`.
