import gymnasium
import numpy as np

from gymnasium_driving.components.obstacles import Circle

class RandomPathObstacles(gymnasium.Wrapper):
    def __init__(
        self,
        env,
        num_obstacles=1,
        min_radius=0.5,
        max_radius=2.0,
        exclude_start_distance=1.0,
        exclude_goal_distance=1.0,
        seed=None,
    ):
        super().__init__(env)

        self.num_obstacles = num_obstacles
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.exclude_start = exclude_start_distance
        self.exclude_goal = exclude_goal_distance
        self.rng = np.random.default_rng(seed)

    def reset(self, **kwargs):
        # NOTE: to ensure spawn position is set
        obs = super().reset(**kwargs)
        
        path = self.unwrapped.path
        path_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
        cumulative = np.concatenate([[0], np.cumsum(path_lengths)])
        total_length = cumulative[-1]

        spawn_pos = self.unwrapped.spawn_pos
        
        # Find the closest point on the path to the spawn position
        distances = np.linalg.norm(path - spawn_pos, axis=1)
        spawn_idx = np.argmin(distances)
        spawn_distance_along_path = cumulative[spawn_idx]
        
        # Obstacles should only be spawned ahead of the car
        valid_start = spawn_distance_along_path + self.exclude_start
        valid_end = total_length - self.exclude_goal
        
        obstacles = []
        
        if valid_start < valid_end:
            for _ in range(self.num_obstacles):
                t = self.rng.uniform(valid_start, valid_end)
                idx = np.searchsorted(cumulative, t) - 1
                local_t = (t - cumulative[idx]) / path_lengths[idx]
                pos = path[idx] + local_t * (path[idx + 1] - path[idx])
                radius = self.rng.uniform(self.min_radius, self.max_radius)
                obstacles.append(Circle(center=tuple(pos), radius=radius))

        self.unwrapped.obstacles = obstacles

        return obs

class RandomSpawn(gymnasium.Wrapper):
    """
    Randomize where on the path the car starts each episode.

    This forces the agent to learn path-following from *any* position
    on the track rather than memorizing a single starting point.
    """

    def __init__(
        self,
        env,
        # What fraction of the path is eligible as a spawn point.
        # 0.0-1.0 — e.g. 0.5 means the first half of the path.
        path_fraction: float = 0.25,
        
        lateral_noise: float = 1.0,
        heading_noise: float = 0.15,
        
        seed=None,
    ):
        super().__init__(env)
        
        self.path_fraction = path_fraction
        self.lateral_noise = lateral_noise
        self.heading_noise = heading_noise
        
        self.rng = np.random.default_rng(seed)

    def reset(self, **kwargs):
        path = self.unwrapped.path
        
        if path is not None and len(path) >= 2:
            max_idx = max(1, int(len(path) * self.path_fraction))
            idx = self.rng.integers(0, max_idx)

            spawn_pos = path[idx].copy()

            if idx < len(path) - 1:
                tangent = path[idx + 1] - path[idx]
            else:
                tangent = path[idx] - path[idx - 1]
            heading = float(np.arctan2(tangent[1], tangent[0]))

            # NOTE: lateral noise (perpendicular to path)
            normal = np.array([-tangent[1], tangent[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-6:
                normal = normal / norm
            spawn_pos += normal * self.rng.uniform(
                -self.lateral_noise, self.lateral_noise
            )

            # NOTE: heading noise
            heading += self.rng.uniform(
                -self.heading_noise, self.heading_noise
            )

            self.unwrapped.spawn_pos = spawn_pos.astype(np.float32)
            self.unwrapped.spawn_heading = heading

        return super().reset(**kwargs)