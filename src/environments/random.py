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
        exclude_start_distance=10.0,
        exclude_goal_distance=10.0,
        seed=None
    ):
        super().__init__(env)
        
        self.num_obstacles = num_obstacles
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.exclude_start = exclude_start_distance
        self.exclude_goal = exclude_goal_distance
        self.rng = np.random.default_rng(seed)
    
    def reset(self, **kwargs):
        # Sample points along the path (excluding start/goal areas)
        path = self.unwrapped.path
        path_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
        cumulative = np.concatenate([[0], np.cumsum(path_lengths)])
        total_length = cumulative[-1]
        
        valid_start = self.exclude_start
        valid_end = total_length - self.exclude_goal
        
        obstacles = []
        
        for _ in range(self.num_obstacles):
            t = self.rng.uniform(valid_start, valid_end)
            idx = np.searchsorted(cumulative, t) - 1
            local_t = (t - cumulative[idx]) / path_lengths[idx]
            pos = path[idx] + local_t * (path[idx + 1] - path[idx])
            radius = self.rng.uniform(self.min_radius, self.max_radius)
            obstacles.append(Circle(center=tuple(pos), radius=radius))
        
        self.unwrapped.obstacles = obstacles
        
        return super().reset(**kwargs)