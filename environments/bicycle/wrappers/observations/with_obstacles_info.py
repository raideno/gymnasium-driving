import typing
import gymnasium

import numpy as np

from ...components.obstacles import Obstacle, Circle, Rectangle

class WithObstaclesInfo(gymnasium.ObservationWrapper):
    """
    Adds obstacle information to observations.
    
    For each obstacle within detection range, provides:
    - Relative position (x, y) in ego-vehicle frame
    - Distance to obstacle
    - Obstacle type and size
    - Relative velocity (if dynamic obstacles)
    
    Args:
        env: The environment to wrap
        detection_range: Maximum distance (meters) at which obstacles are detected
        max_obstacles: Maximum number of obstacles to include in observation
    """
    
    def __init__(
        self,
        env: gymnasium.Env,
        detection_range: float = 50.0,
        max_obstacles: int = 10,
    ):
        super().__init__(env)
        
        self.detection_range = detection_range
        self.max_obstacles = max_obstacles
        self.ego_frame = True
        
        # [exists, rel_x, rel_y, distance, radius/size]
        self.obs_dim = 5
        
        new_spaces = dict(self.observation_space.spaces)
        new_spaces["obstacles/instances"] = gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(max_obstacles, self.obs_dim),
            dtype=np.float32,
        )
        new_spaces["obstacles/num_obstacles_detected"] = gymnasium.spaces.Box(
            low=0,
            high=max_obstacles,
            shape=(1,),
            dtype=np.float32,
        )
        self.observation_space = gymnasium.spaces.Dict(new_spaces)
        
        self._prev_obstacle_positions: typing.Dict[int, np.ndarray] = {}
    
    def observation(self, observation: dict) -> dict:
        state = self.env.unwrapped.state
        ego_pos = state[:2]
        ego_heading = state[2]
        
        # NOTE: rotation matrix to transform to ego frame
        cos_h, sin_h = np.cos(-ego_heading), np.sin(-ego_heading)
        rotation_matrix = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
        
        obstacle_data = []
        
        for i, obstacle in enumerate(self.env.unwrapped.obstacles):
            obs_center = np.array(obstacle.center, dtype=np.float32)
            rel_pos_world = obs_center - ego_pos
            distance = np.linalg.norm(rel_pos_world)
            
            if distance > self.detection_range:
                continue
            
            # NOTE: transform to ego frame if requested
            rel_pos = rotation_matrix @ rel_pos_world if self.ego_frame else rel_pos_world
            
            # NOTE: get obstacle size
            if isinstance(obstacle, Circle):
                size = obstacle.radius
            elif isinstance(obstacle, Rectangle):
                size = max(obstacle.width, obstacle.height) / 2
            else:
                size = 1.0
            
           
            obstacle_data.append([
                1.0,
                rel_pos[0],
                rel_pos[1],
                distance,
                size,
            ])
        
        # NOTE: sort by distance and take closest
        obstacle_data.sort(key=lambda x: x[3])
        obstacle_data = obstacle_data[:self.max_obstacles]
        
        # NOTE: pad with zeros if fewer than max_obstacles
        num_detected = len(obstacle_data)
        while len(obstacle_data) < self.max_obstacles:
            obstacle_data.append([0.0] * self.obs_dim)
        
        observation["obstacles/instances"] = np.array(obstacle_data, dtype=np.float32)
        observation["obstacles/num_obstacles_detected"] = np.array([num_detected], dtype=np.float32)
        
        return observation
    
    def reset(self, **kwargs):
        self._prev_obstacle_positions.clear()
        return super().reset(**kwargs)