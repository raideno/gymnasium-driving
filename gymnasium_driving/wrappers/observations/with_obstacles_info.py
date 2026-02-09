import typing
import gymnasium

import numpy as np

from gymnasium_driving.components.obstacles import Obstacle, Circle, Rectangle

class WithObstaclesInfo(gymnasium.ObservationWrapper):
    """
    Adds obstacle information to observations.
    
    For each obstacle within detection range, provides:
    # TODO: make a toggle to whether to provide in ego frame or global frame
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
        environment: gymnasium.Env,
        detection_range: float = 50.0,
        max_obstacles: int = 10,
    ):
        super().__init__(environment)
        
        self.env = environment
        
        self.detection_range = detection_range
        self.max_obstacles = max_obstacles
        self.ego_frame = True
        
        # TODO: add rel_x > 0
        # TODO: add time to collision ttc
        # [exists, rel_x, rel_y, distance, radius/size]
        self.obstacle_instance_dimension = 5
        
        new_spaces = dict(self.observation_space.spaces)
        new_spaces["obstacles/instances"] = gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.max_obstacles, self.obstacle_instance_dimension),
            dtype=np.float32,
        )
        new_spaces["obstacles/num_obstacles_detected"] = gymnasium.spaces.Box(
            low=0,
            high=self.max_obstacles,
            shape=(1,),
            dtype=np.float32,
        )
        self.observation_space = gymnasium.spaces.Dict(new_spaces)
        
        self._prev_obstacle_positions: typing.Dict[int, np.ndarray] = {}
    
    def observation(self, observation: dict) -> dict:
        ego_position = np.array([self.env.unwrapped.state["x"], self.env.unwrapped.state["y"]], dtype=np.float32)
        ego_heading = self.env.unwrapped.state["yaw"]
        
        # NOTE: rotation matrix to transform to ego frame
        cos_h, sin_h = np.cos(-ego_heading), np.sin(-ego_heading)
        rotation_matrix = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
        
        obstacle_data = []
        
        for i, obstacle in enumerate(self.env.unwrapped.obstacles):
            rel_pos_world = np.array(obstacle.center, dtype=np.float32) - ego_position
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
            obstacle_data.append([0.0] * self.obstacle_instance_dimension)
        
        observation["obstacles/instances"] = np.array(obstacle_data, dtype=np.float32)
        observation["obstacles/num_obstacles_detected"] = np.array([num_detected], dtype=np.float32)
        
        return observation
    
    def reset(self, **kwargs):
        self._prev_obstacle_positions.clear()
        return super().reset(**kwargs)
