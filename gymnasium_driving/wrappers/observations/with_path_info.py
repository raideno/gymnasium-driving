import gymnasium

import numpy as np

from gymnasium_driving.helpers import curvature_windowed

class WithPathInfo(gymnasium.ObservationWrapper):
    """
    Adds path following information to observations.
    
    Provides:
    - Upcoming waypoints in ego frame
    - Cross-track error to the path
    - Heading error relative to path tangent
    - Path curvature at lookahead points
    - Distance traveled along path
    - Progress towards goal (percentage)
    
    Args:
        env: The environment to wrap
        num_waypoints: Number of upcoming waypoints to include
    """
    
    def __init__(
        self,
        environment: gymnasium.Env,
        num_waypoints: int = 10,
    ):
        super().__init__(environment)
        
        self.env = environment
        
        self.num_waypoints = num_waypoints
        
        # [x, y, curvature]
        self.waypoint_dim = 3
        
        new_spaces = dict(self.observation_space.spaces)
        
        # waypoints in ego frame
        new_spaces["path/waypoints"] = gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_waypoints, self.waypoint_dim),
            dtype=np.float32,
        )
        
        # path info: [cte, heading_error, progress, goal_distance]
        new_spaces["path/info"] = gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4,),
            dtype=np.float32,
        )
        
        self.observation_space = gymnasium.spaces.Dict(new_spaces)
        
    def observation(self, observation: dict) -> dict:
        ego_pos = np.array([self.env.unwrapped.state["x"], self.env.unwrapped.state["y"]], dtype=np.float32)
        ego_heading = self.env.unwrapped.state["yaw"]
        
        path = self.env.unwrapped.path
        
        waypoints = np.zeros((self.num_waypoints, self.waypoint_dim), dtype=np.float32)
        path_info = np.zeros(4, dtype=np.float32)
        
        if path is None or len(path) < 2:
            observation["path/waypoints"] = waypoints
            observation["path/info"] = path_info
            return observation
        
        closest_point_index = self.env.unwrapped.state["closest_path_idx"]
        cte = self.env.unwrapped.state["cte"]
        heading_error = self.env.unwrapped.state["heading_error"]
        
        # NOTE: normalized progress along the path
        progress = closest_point_index / max(len(path) - 1, 1)
        
        # NOTE: goal distance
        # TODO: should be measured along the path, not straight line distance
        goal_distance = np.linalg.norm(ego_pos - self.env.unwrapped.goal_pos)
        
        path_info = np.array([cte, heading_error, progress, goal_distance], dtype=np.float32)
        
        # NOTE: waypoints ahead
        cos_h, sin_h = np.cos(-ego_heading), np.sin(-ego_heading)
        rotation = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
        
        for i in range(self.num_waypoints):
            waypoint_idx = closest_point_index + i + 1
            
            if waypoint_idx >= len(path):
                # NOTE: pad with last waypoint
                wp = path[len(path) - 1]
                current_idx = len(path) - 1
            else:
                wp = path[waypoint_idx]
                current_idx = waypoint_idx
            
            rel_pos = rotation @ (wp - ego_pos)
            curvature = curvature_windowed(path, current_idx, window=3)
            
            waypoints[i] = [rel_pos[0], rel_pos[1], curvature]
        
        observation["path/waypoints"] = waypoints
        observation["path/info"] = path_info
        
        return observation
