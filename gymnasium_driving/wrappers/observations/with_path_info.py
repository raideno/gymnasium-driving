import gymnasium

import numpy as np

from gymnasium_driving.helpers import curvature_windowed
from gymnasium_driving.helpers import wrap_to_pi, closest_polyline_index, polyline_tangent, signed_cte_to_polyline, heading_error_to_polyline

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
        env: gymnasium.Env,
        num_waypoints: int = 10,
    ):
        super().__init__(env)
        
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
        
        # NOTE: closest point on path
        closest_point_index = np.argmin(np.linalg.norm(path - ego_pos, axis=1))
        closest_point = path[closest_point_index]
        
        # NOTE: signed cross-track error
        if closest_point_index < len(path) - 1:
            path_direction = path[closest_point_index + 1] - path[closest_point_index]
        elif closest_point_index > 0:
            path_direction = path[closest_point_index] - path[closest_point_index - 1]
        else:
            path_direction = np.array([1.0, 0.0])
        path_heading = np.arctan2(path_direction[1], path_direction[0])
        
        # to_ego = ego_pos - closest_point
        # cross = float(path_direction[0] * to_ego[1] - path_direction[1] * to_ego[0])
        # NOTE: measure how much ego is to the left or right of the path using cross product
        cte = float(np.cross(path_direction, ego_pos - closest_point)) / (np.linalg.norm(path_direction) + 1e-6)
        # cte = cross / (np.linalg.norm(path_direction) + 1e-6)
        
        # NOTE: heading error
        heading_error = ego_heading - path_heading
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        
        # idx = closest_polyline_index(path, ego_pos)
        # cte, idx = signed_cte_to_polyline(path, ego_pos, idx=idx)
        # heading_error = heading_error_to_polyline(path, ego_heading, idx)
        
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
