"""
Observation wrapper that adds path following information to observations.

Provides waypoint lookahead, cross-track error, and path curvature information
essential for path following controllers and RL agents.
"""

import typing
import numpy as np
import gymnasium


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
        
        self._path_length = 0.0
    
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
        distances = np.linalg.norm(path - ego_pos, axis=1)
        closest_idx = np.argmin(distances)
        closest_point = path[closest_idx]
        
        # NOTE: signed cross-track error
        if closest_idx < len(path) - 1:
            path_direction = path[closest_idx + 1] - path[closest_idx]
        elif closest_idx > 0:
            path_direction = path[closest_idx] - path[closest_idx - 1]
        else:
            path_direction = np.array([1.0, 0.0])
        path_heading = np.arctan2(path_direction[1], path_direction[0])
        to_ego = ego_pos - closest_point
        # Manual 2D cross product to avoid numpy shape issues
        cross = float(path_direction[0] * to_ego[1] - path_direction[1] * to_ego[0])
        cte = cross / (np.linalg.norm(path_direction) + 1e-6)
        
        # NOTE: heading error
        heading_error = ego_heading - path_heading
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        
        # NOTE: normalized progress along the path
        progress = closest_idx / max(len(path) - 1, 1)
        
        # NOTE: goal distance
        goal_distance = np.linalg.norm(ego_pos - self.env.unwrapped.goal_pos)
        
        path_info = np.array([cte, heading_error, progress, goal_distance], dtype=np.float32)
        
        # NOTE: waypoints ahead
        cos_h, sin_h = np.cos(-ego_heading), np.sin(-ego_heading)
        rotation = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
        
        for i in range(self.num_waypoints):
            waypoint_idx = closest_idx + i + 1
            
            if waypoint_idx >= len(path):
                # Pad with last waypoint
                wp = path[-1]
                current_idx = len(path) - 1
            else:
                wp = path[waypoint_idx]
                current_idx = waypoint_idx
            
            # Transform to ego frame
            rel_pos = rotation @ (wp - ego_pos)
            
            curvature = self._compute_curvature_at_index(path, current_idx)
            waypoints[i] = [rel_pos[0], rel_pos[1], curvature]
        
        observation["path/waypoints"] = waypoints
        observation["path/info"] = path_info
        
        return observation
    
    def _compute_curvature_at_index(self, path: np.ndarray, idx: int, window: int = 3) -> float:
        if len(path) < 3 or idx < 1 or idx >= len(path) - 1:
            return 0.0
        
        start = max(0, idx - window)
        end = min(len(path), idx + window + 1)
        
        if end - start < 3:
            return 0.0
        
        p1 = path[start]
        p2 = path[idx]
        p3 = path[min(end - 1, len(path) - 1)]
        
        # Three-point curvature formula
        area = 0.5 * abs(
            (p2[0] - p1[0]) * (p3[1] - p1[1]) - 
            (p3[0] - p1[0]) * (p2[1] - p1[1])
        )
        
        d1 = np.linalg.norm(p2 - p1)
        d2 = np.linalg.norm(p3 - p2)
        d3 = np.linalg.norm(p3 - p1)
        
        denom = d1 * d2 * d3
        if denom < 1e-6:
            return 0.0
        
        return 4 * area / denom
