import typing
import gymnasium

import numpy as np

class WithRoadInfo(gymnasium.ObservationWrapper):
    """
    Adds road and lane information to observations.
    
    Provides:
    - Distance to lane center (cross-track error)
    - Distance to left and right lane boundaries
    - Lane heading relative to ego heading
    - Road curvature at current position
    - Whether vehicle is off-road
    
    Args:
        env: The environment to wrap
        num_boundary_points: Number of boundary sample points to include
        lookahead_distance: Distance ahead to sample road curvature
    """
    
    def __init__(
        self,
        env: gymnasium.Env,
        num_boundary_points: int = 5,
        lookahead_distance: float = 20.0,
    ):
        super().__init__(env)
        
        self.num_boundary_points = num_boundary_points
        self.lookahead_distance = lookahead_distance
        
        # Update observation space
        new_spaces = dict(self.observation_space.spaces)
        
        # Core lane info: [cte, dist_left, dist_right, heading_error, curvature, is_off_road]
        new_spaces["road/info"] = gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6,),
            dtype=np.float32,
        )
        
        # Boundary points for more detailed road shape
        new_spaces["road/left_boundary"] = gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_boundary_points, 2),
            dtype=np.float32,
        )
        new_spaces["road/right_boundary"] = gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_boundary_points, 2),
            dtype=np.float32,
        )
        
        self.observation_space = gymnasium.spaces.Dict(new_spaces)
    
    def observation(self, observation: dict) -> dict:
        state = self.env.unwrapped.state
        ego_pos = state[:2]
        ego_heading = state[2]
        
        road_info = np.zeros(6, dtype=np.float32)
        left_boundary = np.zeros((self.num_boundary_points, 2), dtype=np.float32)
        right_boundary = np.zeros((self.num_boundary_points, 2), dtype=np.float32)
        
        if self.env.unwrapped.road_network is None or len(self.env.unwrapped.road_network.roads) == 0:
            observation["road/info"] = road_info
            observation["road/left_boundary"] = left_boundary
            observation["road/right_boundary"] = right_boundary
            return observation
        
        # Find closest road and segment
        road = self.env.unwrapped.road_network.roads[0]  # Assuming single road for simplicity
        half_width = road.half_width
        
        # Find closest point on centerline
        centerline = self._get_road_centerline(road)
        if len(centerline) == 0:
            observation["road/info"] = road_info
            observation["road/left_boundary"] = left_boundary
            observation["road/right_boundary"] = right_boundary
            return observation
        
        distances = np.linalg.norm(centerline - ego_pos, axis=1)
        closest_idx = np.argmin(distances)
        closest_point = centerline[closest_idx]
        
        # Compute cross-track error (signed)
        if closest_idx < len(centerline) - 1:
            road_direction = centerline[closest_idx + 1] - centerline[closest_idx]
        else:
            road_direction = centerline[closest_idx] - centerline[closest_idx - 1]
        
        road_heading = np.arctan2(road_direction[1], road_direction[0])
        
        # Vector from closest point to ego
        to_ego = ego_pos - closest_point
        
        # Cross-track error: positive if to the left of the centerline
        cross = np.cross(road_direction, to_ego)
        cte = cross / (np.linalg.norm(road_direction) + 1e-6)
        
        # Distances to boundaries
        dist_left = half_width - cte
        dist_right = half_width + cte
        
        # Heading error (normalize to [-pi, pi])
        heading_error = ego_heading - road_heading
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        
        # Estimate curvature using nearby points
        curvature = self._estimate_curvature(centerline, closest_idx)
        
        # Check if off-road
        is_off_road = 1.0 if self.env.unwrapped.road_network.is_off_road(ego_pos) else 0.0
        
        road_info = np.array([
            cte,
            dist_left,
            dist_right,
            heading_error,
            curvature,
            is_off_road,
        ], dtype=np.float32)
        
        # Get boundary points in ego frame
        cos_h, sin_h = np.cos(-ego_heading), np.sin(-ego_heading)
        rotation = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
        
        # Sample points ahead on the road
        for i in range(self.num_boundary_points):
            lookahead_idx = min(
                closest_idx + int((i + 1) * self.lookahead_distance / self.num_boundary_points),
                len(centerline) - 1
            )
            
            center_pt = centerline[lookahead_idx]
            
            # Get road direction at this point
            if lookahead_idx < len(centerline) - 1:
                direction = centerline[lookahead_idx + 1] - center_pt
            else:
                direction = center_pt - centerline[lookahead_idx - 1]
            
            direction = direction / (np.linalg.norm(direction) + 1e-6)
            perpendicular = np.array([-direction[1], direction[0]])
            
            left_pt = center_pt + perpendicular * half_width
            right_pt = center_pt - perpendicular * half_width
            
            # Transform to ego frame
            left_boundary[i] = rotation @ (left_pt - ego_pos)
            right_boundary[i] = rotation @ (right_pt - ego_pos)
        
        observation["road/info"] = road_info
        observation["road/left_boundary"] = left_boundary
        observation["road/right_boundary"] = right_boundary
        
        return observation
    
    def _get_road_centerline(self, road, num_points: int = 100) -> np.ndarray:
        """Get centerline points for a road."""
        all_points = []
        for segment in road.segments:
            points = segment.get_centerline_points(num_points // len(road.segments))
            all_points.extend(points)
        return np.array(all_points, dtype=np.float32)
    
    def _estimate_curvature(self, centerline: np.ndarray, idx: int, window: int = 5) -> float:
        """Estimate road curvature using nearby centerline points."""
        if len(centerline) < 3:
            return 0.0
        
        start_idx = max(0, idx - window)
        end_idx = min(len(centerline), idx + window + 1)
        
        if end_idx - start_idx < 3:
            return 0.0
        
        points = centerline[start_idx:end_idx]
        
        # Compute curvature using three-point formula
        curvatures = []
        for i in range(1, len(points) - 1):
            p1, p2, p3 = points[i-1], points[i], points[i+1]
            
            # Area of triangle
            area = 0.5 * abs(
                (p2[0] - p1[0]) * (p3[1] - p1[1]) - 
                (p3[0] - p1[0]) * (p2[1] - p1[1])
            )
            
            # Side lengths
            d1 = np.linalg.norm(p2 - p1)
            d2 = np.linalg.norm(p3 - p2)
            d3 = np.linalg.norm(p3 - p1)
            
            # Curvature = 4 * area / (d1 * d2 * d3)
            denom = d1 * d2 * d3
            if denom > 1e-6:
                curvatures.append(4 * area / denom)
        
        return np.mean(curvatures) if curvatures else 0.0