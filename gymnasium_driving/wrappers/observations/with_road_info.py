import gymnasium

import numpy as np

from gymnasium_driving.helpers import curvature_windowed
from gymnasium_driving.helpers import wrap_to_pi, closest_polyline_index, polyline_tangent, signed_cte_to_polyline, heading_error_to_polyline

class WithRoadInfo(gymnasium.ObservationWrapper):
    """
    Adds road and lane information to observations.
    
    Provides:
    - Distance to lane center (cross-track error)
    - Distance to left and right lane boundaries
    - Lane heading relative to ego heading
    - Road curvature at current position
    - Whether vehicle is off-road
    """
    
    def __init__(
        self,
        environment: gymnasium.Env,
    ):
        super().__init__(environment)
        
        self.env = environment
        
        new_spaces = dict(self.observation_space.spaces)
        
        # NOTE: [cross_track_error, dist_left, dist_right, heading_error, curvature, is_off_road]
        new_spaces["road/info"] = gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6,),
            dtype=np.float32,
        )
        
        self.observation_space = gymnasium.spaces.Dict(new_spaces)
    
    def observation(self, observation: dict) -> dict:
        ego_pos = np.array([self.env.unwrapped.state["x"], self.env.unwrapped.state["y"]], dtype=np.float32)
        ego_heading = self.env.unwrapped.state["yaw"]
        
        road_info = np.zeros(6, dtype=np.float32)
        
        if self.env.unwrapped.road_network is None or len(self.env.unwrapped.road_network.roads) == 0:
            observation["road/info"] = road_info
            return observation
        
        # TODO: handle multiple roads, handle out of road case
        road = self.env.unwrapped.road_network.roads[0]
        half_width = road.half_width
        
        centerline = self._get_road_centerline(road)
        if len(centerline) == 0:
            observation["road/info"] = road_info
            return observation
        
        closest_idx = closest_polyline_index(centerline, ego_pos)
        cross_track_error, closest_idx = signed_cte_to_polyline(centerline, ego_pos, idx=closest_idx)
        heading_error = heading_error_to_polyline(centerline, ego_heading, closest_idx)
        
        dist_left = half_width - cross_track_error
        dist_right = half_width + cross_track_error
        
        curvature = curvature_windowed(centerline, closest_idx, window=5)
        
        is_off_road = 1.0 if self.env.unwrapped.road_network.is_off_road(ego_pos) else 0.0
        
        road_info = np.array([
            cross_track_error,
            dist_left,
            dist_right,
            heading_error,
            curvature,
            is_off_road,
        ], dtype=np.float32)
        
        observation["road/info"] = road_info
        
        return observation
    
    def _get_road_centerline(self, road, num_points: int = 100) -> np.ndarray:
        all_points = []
        for segment in road.segments:
            points = segment.get_centerline_points(num_points // len(road.segments))
            all_points.extend(points)
        return np.array(all_points, dtype=np.float32)
