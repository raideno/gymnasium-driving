import gymnasium
import numpy

from gymnasium_driving.helpers import (
    closest_polyline_index,
    curvature_windowed,
    heading_error_to_polyline,
    signed_cte_to_polyline,
)


class WithRoadInfo(gymnasium.ObservationWrapper):
    """
    A gymnasium.ObservationWrapper that augments the observation dictionary
    with basic road and lane-related information derived from the current
    vehicle state and the active road.

    Added observation keys:

    - "road/info" (Box, shape=(2,))
        Contains:
            [distance_to_left_boundary, distance_to_right_boundary]

        Distances are computed from the vehicle position relative to the
        road centerline using the signed cross-track error. Positive and
        negative signs depend on the underlying signed CTE convention.

    All values are returned as numpy.float32.

    Requirements:
        The wrapped environment must expose:

        - env.unwrapped.state with keys:
            - "x"
            - "y"

        - env.unwrapped.road_network with:
            - attribute "roads" (list of road objects)

        Each road object must provide:
            - attribute "half_width"
            - attribute "segments"

        Each segment must implement:
            - get_centerline_points(num_points: int) -> array-like

        The following helper functions must be available:
            - closest_polyline_index
            - signed_cte_to_polyline
    """

    def __init__(
        self,
        environment: gymnasium.Env,
    ):
        super().__init__(environment)

        new_spaces = dict(self.observation_space.spaces)

        # NOTE: [dist_left, dist_right]
        new_spaces["road/info"] = gymnasium.spaces.Box(
            low=-numpy.inf,
            high=numpy.inf,
            shape=(2,),
            dtype=numpy.float32,
        )

        self.observation_space = gymnasium.spaces.Dict(new_spaces)

    def observation(self, observation: dict) -> dict:
        ego_pos = numpy.array(
            [
                self.env.unwrapped.state["x"],
                self.env.unwrapped.state["y"],
            ],
            dtype=numpy.float32,
        )

        road_info = numpy.zeros(2, dtype=numpy.float32)

        if (
            self.env.unwrapped.road_network is None
            or len(self.env.unwrapped.road_network.roads) == 0
        ):
            observation["road/info"] = road_info
            return observation

        # todo: handle multiple roads, handle out of road case
        road = self.env.unwrapped.road_network.roads[0]
        half_width = road.half_width

        centerline = self._get_road_centerline(road)
        if len(centerline) == 0:
            observation["road/info"] = road_info
            return observation

        cross_track_error, closest_idx = signed_cte_to_polyline(
            centerline,
            ego_pos,
            idx=closest_polyline_index(centerline, ego_pos),
        )

        dist_left = half_width - cross_track_error
        dist_right = half_width + cross_track_error

        road_info = numpy.array(
            [
                dist_left,
                dist_right,
            ],
            dtype=numpy.float32,
        )

        observation["road/info"] = road_info

        return observation

    def _get_road_centerline(
        self,
        road,
        num_points: int = 100,
    ) -> numpy.ndarray:
        all_points = []
        for segment in road.segments:
            all_points.extend(
                segment.get_centerline_points(num_points // len(road.segments))
            )
        return numpy.array(all_points, dtype=numpy.float32)
