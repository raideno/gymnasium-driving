import gymnasium
import numpy as np

from gymnasium_driving.helpers import curvature_windowed


class WithPathInfo(gymnasium.ObservationWrapper):
    """
    A gymnasium.ObservationWrapper that augments the observation dictionary
    with path-following information derived from the environment's reference
    path.

    Added observation keys:

    - "path/waypoints" (Box, shape=(num_waypoints, 3))
        Upcoming waypoints expressed in the ego (vehicle) frame.
        Each waypoint contains:
            [relative_x, relative_y, curvature]

        Waypoints are taken ahead of the closest path index. If the end of
        the path is reached, the last waypoint is repeated for padding.
        Curvature is computed using curvature_windowed.

    - "path/info" (Box, shape=(3,))
        Contains: [cross_track_error, heading_error, normalized_progress]

        - cross_track_error: Lateral distance to the path
        - heading_error: Difference between vehicle heading and path tangent
        - normalized_progress: Closest path index normalized to [0, 1]

    All values are returned as numpy.float32.

    Args:
        environment (gymnasium.Env):
            The environment to wrap.

        num_waypoints (int):
            Number of upcoming waypoints to include in the observation.

    Requirements:
        The wrapped environment must expose:

        - env.unwrapped.state with keys:
            - "x"
            - "y"
            - "yaw"
            - "closest_path_idx"
            - "cte"
            - "heading_error"

        - env.unwrapped.path
            A sequence of 2D waypoints.

        - env.unwrapped.goal_pos
            Goal position as a 2D array-like object.
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

        # path info: [cte, heading_error, progress]
        new_spaces["path/info"] = gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3,),
            dtype=np.float32,
        )

        self.observation_space = gymnasium.spaces.Dict(new_spaces)

    def observation(self, observation: dict) -> dict:
        ego_pos = np.array(
            [self.env.unwrapped.state["x"], self.env.unwrapped.state["y"]],
            dtype=np.float32,
        )
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

        path_info = np.array([cte, heading_error, progress], dtype=np.float32)

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
