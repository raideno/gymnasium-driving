import typing

import gymnasium
import numpy

import gymnasium_driving
import gymnasium_driving.components
import gymnasium_driving.components.obstacles


class WithObstaclesInfo(gymnasium.ObservationWrapper):
    """
    A gymnasium.ObservationWrapper that augments the
    gymnasium_driving.environment.CarEnvironment observation dictionary
    with obstacle information detected within a configurable range.

    Added observation keys:

    - "obstacles/instances" (Box, shape=(max_obstacles, 5))
        For each obstacle (sorted by distance, closest first), provides:

            [exists, rel_x, rel_y, distance, size]

        where:
            - exists: 1.0 if a valid obstacle entry, 0.0 if padded
            - rel_x: relative x-position in ego frame (meters)
            - rel_y: relative y-position in ego frame (meters)
            - distance: Euclidean distance in world frame (meters)
            - size: obstacle radius or half-extent approximation

        If fewer than max_obstacles are detected, remaining rows are
        zero-padded.

    - "obstacles/num_obstacles_detected" (Box, shape=(1,))
        Number of obstacles detected within detection_range
        (clipped to max_obstacles).

    All values are returned as numpy.float32.

    Detection behavior:
        - Only obstacles within detection_range (meters) are included.
        - Obstacles are sorted by increasing distance.
        - Positions are expressed in the ego-vehicle frame by default.

    Args:
        environment: The environment to wrap.
        detection_range: Maximum detection distance in meters.
        max_obstacles: Maximum number of obstacles included in the
            observation.

    Requirements:
        The wrapped environment must expose:

        - env.unwrapped.state with keys:
            - "x"
            - "y"
            - "yaw"

        - env.unwrapped.obstacles:
            Iterable of obstacle objects with:
                - obstacle.center -> (x, y)
                - For Circle: obstacle.radius
                - For Rectangle: obstacle.width, obstacle.height
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
        self.obstacle_instance_dimension = 5

        new_spaces = dict(self.observation_space.spaces)
        # [exists, rel_x, rel_y, distance, radius/size]
        new_spaces["obstacles/instances"] = gymnasium.spaces.Box(
            low=-numpy.inf,
            high=numpy.inf,
            shape=(self.max_obstacles, self.obstacle_instance_dimension),
            dtype=numpy.float32,
        )
        new_spaces["obstacles/num_obstacles_detected"] = gymnasium.spaces.Box(
            low=0,
            high=self.max_obstacles,
            shape=(1,),
            dtype=numpy.float32,
        )
        self.observation_space = gymnasium.spaces.Dict(new_spaces)

        self._prev_obstacle_positions: typing.Dict[int, numpy.ndarray] = {}

    def observation(self, observation: dict) -> dict:
        ego_position = numpy.array(
            [self.env.unwrapped.state["x"], self.env.unwrapped.state["y"]],
            dtype=numpy.float32,
        )
        ego_heading = self.env.unwrapped.state["yaw"]

        # NOTE: rotation matrix to transform to ego frame
        cos_h, sin_h = numpy.cos(-ego_heading), numpy.sin(-ego_heading)
        rotation_matrix = numpy.array([[cos_h, -sin_h], [sin_h, cos_h]])

        obstacle_data = []

        for i, obstacle in enumerate(self.env.unwrapped.obstacles):
            rel_pos_world = (
                numpy.array(obstacle.center, dtype=numpy.float32) - ego_position
            )
            distance = numpy.linalg.norm(rel_pos_world)

            if distance > self.detection_range:
                continue

            # NOTE: transform to ego frame if requested
            rel_pos = (
                rotation_matrix @ rel_pos_world if self.ego_frame else rel_pos_world
            )

            # NOTE: get obstacle size
            if isinstance(obstacle, gymnasium_driving.components.obstacles.Circle):
                size = obstacle.radius
            elif isinstance(obstacle, gymnasium_driving.components.obstacles.Rectangle):
                size = max(obstacle.width, obstacle.height) / 2
            else:
                size = 1.0

            obstacle_data.append(
                [
                    1.0,
                    rel_pos[0],
                    rel_pos[1],
                    distance,
                    size,
                ]
            )

        # NOTE: sort by distance and take closest
        obstacle_data.sort(key=lambda x: x[3])
        obstacle_data = obstacle_data[: self.max_obstacles]
        detected_count = len(obstacle_data)

        # NOTE: pad with zeros if fewer than max_obstacles
        while len(obstacle_data) < self.max_obstacles:
            obstacle_data.append([0.0] * self.obstacle_instance_dimension)

        observation["obstacles/instances"] = numpy.array(
            obstacle_data, dtype=numpy.float32
        )
        observation["obstacles/num_obstacles_detected"] = numpy.array(
            [detected_count], dtype=numpy.float32
        )

        return observation

    def reset(self, **kwargs):
        self._prev_obstacle_positions.clear()
        return super().reset(**kwargs)
