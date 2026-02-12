import gymnasium
import numpy as np

from gymnasium_driving.components.obstacles import Circle


class RandomPathObstacles(gymnasium.Wrapper):
    def __init__(
        self,
        env,
        num_obstacles=1,
        min_radius=0.5,
        max_radius=1.0,
        lateral_offset=2.0,
        exclude_start_distance=1.0,
        exclude_goal_distance=1.0,
        seed=None,
    ):
        super().__init__(env)

        self.num_obstacles = num_obstacles
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.lateral_offset = lateral_offset
        self.exclude_start = exclude_start_distance
        self.exclude_goal = exclude_goal_distance
        self.rng = np.random.default_rng(seed)

    def reset(self, **kwargs):
        # Reseed if provided
        seed = kwargs.get("seed")
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # NOTE: to ensure spawn position is set
        obs = super().reset(**kwargs)

        path = self.unwrapped.path
        if path is None or len(path) < 2:
            self.unwrapped.obstacles = []
            return obs
        path_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
        cumulative = np.concatenate([[0], np.cumsum(path_lengths)])
        total_length = cumulative[-1]

        spawn_pos = self.unwrapped.spawn_pos

        # Find the closest point on the path to the spawn position
        distances = np.linalg.norm(path - spawn_pos, axis=1)
        spawn_idx = np.argmin(distances)
        spawn_distance_along_path = cumulative[spawn_idx]

        # Obstacles should only be spawned ahead of the car
        valid_start = spawn_distance_along_path + self.exclude_start
        valid_end = total_length - self.exclude_goal

        obstacles = []

        def sample_position() -> np.ndarray:
            if valid_start < valid_end:
                t = self.rng.uniform(valid_start, valid_end)
                idx = np.searchsorted(cumulative, t) - 1
                local_t = (t - cumulative[idx]) / path_lengths[idx]
                base = path[idx] + local_t * (path[idx + 1] - path[idx])
                tangent = path[idx + 1] - path[idx]
            else:
                idx = int(self.rng.integers(0, len(path) - 1))
                base = path[idx]
                tangent = path[idx + 1] - path[idx]

            norm = np.linalg.norm(tangent)
            if norm > 1e-6:
                tangent = tangent / norm
                normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
                max_offset = self.lateral_offset
                road_network = getattr(self.unwrapped, "road_network", None)
                if road_network is not None and len(road_network.roads) > 0:
                    half_width = road_network.roads[0].half_width
                    max_offset = min(max_offset, half_width * 0.8)
                offset = self.rng.uniform(-max_offset, max_offset)
                base = base + normal * offset
            return base

        goal_pos = np.array(self.unwrapped.goal_pos, dtype=np.float32)
        for _ in range(self.num_obstacles):
            placed = False
            for _ in range(50):
                pos = sample_position()
                if np.linalg.norm(pos - spawn_pos) < self.exclude_start:
                    continue
                if np.linalg.norm(pos - goal_pos) < self.exclude_goal:
                    continue
                radius = self.rng.uniform(self.min_radius, self.max_radius)
                obstacles.append(Circle(center=tuple(pos), radius=radius))
                placed = True
                break
            if not placed:
                radius = self.rng.uniform(self.min_radius, self.max_radius)
                obstacles.append(
                    Circle(center=tuple(sample_position()), radius=radius)
                )

        self.unwrapped.obstacles = obstacles

        return obs


class RandomRoadNetwork(gymnasium.Wrapper):
    """
    Randomize the road network each episode for domain generalization.
    """

    def __init__(
        self,
        env,
        center=(50.0, 50.0),
        length_range=(50.0, 90.0),
        height_range=(30.0, 70.0),
        turn_radius_range=(6.0, 14.0),
        width_range=(6.0, 10.0),
        enforce_road=True,
        solid_road_borders=True,
        seed=None,
    ):
        super().__init__(env)

        self.center = np.array(center, dtype=np.float32)
        self.length_range = length_range
        self.height_range = height_range
        self.turn_radius_range = turn_radius_range
        self.width_range = width_range
        self.enforce_road = enforce_road
        self.solid_road_borders = solid_road_borders
        self.rng = np.random.default_rng(seed)

    def _sample_track(self):
        import gymnasium_driving

        for _ in range(25):
            length = self.rng.uniform(*self.length_range)
            height = self.rng.uniform(*self.height_range)
            turn_radius = self.rng.uniform(*self.turn_radius_range)
            if length > 2 * turn_radius + 2.0 and height > 2 * turn_radius + 2.0:
                break
        width = self.rng.uniform(*self.width_range)

        road = gymnasium_driving.components.roads.create_rectangular_track(
            center=tuple(self.center),
            length=float(length),
            height=float(height),
            turn_radius=float(turn_radius),
            width=float(width),
        )

        road_network = gymnasium_driving.components.roads.RoadNetwork(
            roads=[road],
            enforce_road=self.enforce_road,
            solid_road_borders=self.solid_road_borders,
        )

        return road_network

    def reset(self, **kwargs):
        # Reseed if provided
        seed = kwargs.get("seed")
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        road_network = self._sample_track()
        self.unwrapped.road_network = road_network

        # Update spawn to road start
        start_pos, start_heading = road_network.roads[0].start_pose
        self.unwrapped.spawn_pos = start_pos.astype(np.float32)
        self.unwrapped.spawn_heading = float(start_heading)

        self.unwrapped._compute_global_path()
        self.unwrapped.refresh_world_bounds()

        return super().reset(**kwargs)


class RandomGoal(gymnasium.Wrapper):
    """
    Randomize the goal position along the path, optionally ensuring it is ahead of spawn.
    """

    def __init__(
        self,
        env,
        min_progress: float = 0.25,
        max_progress: float = 0.85,
        goal_radius: float = 2.5,
        seed=None,
    ):
        super().__init__(env)

        self.min_progress = min_progress
        self.max_progress = max_progress
        self.goal_radius = goal_radius
        self.rng = np.random.default_rng(seed)

    def reset(self, **kwargs):
        # Reseed if provided
        seed = kwargs.get("seed")
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        obs = super().reset(**kwargs)

        path = getattr(self.unwrapped, "path", None)

        if path is not None and len(path) >= 2:
            min_idx = int(len(path) * self.min_progress)
            max_idx = int(len(path) * self.max_progress)
            max_idx = max(min_idx + 1, max_idx)
            idx = int(self.rng.integers(min_idx, max_idx))
            self.unwrapped.goal_pos = path[idx].copy()
            self.unwrapped.goal_radius = float(self.goal_radius)

        return obs


class RandomSpawn(gymnasium.Wrapper):
    """
    Randomize where on the path the car starts each episode.

    This forces the agent to learn path-following from *any* position
    on the track rather than memorizing a single starting point.
    """

    def __init__(
        self,
        env,
        # What fraction of the path is eligible as a spawn point.
        # 0.0-1.0 — e.g. 0.5 means the first half of the path.
        path_fraction: float = 0.25,
        lateral_noise: float = 1.0,
        heading_noise: float = 0.15,
        seed=None,
    ):
        super().__init__(env)

        self.path_fraction = path_fraction
        self.lateral_noise = lateral_noise
        self.heading_noise = heading_noise

        self.rng = np.random.default_rng(seed)

    def reset(self, **kwargs):
        # Reseed if provided
        seed = kwargs.get("seed")
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        path = self.unwrapped.path

        if path is not None and len(path) >= 2:
            max_idx = max(1, int(len(path) * self.path_fraction))
            idx = self.rng.integers(0, max_idx)

            spawn_pos = path[idx].copy()

            if idx < len(path) - 1:
                tangent = path[idx + 1] - path[idx]
            else:
                tangent = path[idx] - path[idx - 1]
            heading = float(np.arctan2(tangent[1], tangent[0]))

            # NOTE: lateral noise (perpendicular to path)
            normal = np.array([-tangent[1], tangent[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-6:
                normal = normal / norm
            spawn_pos += normal * self.rng.uniform(
                -self.lateral_noise, self.lateral_noise
            )

            # NOTE: heading noise
            heading += self.rng.uniform(-self.heading_noise, self.heading_noise)

            self.unwrapped.spawn_pos = spawn_pos.astype(np.float32)
            self.unwrapped.spawn_heading = heading

        return super().reset(**kwargs)
