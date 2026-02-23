import gymnasium as gym
import numpy as np
from gymnasium_driving.components.obstacles import Circle

class RandomWaypointObstacles(gym.Wrapper):
    def __init__(
        self,
        environment,
        *,
        num_obstacles: int = 3,
        min_radius: float = 0.6,
        max_radius: float = 1.2,
        exclude_start_k: int = 3,
        exclude_end_k: int = 3,
        min_lateral_offset: float = 0.0,
        max_lateral_offset: float = 2.0,
        seed: int | None = None,
    ):
        super().__init__(environment)
        self.env = environment
        self.num_obstacles = int(num_obstacles)
        self.min_radius = float(min_radius)
        self.max_radius = float(max_radius)
        self.exclude_start_k = int(exclude_start_k)
        self.exclude_end_k = int(exclude_end_k)

        self.min_lateral_offset = float(min_lateral_offset)
        self.max_lateral_offset = float(max_lateral_offset)

        self.rng = np.random.default_rng(seed)

    def _normal_at(self, path: np.ndarray, i: int) -> np.ndarray:
        """Unit normal at path index i, based on local tangent."""
        n = len(path)
        i0 = max(0, i - 1)
        i1 = min(n - 1, i + 1)
        t = path[i1] - path[i0]  # tangent-ish
        norm = float(np.linalg.norm(t))
        if norm < 1e-8:
            return np.array([0.0, 0.0], dtype=np.float32)
        t = t / norm
        # rotate 90 degrees to get normal (left-hand normal)
        return np.array([-t[1], t[0]], dtype=np.float32)

    def reset(self, **kwargs):
        seed = kwargs.get("seed")
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        obs = super().reset(**kwargs)

        path = getattr(self.unwrapped, "path", None)
        if path is None or len(path) < 2:
            self.unwrapped.obstacles = []
            return obs

        path = np.asarray(path, dtype=np.float32)
        n_pts = len(path)

        lo = self.exclude_start_k
        hi = n_pts - self.exclude_end_k
        if hi <= lo:
            self.unwrapped.obstacles = []
            return obs

        k = min(self.num_obstacles, hi - lo)
        idx = self.rng.choice(np.arange(lo, hi), size=k, replace=False)

        obstacles = []
        for i in idx:
            r = float(self.rng.uniform(self.min_radius, self.max_radius))

            p = path[int(i)]
            nrm = self._normal_at(path, int(i))

            # random signed lateral displacement
            if self.max_lateral_offset > 0:
                d = float(
                    self.rng.uniform(self.min_lateral_offset, self.max_lateral_offset)
                )
                sign = float(self.rng.choice([-1.0, 1.0]))
                offset = sign * d * nrm
            else:
                offset = np.array([0.0, 0.0], dtype=np.float32)

            c = p + offset
            obstacles.append(
                Circle(center=(float(c[0]), float(c[1])), radius=r)
            )

        self.unwrapped.obstacles = obstacles
        return obs