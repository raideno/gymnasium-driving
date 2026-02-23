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
        min_spacing_m: float = 2.0,
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
        self.min_spacing_m = float(min_spacing_m)
        self.rng = np.random.default_rng(seed)

    def _normal_at(self, path: np.ndarray, i: int) -> np.ndarray:
        """Unit normal at path index i, based on local tangent."""
        n = len(path)
        i0 = max(0, i - 1)
        i1 = min(n - 1, i + 1)
        t = path[i1] - path[i0]
        norm = float(np.linalg.norm(t))
        if norm < 1e-8:
            return np.array([0.0, 0.0], dtype=np.float32)
        t = t / norm
        return np.array([-t[1], t[0]], dtype=np.float32)

    def _build_arc_lengths(self, path: np.ndarray) -> np.ndarray:
        """Cumulative arc-length in meters at each path index."""
        deltas = np.linalg.norm(np.diff(path, axis=0), axis=1)
        return np.concatenate([[0.0], np.cumsum(deltas)]).astype(np.float32)

    def _sample_spaced_indices(
        self,
        arc_lengths: np.ndarray,
        lo: int,
        hi: int,
        k: int,
    ) -> list[int]:
        """
        Sample up to k indices from [lo, hi) such that no two chosen
        indices are closer than self.min_spacing_m along the path.

        Uses a random sequential approach: shuffle the candidates and
        greedily accept those that respect the spacing constraint.
        """
        candidates = np.arange(lo, hi)
        self.rng.shuffle(candidates)

        chosen = []
        for idx in candidates:
            s = arc_lengths[idx]
            if all(abs(s - arc_lengths[c]) >= self.min_spacing_m for c in chosen):
                chosen.append(int(idx))
            if len(chosen) == k:
                break

        return chosen

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
        arc_lengths = self._build_arc_lengths(path)

        lo = self.exclude_start_k
        hi = n_pts - self.exclude_end_k
        if hi <= lo:
            self.unwrapped.obstacles = []
            return obs

        chosen_indices = self._sample_spaced_indices(
            arc_lengths, lo, hi, k=self.num_obstacles
        )

        obstacles = []
        for i in chosen_indices:
            r = float(self.rng.uniform(self.min_radius, self.max_radius))
            p = path[i]
            nrm = self._normal_at(path, i)

            if self.max_lateral_offset > 0:
                d = float(self.rng.uniform(self.min_lateral_offset, self.max_lateral_offset))
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