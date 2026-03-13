import numpy as np

import gymnasium_driving
import gymnasium_driving.environment


def make_empty_obstacles_factory() -> gymnasium_driving.environment.ObstaclesFactory:
    def factory(
        env: "gymnasium_driving.environment.CarEnvironment",
    ) -> gymnasium_driving.environment.ObstacleList:
        return []

    return factory


def make_random_obstacles_factory(
    num_obstacles: int,
    min_spacing_m: float,
) -> gymnasium_driving.environment.ObstaclesFactory:
    """
    Returns an obstacles factory that places circular obstacles along the
    episode path.

    Parameters
    ----------
    num_obstacles:
        How many obstacles to place.
    min_spacing_m:
        Minimum arc-length distance (meters) between any two obstacles.

    Fixed internals
    ---------------
    radius          : 0.9 m
    lateral_offset  : abs(Normal(1.0, 0.3)) m, side sampled randomly
    exclude_start   : 8.0 m from spawn point
    exclude_end     : 5.0 m from goal point
    """

    _RADIUS = 1.25
    _LATERAL_MEAN = 0
    _LATERAL_STD = 0.0
    _EXCLUDE_START_M = 10
    _EXCLUDE_END_M = 10

    def _build_arc_lengths(path: np.ndarray) -> np.ndarray:
        deltas = np.linalg.norm(np.diff(path, axis=0), axis=1)
        return np.concatenate([[0.0], np.cumsum(deltas)]).astype(np.float32)

    def _unit_normal_at(path: np.ndarray, i: int) -> np.ndarray:
        n = len(path)
        i0 = max(0, i - 1)
        i1 = min(n - 1, i + 1)
        t = path[i1] - path[i0]
        norm = float(np.linalg.norm(t))
        if norm < 1e-8:
            return np.array([0.0, 1.0], dtype=np.float32)
        t = t / norm
        return np.array([-t[1], t[0]], dtype=np.float32)

    def _sample_spaced_indices(
        rng: np.random.Generator,
        arc_lengths: np.ndarray,
        lo_s: float,
        hi_s: float,
        k: int,
    ) -> list:
        valid_mask = (arc_lengths >= lo_s) & (arc_lengths <= hi_s)
        candidates = np.where(valid_mask)[0]
        rng.shuffle(candidates)
        chosen = []
        for idx in candidates:
            s = arc_lengths[idx]
            if all(abs(s - arc_lengths[c]) >= min_spacing_m for c in chosen):
                chosen.append(int(idx))
            if len(chosen) == k:
                break
        return chosen

    def factory(
        env: "gymnasium_driving.environment.CarEnvironment",
    ) -> gymnasium_driving.environment.ObstacleList:
        path = getattr(env, "path", None)
        if path is None or len(path) < 2:
            return []

        path = np.asarray(path, dtype=np.float32)
        arc_lengths = _build_arc_lengths(path)
        total_length = float(arc_lengths[-1])
        rng = env.np_random

        lo_s = _EXCLUDE_START_M
        hi_s = total_length - _EXCLUDE_END_M
        if hi_s <= lo_s:
            return []

        chosen_indices = _sample_spaced_indices(
            rng, arc_lengths, lo_s, hi_s, k=num_obstacles
        )

        obstacles: gymnasium_driving.environment.ObstacleList = []

        for path_idx in chosen_indices:
            p = path[path_idx]
            nrm = _unit_normal_at(path, path_idx)

            d = abs(float(rng.normal(_LATERAL_MEAN, _LATERAL_STD)))
            sign = float(rng.choice([-1.0, 1.0]))
            center = (p + sign * d * nrm).astype(np.float32)

            obstacles.append(
                gymnasium_driving.environment.Circle(
                    center=(float(center[0]), float(center[1])),
                    radius=_RADIUS,
                )
            )

        return obstacles

    return factory
