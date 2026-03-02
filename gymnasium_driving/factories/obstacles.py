import numpy as np

import gymnasium_driving
import gymnasium_driving.environment


def make_empty_obstacles_factory() -> gymnasium_driving.environment.ObstaclesFactory:
    """
    Returns an obstacles factory that places no obstacles.
    """

    def factory(
        env: "gymnasium_driving.environment.CarEnvironment",
    ) -> gymnasium_driving.environment.ObstacleList:
        return []

    return factory


def make_random_obstacles_factory(
    num_obstacles: int = 3,
    min_radius: float = 0.6,
    max_radius: float = 1.2,
    exclude_start_k: int = 5,
    exclude_end_k: int = 3,
    min_lateral_offset: float = 0.3,
    max_lateral_offset: float = 1.8,
    min_spacing_m: float = 3.0,
    spawn_clearance: float = 4.0,
    navigability_check: bool = True,
    max_attempts_per_obstacle: int = 8,
) -> gymnasium_driving.environment.ObstaclesFactory:
    """
    Returns an obstacles factory that places circular obstacles along the
    episode path.

    Parameters
    ----------
    num_obstacles:
        How many obstacles to attempt to place.
    min_radius / max_radius:
        Radius range (meters) for each circular obstacle.
    exclude_start_k / exclude_end_k:
        Number of path indices to skip at the start and end of the path.
        Increase exclude_start_k to widen the spawn-area buffer.
    min_lateral_offset / max_lateral_offset:
        Lateral displacement range (meters) from the path centerline.
        A non-zero min_lateral_offset biases obstacles off-center so the
        path is easier to navigate.
    min_spacing_m:
        Minimum arc-length distance (meters) between any two obstacles.
    spawn_clearance:
        Additional hard exclusion radius (meters) around the spawn position.
        Any candidate whose circle overlaps this zone is rejected.
    navigability_check:
        When True, a candidate is rejected if the obstacle center falls
        directly on the path (i.e. the lateral offset after clamping is
        smaller than min_lateral_offset + radius), preventing a full block.
    max_attempts_per_obstacle:
        Number of random lateral-side attempts per chosen path index before
        giving up on that index.
    """

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
        lo: int,
        hi: int,
        k: int,
    ) -> list:
        candidates = np.arange(lo, hi)
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
        n_pts = len(path)
        arc_lengths = _build_arc_lengths(path)
        rng = env.np_random  # use the env's seeded RNG for reproducibility

        lo = exclude_start_k
        hi = n_pts - exclude_end_k
        if hi <= lo:
            return []

        chosen_indices = _sample_spaced_indices(
            rng, arc_lengths, lo, hi, k=num_obstacles
        )

        spawn_pos = np.asarray(env.spawn_pos, dtype=np.float32)

        obstacles: gymnasium_driving.environment.ObstacleList = []

        for path_idx in chosen_indices:
            p = path[path_idx]
            nrm = _unit_normal_at(path, path_idx)

            placed = False
            for _ in range(max_attempts_per_obstacle):
                r = float(rng.uniform(min_radius, max_radius))
                d = float(rng.uniform(min_lateral_offset, max_lateral_offset))
                sign = float(rng.choice([-1.0, 1.0]))
                center = (p + sign * d * nrm).astype(np.float32)

                # reject if too close to the spawn position
                if float(np.linalg.norm(center - spawn_pos)) < spawn_clearance + r:
                    continue

                # reject if the circle blocks the path (center too close to
                # the centerline to leave a navigable gap on either side)
                if navigability_check and d < min_lateral_offset + r:
                    continue

                obstacles.append(
                    gymnasium_driving.environment.Circle(
                        center=(float(center[0]), float(center[1])), radius=r
                    )
                )
                placed = True
                break

            # if no valid placement was found for this index, skip silently

        return obstacles

    return factory
