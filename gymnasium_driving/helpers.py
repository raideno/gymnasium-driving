import numpy as np

def curvature_windowed(path: np.ndarray, idx: int, window: int = 3) -> float:
    if len(path) < 3 or idx < 1 or idx >= len(path) - 1:
        return 0.0

    start = max(0, idx - window)
    end = min(len(path), idx + window + 1)
    if end - start < 3:
        return 0.0

    p1 = path[start]
    p2 = path[idx]
    p3 = path[min(end - 1, len(path) - 1)]

    area = 0.5 * abs(
        (p2[0] - p1[0]) * (p3[1] - p1[1])
        - (p3[0] - p1[0]) * (p2[1] - p1[1])
    )

    d1 = np.linalg.norm(p2 - p1)
    d2 = np.linalg.norm(p3 - p2)
    d3 = np.linalg.norm(p3 - p1)

    denom = d1 * d2 * d3
    if denom < 1e-6:
        return 0.0

    return float(4.0 * area / denom)

def wrap_to_pi(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return float(np.arctan2(np.sin(angle), np.cos(angle)))

def closest_polyline_index(polyline_xy: np.ndarray, point_xy: np.ndarray) -> int:
    """
    polyline_xy: (N, 2)
    point_xy: (2,)
    """
    dists = np.linalg.norm(polyline_xy - point_xy, axis=1)
    return int(np.argmin(dists))

def polyline_tangent(polyline_xy: np.ndarray, idx: int) -> np.ndarray:
    """
    Returns a (2,) tangent vector around polyline index idx.
    Uses forward diff except at the end, where it uses backward diff.
    """
    n = len(polyline_xy)
    if n < 2:
        return np.array([1.0, 0.0], dtype=np.float32)

    if idx <= 0:
        t = polyline_xy[1] - polyline_xy[0]
    elif idx >= n - 1:
        t = polyline_xy[-1] - polyline_xy[-2]
    else:
        # central-ish difference tends to be smoother than pure forward diff
        t = polyline_xy[idx + 1] - polyline_xy[idx - 1]

    norm = np.linalg.norm(t)
    if norm < 1e-8:
        return np.array([1.0, 0.0], dtype=np.float32)
    return (t / norm).astype(np.float32)

def signed_cte_to_polyline(
    polyline_xy: np.ndarray,
    ego_xy: np.ndarray,
    idx: int | None = None,
) -> tuple[float, int]:
    """
    Signed cross-track error relative to polyline tangent at closest point.
    Positive means ego is to the left of the tangent direction.
    Returns: (cte, idx_used)
    """
    if polyline_xy is None or len(polyline_xy) < 2:
        return 0.0, 0

    if idx is None:
        idx = closest_polyline_index(polyline_xy, ego_xy)

    closest_pt = polyline_xy[idx]
    t_hat = polyline_tangent(polyline_xy, idx)  # unit tangent
    v = ego_xy - closest_pt

    # 2D cross product z-component: t_hat.x * v.y - t_hat.y * v.x
    cte = float(t_hat[0] * v[1] - t_hat[1] * v[0])
    return cte, idx

def heading_error_to_polyline(
    polyline_xy: np.ndarray,
    ego_yaw: float,
    idx: int,
) -> float:
    """Heading error (ego - path_heading) wrapped to [-pi, pi]."""
    t_hat = polyline_tangent(polyline_xy, idx)
    path_heading = float(np.arctan2(t_hat[1], t_hat[0]))
    return wrap_to_pi(ego_yaw - path_heading)
