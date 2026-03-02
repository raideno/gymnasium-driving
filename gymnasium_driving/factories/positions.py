import typing

import numpy as np

import gymnasium_driving
import gymnasium_driving.environment

# TODO: add a classic deterministic one


def make_centerline_positions_factory(
    proportion: typing.Union[float, typing.Tuple[float, float]] = (0.7, 0.05),
    noise: typing.Tuple[float, float] = (0.15, 0.5),
) -> gymnasium_driving.environment.PositionsFactory:
    """
    Returns a positions factory that replicates the original random spawn/goal
    sampling behaviour.

    proportion: either a fixed float or a (mean, std) tuple controlling how far
                along the centerline the goal is placed relative to the spawn.
    noise:      (heading_std_rad, lateral_offset_std_meters) applied to the
                sampled spawn pose.
    """
    if isinstance(proportion, (int, float)):
        proportion_mean = float(proportion)
        proportion_var = 0.0
        fixed_proportion = True
    else:
        proportion_mean, proportion_var = proportion
        fixed_proportion = False

    heading_noise, cte_noise = noise

    def _build_centerline(
        env: "gymnasium_driving.environment.CarEnvironment",
    ) -> np.ndarray:
        all_points = []
        for road in env.road_network.roads:
            for segment in road.segments:
                num_points = max(50, int(segment.get_length() * 2))
                all_points.extend(segment.get_centerline_points(num_points))
        return np.array(all_points, dtype=np.float32)

    def factory(
        env: "gymnasium_driving.environment.CarEnvironment",
    ) -> gymnasium_driving.environment.SpawnGoalInfo:
        centerline = _build_centerline(env)
        n = len(centerline)

        if fixed_proportion:
            prop = np.clip(proportion_mean, 0.01, 0.99)
        else:
            prop = np.clip(
                env.np_random.normal(proportion_mean, proportion_var),
                0.01,
                0.99,
            )

        goal_span = int(prop * (n - 1))
        max_spawn_idx = (n - 1) - goal_span

        chosen_spawn_idx = 0
        chosen_spawn_on_path = centerline[0].copy()
        chosen_spawn_pos = chosen_spawn_on_path.copy()
        chosen_spawn_heading = 0.0

        # TODO: probably useless as obstacles are generated after spawn points and thus no obstacles are available at this point, empty
        for _ in range(80):
            if fixed_proportion:
                spawn_idx = 0
            else:
                spawn_idx = int(env.np_random.uniform(0, max_spawn_idx + 1))
                spawn_idx = int(np.clip(spawn_idx, 0, n - 2))

            spawn_on_path = centerline[spawn_idx].copy()

            next_idx = min(spawn_idx + 1, n - 1)
            tangent = centerline[next_idx] - centerline[spawn_idx]
            length = float(np.linalg.norm(tangent))
            if length < 1e-9:
                tangent = np.array([1.0, 0.0], dtype=np.float32)
            else:
                tangent = tangent / length

            normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)

            base_heading = float(np.arctan2(tangent[1], tangent[0]))
            candidate_heading = base_heading + float(
                env.np_random.normal(0.0, heading_noise)
            )
            lateral_offset = float(env.np_random.normal(0.0, cte_noise))
            candidate_pos = (spawn_on_path + lateral_offset * normal).astype(np.float32)

            if not env._check_collision(
                position=candidate_pos,
                yaw=candidate_heading,
                check_road=True,
                respect_solid_road_borders=False,
            ):
                chosen_spawn_idx = spawn_idx
                chosen_spawn_on_path = spawn_on_path
                chosen_spawn_pos = candidate_pos
                chosen_spawn_heading = candidate_heading
                break

        goal_idx = min(chosen_spawn_idx + goal_span, n - 1)
        goal_pos = centerline[goal_idx].copy()

        path = np.array(
            [chosen_spawn_on_path]
            + [centerline[i] for i in range(chosen_spawn_idx + 1, goal_idx + 1)],
            dtype=np.float32,
        )

        return gymnasium_driving.environment.SpawnGoalInfo(
            spawn_pos=chosen_spawn_pos,
            spawn_heading=chosen_spawn_heading,
            goal_pos=goal_pos,
            path=path,
            goal_radius=3.0,
        )

    return factory
