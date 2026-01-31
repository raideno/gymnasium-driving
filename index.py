import cristal

import environments
import environments.bicycle as bicycle

import controllers.manual as manual
import controllers.stanley as stanley
import controllers.clothoids as clothoids
import controllers.purepursuit as purepursuit
import controllers.clothoids_road_aware as clothoids_road_aware


env = cristal.build_cristal_environment(render_mode="human")

controller = clothoids.ClothoidTentaclesController(
    num_tentacles=41,
    # Time horizon constant,
    t0=7.0,
    # Length offset constant,
    l0=5.0,
    num_points_per_tentacle=50,

    wheelbase=1.75,
    # m/s^2,
    max_lateral_accel=4.0,
    # Comfortable braking deceleration,
    max_decel=1.5,

    # Selection weights (from paper),
    weight_clearance=0.1,          # a0,
    weight_curvature=0.2,          # a1,
    weight_trajectory=0.5,         # a2,

    # Velocity control,
    # m/s,
    target_velocity=5.0,
    kp_velocity=2.0,
)

controller = clothoids_road_aware.ClothoidRoadAwareController(
    num_tentacles=41,
    t0=7.0,
    l0=5.0,
    num_points_per_tentacle=50,

    wheelbase=1.75,
    vehicle_width=1.8,
    max_lateral_accel=4.0,
    max_decel=1.5,

    # Road boundary parameters
    road_boundary_margin=0.3,      # Safety margin from road edge
    
    # Selection weights (enhanced with road boundary)
    weight_clearance=0.1,          # a0
    weight_curvature=0.15,         # a1
    weight_trajectory=0.45,        # a2
    weight_road_boundary=0.3,      # a3 (new)

    target_velocity=6.0,
    kp_velocity=2.0,
)

controller = purepursuit.PurePursuitController(
    lookahead_distance=6.0,
    min_lookahead=3.0,
    max_lookahead=12.0,
    wheelbase=1.75,
    target_velocity=8.0,
    kp_velocity=2.0
)

controller = manual.ManualController(
    steering_speed=0.8,    # How fast steering changes (rad/s)
)

obs, info = env.reset()

for i in range(500):
    action = controller.get_action(
        observation=obs,
        path=env.path,
        obstacles=env.obstacles,
        road_network=env.road_network,
        dt=env.dt,
        max_steering=env.max_steering,
        max_acceleration=env.max_acceleration,
    )
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    env.overlay_manager.clear()
    controller.draw_debug(env, obs, env.path)
    
    # helpers.preview(env)

    if terminated or truncated:
        break

env.close()

print(f"\nSimulation completed after {env.performance_tracker.step_count} steps")
print(f"Final position: {obs['position']}")
print(f"Final velocity: {obs['velocity'][0]:.2f} m/s")