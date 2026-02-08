import os
import json

import numpy as np

import matplotlib.pyplot as plt

from src.metrics.cross_track_error import compute_cross_track_error
from src.metrics.steering_smoothness import compute_steering_smoothness

def compute_episode_metrics(env, target_velocity: float = 5.0) -> dict:
    """Compute all metrics from the recorder data."""
    recorder = env.unwrapped.recorder
    data = recorder.to_arrays()
    
    if not data:
        return {}
    
    # Cross-track error
    positions = [pos for pos in data['positions']]
    reference_path = env.unwrapped.path
    cte_metrics = compute_cross_track_error(positions, reference_path)
    
    # Steering smoothness
    steering_angles = list(data['steering_angles'])
    dt = env.unwrapped.DELTA_TIME
    steering_metrics = compute_steering_smoothness(steering_angles, dt)
    
    # Velocity tracking metrics
    velocities = data['velocities']
    velocity_error = np.abs(velocities - target_velocity)
    velocity_metrics = {
        'velocity_mean': float(np.mean(velocities)),
        'velocity_std': float(np.std(velocities)),
        'velocity_error_mean': float(np.mean(velocity_error)),
        'velocity_error_rms': float(np.sqrt(np.mean(velocity_error ** 2))),
    }
    
    # Episode stats
    episode_metrics = {
        'episode_length': len(data['timestamps']),
        'episode_duration': float(data['timestamps'][-1]) if len(data['timestamps']) > 0 else 0.0,
        'terminated': bool(data.get('terminated', False)),
        'truncated': bool(data.get('truncated', False)),
    }
    
    return {**cte_metrics, **steering_metrics, **velocity_metrics, **episode_metrics}


def evaluate(model, env, num_episodes: int = 10, save_gif: bool = True):
    """Run evaluation episodes and produce diagnostic plots / gif."""

    print(f"\n{'─' * 60}")
    print(f"  Evaluation  ({num_episodes} episodes, deterministic)")
    print(f"{'─' * 60}")

    all_rewards = []
    all_steps = []
    all_metrics = []
    episode_reward_history = []  # Track cumulative reward per episode

    for ep in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        steps = 0
        rewards_in_episode = []

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            rewards_in_episode.append(reward)
            steps += 1
            if terminated or truncated:
                break

        vel = env.unwrapped.state["velocity"]
        metrics = compute_episode_metrics(env)
        
        all_rewards.append(total_reward)
        all_steps.append(steps)
        all_metrics.append(metrics)
        episode_reward_history.append(total_reward)
        
        print(f"    ep {ep + 1:>2d}:  steps={steps:>4d}  reward={total_reward:>8.1f}  "
              f"final_v={vel:.2f} m/s  cte_rms={metrics.get('cte_rms', 0):.3f}m")

    # Aggregate metrics
    agg_metrics = {}
    if all_metrics:
        for key in all_metrics[0].keys():
            if isinstance(all_metrics[0][key], (int, float)):
                values = [m[key] for m in all_metrics if key in m]
                agg_metrics[f'{key}_mean'] = float(np.mean(values))
                agg_metrics[f'{key}_std'] = float(np.std(values))
    
    agg_metrics['reward_mean'] = float(np.mean(all_rewards))
    agg_metrics['reward_std'] = float(np.std(all_rewards))
    agg_metrics['steps_mean'] = float(np.mean(all_steps))
    agg_metrics['num_episodes'] = num_episodes

    print(f"\n    ── Summary Statistics ──")
    print(f"    reward:       {np.mean(all_rewards):>8.1f} ± {np.std(all_rewards):.1f}")
    print(f"    steps:        {np.mean(all_steps):>8.0f} ± {np.std(all_steps):.0f}")
    print(f"    cte_rms:      {agg_metrics.get('cte_rms_mean', 0):>8.3f} ± {agg_metrics.get('cte_rms_std', 0):.3f} m")
    print(f"    steer_jerk:   {agg_metrics.get('steering_jerk_rms_mean', 0):>8.3f} ± {agg_metrics.get('steering_jerk_rms_std', 0):.3f} rad/s³")
    print(f"    velocity_err: {agg_metrics.get('velocity_error_rms_mean', 0):>8.3f} ± {agg_metrics.get('velocity_error_rms_std', 0):.3f} m/s")

    # ── Visual evaluation (single long episode) ──
    print("\n  Running visual evaluation …")
    obs, _ = env.reset()
    positions, velocities, frames = [], [], []

    for _ in range(1500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        ego_pos = np.array([env.unwrapped.state["x"], env.unwrapped.state["y"]], dtype=np.float32)
        ego_velocity = env.unwrapped.state["yaw"]
        
        positions.append(ego_pos.copy())
        velocities.append(ego_velocity)

        frame = env.render()
        if frame is not None:
            frames.append(frame)

        if terminated or truncated:
            break

    positions = np.array(positions)
    velocities = np.array(velocities)

    # ── Save animated gif ──
    if save_gif and len(frames) > 0:
        try:
            from PIL import Image as PILImage
            images = [PILImage.fromarray(f) for f in frames[::2]]  # every other frame
            gif_path = os.path.join(os.path.dirname(__file__), "evaluation.local.gif")
            images[0].save(
                gif_path,
                save_all=True,
                append_images=images[1:],
                duration=66,
                loop=0,
            )
            print(f"    GIF saved → {gif_path}")
        except Exception as exc:
            print(f"    (could not save gif: {exc})")

    # ── Plots ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Reward progression across episodes
    ax = axes[0, 0]
    episodes = np.arange(1, num_episodes + 1)
    ax.bar(episodes, episode_reward_history, color='steelblue', alpha=0.7)
    ax.axhline(np.mean(episode_reward_history), ls='--', color='red', label=f'Mean: {np.mean(episode_reward_history):.1f}')
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Reward per Episode")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Trajectory with CTE visualization
    ax = axes[0, 1]
    path = env.unwrapped.path
    if path is not None:
        ax.plot(path[:, 0], path[:, 1], "b--", alpha=0.25, linewidth=2, label="Reference path")
    sc = ax.scatter(positions[:, 0], positions[:, 1],
                    c=velocities, cmap="RdYlGn", s=5, label="Agent")
    ax.scatter(*positions[0], c="green", s=120, marker="*", zorder=5, label="Start")
    ax.scatter(*positions[-1], c="red", s=120, marker="X", zorder=5, label="End")
    plt.colorbar(sc, ax=ax, label="Velocity (m/s)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Learned Trajectory")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")

    # Plot 3: Velocity profile
    ax = axes[1, 0]
    time_axis = np.arange(len(velocities)) * env.unwrapped.DELTA_TIME
    ax.plot(time_axis, velocities, label='Velocity')
    ax.axhline(5.0, ls="--", color="gray", alpha=0.5, label="Target 5 m/s")
    ax.fill_between(time_axis, velocities, 5.0, alpha=0.2, color='red')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title("Velocity Profile")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Key metrics comparison
    ax = axes[1, 1]
    metric_names = ['CTE RMS\n(m)', 'Steering Jerk\n(rad/s³)', 'Velocity Error\n(m/s)']
    metric_values = [
        agg_metrics.get('cte_rms_mean', 0),
        agg_metrics.get('steering_jerk_rms_mean', 0),
        agg_metrics.get('velocity_error_rms_mean', 0),
    ]
    metric_stds = [
        agg_metrics.get('cte_rms_std', 0),
        agg_metrics.get('steering_jerk_rms_std', 0),
        agg_metrics.get('velocity_error_rms_std', 0),
    ]
    bars = ax.bar(metric_names, metric_values, color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.7)
    ax.errorbar(metric_names, metric_values, yerr=metric_stds, fmt='none', color='black', capsize=5)
    ax.set_ylabel("Value")
    ax.set_title("Key Performance Metrics (lower is better)")
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), "evaluation_results.local.png")
    plt.savefig(plot_path, dpi=150)
    print(f"    Plot saved → {plot_path}")
    plt.close()

    # ── Save metrics to JSON ──
    metrics_path = os.path.join(os.path.dirname(__file__), "evaluation_metrics.local.json")
    with open(metrics_path, 'w') as f:
        json.dump(agg_metrics, f, indent=2)
    print(f"    Metrics saved → {metrics_path}")

    return agg_metrics
