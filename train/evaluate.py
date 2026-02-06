import os

import numpy as np

import matplotlib.pyplot as plt

def evaluate(model, env, num_episodes: int = 5, save_gif: bool = True):
    """Run evaluation episodes and produce diagnostic plots / gif."""

    print(f"\n{'─' * 60}")
    print(f"  Evaluation  ({num_episodes} episodes, deterministic)")
    print(f"{'─' * 60}")

    all_rewards = []
    all_steps = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        steps = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            if terminated or truncated:
                break

        vel = env.unwrapped.state[3]
        all_rewards.append(total_reward)
        all_steps.append(steps)
        print(f"    ep {ep + 1:>2d}:  steps={steps:>4d}  reward={total_reward:>8.1f}  "
              f"final_v={vel:.2f} m/s")

    print(f"\n    mean reward = {np.mean(all_rewards):.1f}  "
          f"mean steps = {np.mean(all_steps):.0f}")

    # ── Visual evaluation (single long episode) ──
    print("\n  Running visual evaluation …")
    obs, _ = env.reset()
    positions, velocities, frames = [], [], []

    for _ in range(1500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        state = env.unwrapped.state
        positions.append(state[:2].copy())
        velocities.append(state[3])

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
            gif_path = os.path.join(os.path.dirname(__file__), "evaluation.gif")
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

    # ── Trajectory + velocity plot ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    path = env.unwrapped.path
    if path is not None:
        ax.plot(path[:, 0], path[:, 1], "b--", alpha=0.25, label="Reference path")
    sc = ax.scatter(positions[:, 0], positions[:, 1],
                    c=velocities, cmap="RdYlGn", s=5, label="Agent")
    ax.scatter(*positions[0], c="green", s=120, marker="*", zorder=5, label="Start")
    ax.scatter(*positions[-1], c="red", s=120, marker="X", zorder=5, label="End")
    plt.colorbar(sc, ax=ax, label="Velocity (m/s)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Learned trajectory")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")

    ax = axes[1]
    time_axis = np.arange(len(velocities)) * env.unwrapped.dt
    ax.plot(time_axis, velocities)
    ax.axhline(5.0, ls="--", color="gray", alpha=0.5, label="target 5 m/s")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title("Velocity profile")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), "training_results.png")
    plt.savefig(plot_path, dpi=150)
    print(f"    Plot saved → {plot_path}")
    plt.close()
