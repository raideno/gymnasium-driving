import os

import numpy as np

import matplotlib.pyplot as plt

from stable_baselines3.common.results_plotter import load_results, ts2xy

def plot_training_results(log_dir: str, output_path: str):
    """Generate separate plots for training results using ts2xy with different x-axis options."""
    
    x_axis_options = ["timesteps", "episodes", "walltime_hrs"]
    x_axis_labels = {
        "timesteps": "Timesteps",
        "episodes": "Episodes", 
        "walltime_hrs": "Wall Time (hours)"
    }
    
    try:
        results = load_results(log_dir)
    except Exception as e:
        print(f"    Could not load training results: {e}")
        return
    
    for x_axis in x_axis_options:
        try:
            x, y = ts2xy(results, x_axis)
            
            if len(x) == 0:
                print(f"    No data for {x_axis}, skipping...")
                continue
            
            plt.figure(figsize=(10, 6))
            plt.plot(x, y, alpha=0.7, linewidth=1)
            
            # Add moving average
            if len(y) > 10:
                window = min(50, len(y) // 5)
                moving_avg = np.convolve(y, np.ones(window)/window, mode='valid')
                plt.plot(
                    x[window - 1:],
                    moving_avg,
                    color="red",
                    linewidth=2,
                    label=f"Moving Avg ({window} ep)",
                )
                plt.legend()

            plt.xlabel(x_axis_labels[x_axis])
            plt.ylabel("Episode Reward")
            plt.title(f"Training Progress ({x_axis_labels[x_axis]})")
            plt.grid(True, alpha=0.3)
            
            plot_path = os.path.join(output_path, f"training_progress_{x_axis}.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"[plot]: {plot_path}")
            
        except Exception as e:
            print(f"[plot](error): could not generate plot for {x_axis}: {e}")
            
    print()

def evaluate(model, env, output_path: str, num_episodes: int = 10):
    """Run evaluation episodes and produce diagnostic plots / gif.
    
    Args:
        model: The trained model to evaluate.
        env: The environment to evaluate in.
        output_path: Directory path where all outputs will be saved.
        num_episodes: Number of evaluation episodes to run.
    """
    
    os.makedirs(output_path, exist_ok=True)

    all_rewards = []
    all_steps = []
    episode_reward_history = []

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

        all_rewards.append(total_reward)
        all_steps.append(steps)
        episode_reward_history.append(total_reward)
        
        print(f"[ep-{(ep + 1):>2}]: steps={steps:>4d}; reward={total_reward:>4.1f}")

    print()
    print(f"[reward]: {np.mean(all_rewards):>8.1f} ± {np.std(all_rewards):.1f}")
    print(f"[steps]: {np.mean(all_steps):>8.0f} ± {np.std(all_steps):.0f}")

    plt.figure(figsize=(10, 6))
    episodes = np.arange(1, num_episodes + 1)
    plt.bar(episodes, episode_reward_history, color="steelblue", alpha=0.7)
    plt.axhline(
        np.mean(episode_reward_history),
        ls="--",
        color="red",
        label=f"Mean: {np.mean(episode_reward_history):.1f}",
    )
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Reward per Episode")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plot_path = os.path.join(output_path, "evaluation_reward_per_episode.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Plot saved → {plot_path}")

    agg_metrics = {
        "mean_reward": float(np.mean(all_rewards)),
        "std_reward": float(np.std(all_rewards)),
        "mean_steps": float(np.mean(all_steps)),
        "std_steps": float(np.std(all_steps)),
    }

    return agg_metrics
