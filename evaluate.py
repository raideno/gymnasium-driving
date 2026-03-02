import os
import shutil

import hydra
import hydra.core
import hydra.core.hydra_config
import matplotlib.pyplot as plt
import numpy as np
import omegaconf

import src.helpers as helpers


def _make_eval_env(env_config, reward_config, wrappers):
    env = hydra.utils.instantiate(env_config)
    env = hydra.utils.instantiate(reward_config, environment=env)
    for wrapper in wrappers:
        env = hydra.utils.instantiate(wrapper, environment=env)
    return env


@hydra.main(version_base=None, config_path="configurations", config_name="evaluate")
def evaluate(configuration: omegaconf.DictConfig):
    output_directory = configuration.output_path

    if output_directory is None or not os.path.exists(output_directory):
        print("[evaluate]:", "no output path provided, cannot load pretrained model")
        return

    target_configuration = helpers.load_configuration(
        output_directory=output_directory,
        expected_script="train",
    )

    best_model_path = os.path.join(output_directory, "best_model.zip")

    controller, train_environment, _ = helpers.instantiate_configuration(
        configuration=target_configuration,
        output_directory=output_directory,
        load_best_model=os.path.exists(best_model_path),
        base_dir="./configurations",
    )

    has_custom_controller = "controller" in configuration

    if has_custom_controller:
        controller = hydra.utils.instantiate(
            configuration.controller,
            environment=train_environment,
        )

        hydra_output = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        evaluation_directory = os.path.join(hydra_output, "evaluation")
        os.makedirs(evaluation_directory, exist_ok=True)

        _copy_directory_excluding(
            src=output_directory,
            dst=os.path.join(evaluation_directory, "source"),
            exclude_dirs={"logs"},
        )

        print(
            f"[evaluate]: custom controller provided — results will be saved to:"
            f" {evaluation_directory}"
        )
    else:
        print(
            "[evaluate]: no controller specified in configuration,"
            " using pretrained model as controller"
        )
        evaluation_directory = os.path.join(output_directory, "evaluation")
        os.makedirs(evaluation_directory, exist_ok=True)
        print(f"[evaluate]: results will be saved to: {evaluation_directory}")

    n_eval_episodes = configuration.n_eval_episodes
    deterministic = configuration.deterministic

    wrappers = target_configuration.get("wrappers", [])

    env_configs = [
        ("cristal", target_configuration.cristal),
        ("straight", target_configuration.straight),
    ]

    all_episode_rewards = []
    all_episode_successes = []
    all_episode_ctes = []
    all_episode_heading_errors = []
    all_episode_jerks = []
    all_episode_env_names = []

    for env_name, env_cfg in env_configs:
        print(f"\n[evaluate](#episodes={n_eval_episodes}): evaluating on {env_name}...")

        eval_environment = _make_eval_env(
            env_cfg, target_configuration.reward, wrappers
        )

        for episode in range(n_eval_episodes):
            observation, _ = eval_environment.reset()
            done = False
            truncated = False

            total_reward = 0.0
            success = False
            step_ctes = []
            step_heading_errors = []
            step_steerings = []

            while not (done or truncated):
                action, _ = controller.predict(
                    observation,
                    deterministic=deterministic,
                )

                observation, reward, done, truncated, info = eval_environment.step(
                    action
                )

                total_reward += reward

                base_env = eval_environment.unwrapped
                step_ctes.append(abs(base_env.state["cte"]))
                step_heading_errors.append(abs(base_env.state["heading_error"]))
                step_steerings.append(float(action[0]))

                if isinstance(info, dict) and "is_success" in info:
                    success = success or bool(info["is_success"])

            episode_mean_cte = float(np.mean(step_ctes)) if step_ctes else 0.0
            episode_mean_heading_error = (
                float(np.mean(step_heading_errors)) if step_heading_errors else 0.0
            )
            if len(step_steerings) > 2:
                s = np.array(step_steerings)
                episode_mean_jerk = float(np.mean(np.abs(np.diff(np.diff(s)))))
            else:
                episode_mean_jerk = 0.0

            all_episode_rewards.append(total_reward)
            all_episode_successes.append(success)
            all_episode_ctes.append(episode_mean_cte)
            all_episode_heading_errors.append(episode_mean_heading_error)
            all_episode_jerks.append(episode_mean_jerk)
            all_episode_env_names.append(env_name)

            print(
                f"  [{env_name}][ep-{(episode + 1):>2}]:"
                f" reward={total_reward:>8.3f};"
                f" cte={episode_mean_cte:>6.3f};"
                f" heading_error={episode_mean_heading_error:>6.3f};"
                f" jerk={episode_mean_jerk:>6.4f};"
                f" success={success}"
            )

        eval_environment.close()

    mean_reward = np.mean(all_episode_rewards)
    std_reward = np.std(all_episode_rewards)
    success_rate = 100.0 * np.mean(all_episode_successes)
    mean_cte = float(np.mean(all_episode_ctes))
    mean_heading_error = float(np.mean(all_episode_heading_errors))
    mean_jerk = float(np.mean(all_episode_jerks))

    print("\n[evaluate] Results:")
    print(f"[evaluate] Mean reward:        {mean_reward:.3f} ± {std_reward:.3f}")
    print(f"[evaluate] Success rate:       {success_rate:.2f}%")
    print(f"[evaluate] Mean CTE:           {mean_cte:.3f} m")
    print(f"[evaluate] Mean heading error: {mean_heading_error:.3f} rad")
    print(f"[evaluate] Mean jerk:          {mean_jerk:.4f} rad/step²")

    results = {
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "success_rate": float(success_rate),
        "mean_cte": mean_cte,
        "mean_heading_error": mean_heading_error,
        "mean_jerk": mean_jerk,
        "n_eval_episodes": n_eval_episodes,
        "deterministic": deterministic,
        "episode_rewards": [float(r) for r in all_episode_rewards],
        "episode_successes": [bool(s) for s in all_episode_successes],
        "episode_ctes": [float(c) for c in all_episode_ctes],
        "episode_heading_errors": [float(h) for h in all_episode_heading_errors],
        "episode_jerks": [float(j) for j in all_episode_jerks],
        "episode_env_names": list(all_episode_env_names),
    }

    _save_evaluation_results(results, evaluation_directory)
    _save_plots(results, evaluation_directory)


def _copy_directory_excluding(src: str, dst: str, exclude_dirs: set):
    os.makedirs(dst, exist_ok=True)
    for item in os.listdir(src):
        if item in exclude_dirs:
            continue
        src_path = os.path.join(src, item)
        dst_path = os.path.join(dst, item)
        if os.path.isdir(src_path):
            _copy_directory_excluding(src_path, dst_path, exclude_dirs)
        else:
            shutil.copy2(src_path, dst_path)


def _save_evaluation_results(results: dict, output_directory: str):
    summary_path = os.path.join(output_directory, "evaluation_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"mean_reward:        {results['mean_reward']:.3f}\n")
        f.write(f"std_reward:         {results['std_reward']:.3f}\n")
        f.write(f"success_rate:       {results['success_rate']:.2f}%\n")
        f.write(f"mean_cte:           {results['mean_cte']:.3f} m\n")
        f.write(f"mean_heading_error: {results['mean_heading_error']:.3f} rad\n")
        f.write(f"mean_jerk:          {results['mean_jerk']:.4f} rad/step²\n")
        f.write(f"n_eval_episodes:    {results['n_eval_episodes']}\n")
        f.write(f"deterministic:      {results['deterministic']}\n")
        f.write("\nPer-episode results:\n")
        for i, (r, s, c, h, j, e) in enumerate(
            zip(
                results["episode_rewards"],
                results["episode_successes"],
                results["episode_ctes"],
                results["episode_heading_errors"],
                results["episode_jerks"],
                results["episode_env_names"],
            ),
            start=1,
        ):
            f.write(
                f"  ep {i:>3} [{e:>8}]: reward={r:>10.3f}  cte={c:>7.3f}"
                f"  heading_error={h:>7.3f}  jerk={j:>8.4f}  success={s}\n"
            )

    npz_path = os.path.join(output_directory, "evaluation_results.npz")
    np.savez(
        npz_path,
        episode_rewards=np.array(results["episode_rewards"]),
        episode_successes=np.array(results["episode_successes"]),
        episode_ctes=np.array(results["episode_ctes"]),
        episode_heading_errors=np.array(results["episode_heading_errors"]),
        episode_jerks=np.array(results["episode_jerks"]),
    )

    print(f"[evaluate]: summary saved → {summary_path}")
    print(f"[evaluate]: npz saved     → {npz_path}")


def _save_plots(results: dict, output_directory: str):
    env_names = results["episode_env_names"]
    unique_envs = sorted(set(env_names))
    palette = {name: f"C{i}" for i, name in enumerate(unique_envs)}
    colors = [palette[n] for n in env_names]
    xs = range(len(env_names))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Evaluation Results", fontsize=14)

    panels = [
        (axes[0, 0], "Reward", results["episode_rewards"]),
        (axes[0, 1], "Mean CTE (m)", results["episode_ctes"]),
        (axes[1, 0], "Mean heading error (rad)", results["episode_heading_errors"]),
        (axes[1, 1], "Mean steering jerk", results["episode_jerks"]),
    ]

    for ax, label, values in panels:
        ax.bar(xs, values, color=colors, edgecolor="none")
        mean_val = np.mean(values)
        ax.axhline(
            mean_val,
            color="red",
            linestyle="--",
            linewidth=1,
            label=f"mean={mean_val:.3f}",
        )
        ax.set_title(label)
        ax.set_xlabel("Episode")
        ax.legend(fontsize=8)

    handles = [plt.Rectangle((0, 0), 1, 1, color=palette[n]) for n in unique_envs]
    fig.legend(
        handles, unique_envs, loc="lower center", ncol=len(unique_envs), frameon=False
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    plot_path = os.path.join(output_directory, "evaluation_plots.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[evaluate]: plots saved    → {plot_path}")


if __name__ == "__main__":
    evaluate()
