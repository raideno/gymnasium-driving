import os
import shutil

import hydra
import hydra.core
import hydra.core.hydra_config
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

            while not (done or truncated):
                action, _ = controller.predict(
                    observation,
                    deterministic=deterministic,
                )

                observation, reward, done, truncated, info = eval_environment.step(
                    action
                )

                total_reward += reward

                if isinstance(info, dict) and "is_success" in info:
                    success = success or bool(info["is_success"])

            all_episode_rewards.append(total_reward)
            all_episode_successes.append(success)

            print(
                f"  [{env_name}][ep-{(episode + 1):>2}]:"
                f" reward={total_reward:>8.3f};"
                f" success={success}"
            )

        eval_environment.close()

    mean_reward = np.mean(all_episode_rewards)
    std_reward = np.std(all_episode_rewards)
    success_rate = 100.0 * np.mean(all_episode_successes)

    print("\n[evaluate] Results:")
    print(f"[evaluate] Mean reward:   {mean_reward:.3f} ± {std_reward:.3f}")
    print(f"[evaluate] Success rate:  {success_rate:.2f}%")

    results = {
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "success_rate": float(success_rate),
        "n_eval_episodes": n_eval_episodes,
        "deterministic": deterministic,
        "episode_rewards": [float(r) for r in all_episode_rewards],
        "episode_successes": [bool(s) for s in all_episode_successes],
    }

    _save_evaluation_results(results, evaluation_directory)


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
        f.write(f"mean_reward:    {results['mean_reward']:.3f}\n")
        f.write(f"std_reward:     {results['std_reward']:.3f}\n")
        f.write(f"success_rate:   {results['success_rate']:.2f}%\n")
        f.write(f"n_eval_episodes:{results['n_eval_episodes']}\n")
        f.write(f"deterministic:  {results['deterministic']}\n")
        f.write("\nPer-episode rewards:\n")
        for i, (r, s) in enumerate(
            zip(results["episode_rewards"], results["episode_successes"]), start=1
        ):
            f.write(f"  ep {i:>3}: reward={r:>10.3f}  success={s}\n")

    npz_path = os.path.join(output_directory, "evaluation_results.npz")
    np.savez(
        npz_path,
        episode_rewards=np.array(results["episode_rewards"]),
        episode_successes=np.array(results["episode_successes"]),
    )

    print(f"[evaluate]: summary saved → {summary_path}")
    print(f"[evaluate]: npz saved     → {npz_path}")


if __name__ == "__main__":
    evaluate()
