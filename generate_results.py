#!/usr/bin/env python3
"""
generate_results.py
===================
One-shot script that:
  1. Trains PPO (continuous), TRPO (continuous) and DQN (discrete) for
     `TOTAL_TIMESTEPS` steps each.
  2. Evaluates all five controllers:
       - Clothoid Tentacles  (no training)
       - Pure Pursuit        (no training)
       - PPO  (freshly trained)
       - TRPO (freshly trained)
       - DQN  (freshly trained)
  3. Writes to  results/
       - table_data.json          →  all numbers needed for the LaTeX table
       - fig_train_reward.png     →  mean episode reward training curve
       - fig_train_cte.png        →  mean absolute CTE training curve

Checkpointing
-------------
The script is fully resumable.  For each RL agent it saves:
  - results/checkpoints/<name>_model.zip      trained SB3 model
  - results/checkpoints/<name>_recorder.json  reward/CTE history

For evaluation it saves one file per controller as soon as it finishes:
  - results/eval_<name>.json

On restart, any already-present checkpoint / eval file is loaded directly and
the corresponding step is skipped entirely.

Usage
-----
    cd code/experimentations/gymnasium
    python generate_results.py
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import stable_baselines3
import stable_baselines3.common.callbacks
import stable_baselines3.common.vec_env

import gymnasium_driving.wrappers.observations.with_base_info as base_info_module
import gymnasium_driving.wrappers.observations.with_obstacles_info as obs_info_module
import gymnasium_driving.wrappers.observations.with_path_info as path_info_module
import gymnasium_driving.wrappers.observations.with_road_info as road_info_module
import gymnasium_driving.wrappers.rewards.reward as reward_module
import src.controllers.clothoids as clothoids_module
import src.controllers.dqn as dqn_module
import src.controllers.ppo as ppo_module
import src.controllers.purepursuit as pp_module
import src.controllers.trpo as trpo_module
import src.environments.cristal as cristal_module
import src.environments.straight as straight_module

# ===========================================================================
# CONFIGURATION  —  edit these if needed
# ===========================================================================

TOTAL_TIMESTEPS = 1_000_000  # steps per RL agent
N_EVAL_EPISODES = 20  # evaluation episodes per controller
RESULTS_DIR = "results"  # output directory (created automatically)
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
MOVING_AVG_WINDOW = 20  # smoothing window for training curves


# ===========================================================================
# CHECKPOINT HELPERS
# ===========================================================================


def _recorder_path(name: str) -> str:
    return os.path.join(CHECKPOINT_DIR, f"{name}_recorder.json")


def _model_path(name: str) -> str:
    return os.path.join(CHECKPOINT_DIR, f"{name}_model")  # SB3 adds .zip itself


def _model_zip_path(name: str) -> str:
    return _model_path(name) + ".zip"


def _eval_path(name: str) -> str:
    # Use a filesystem-safe version of the controller name as filename
    safe = name.lower().replace(" ", "_")
    return os.path.join(RESULTS_DIR, f"eval_{safe}.json")


def _save_recorder(name: str, recorder: "TrainingRecorder") -> None:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    data = {
        "reward_history": recorder.reward_history,
        "cte_history": recorder.cte_history,
    }
    with open(_recorder_path(name), "w") as f:
        json.dump(data, f)
    print(f"[checkpoint] recorder saved → {_recorder_path(name)}")


def _load_recorder(name: str) -> "TrainingRecorder | None":
    path = _recorder_path(name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    rec = TrainingRecorder()
    rec.reward_history = [tuple(x) for x in data["reward_history"]]
    rec.cte_history = [tuple(x) for x in data["cte_history"]]
    print(f"[checkpoint] recorder loaded ← {path}")
    return rec


def _save_eval(name: str, metrics: dict) -> None:
    with open(_eval_path(name), "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[checkpoint] eval saved → {_eval_path(name)}")


def _load_eval(name: str) -> "dict | None":
    path = _eval_path(name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        metrics = json.load(f)
    print(f"[checkpoint] eval loaded ← {path}  (skipping evaluation)")
    return metrics


# ===========================================================================
# ENVIRONMENT FACTORIES
# ===========================================================================


def _apply_observation_wrappers(env):
    """Apply the standard observation wrappers used during training."""
    env = path_info_module.WithPathInfo(environment=env)
    env = obs_info_module.WithObstaclesInfo(environment=env)
    env = base_info_module.WithBaseInfo(environment=env)
    env = road_info_module.WithRoadInfo(environment=env)
    return env


def _wrap_eval_env(env):
    """Apply reward + observation wrappers for a standalone eval environment."""
    env = reward_module.Reward(environment=env)
    env = _apply_observation_wrappers(env)
    return env


def make_vec_env(discrete: bool):
    """
    Build a DummyVecEnv with two sub-environments (cristal + straight),
    matching the setup used in train.py.
    """

    def _make_cristal():
        env = cristal_module.make_environment(discrete=discrete)
        env = reward_module.Reward(environment=env)
        env = _apply_observation_wrappers(env)
        return env

    def _make_straight():
        env = straight_module.make_environment(discrete=discrete)
        env = reward_module.Reward(environment=env)
        env = _apply_observation_wrappers(env)
        return env

    return stable_baselines3.common.vec_env.DummyVecEnv([_make_cristal, _make_straight])


# ===========================================================================
# TRAINING
# ===========================================================================


class TrainingRecorder(stable_baselines3.common.callbacks.BaseCallback):
    """
    Records at the end of every rollout:
      - mean episode reward  (from ep_info_buffer)
      - mean absolute CTE    (accumulated per step)
    """

    def __init__(self):
        super().__init__()
        self.reward_history: list[tuple[int, float]] = []  # (timestep, mean_reward)
        self.cte_history: list[tuple[int, float]] = []  # (timestep, mean_abs_cte)
        self._step_ctes: list[float] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "cte" in info:
                self._step_ctes.append(abs(info["cte"]))
        return True

    def _on_rollout_end(self) -> None:
        if not self.model.ep_info_buffer:
            return
        rewards = [ep["r"] for ep in self.model.ep_info_buffer]
        self.reward_history.append((self.num_timesteps, float(np.mean(rewards))))
        if self._step_ctes:
            self.cte_history.append(
                (self.num_timesteps, float(np.mean(self._step_ctes)))
            )
            self._step_ctes = []


def train_rl_agent(name: str, controller, total_timesteps: int) -> TrainingRecorder:
    """
    Train `controller` for `total_timesteps` steps.

    If a checkpoint already exists for `name`, the model weights and recorder
    history are restored and training is skipped entirely.
    """
    # --- check for existing checkpoint ---
    existing_recorder = _load_recorder(name)
    if existing_recorder is not None and os.path.exists(_model_zip_path(name)):
        print(
            f"[checkpoint] model loaded  ← {_model_zip_path(name)}  (skipping training)"
        )
        controller.model = controller.model.load(
            _model_zip_path(name),
            env=controller.env,
        )
        return existing_recorder

    # --- train from scratch ---
    print(f"\n{'=' * 60}")
    print(f"  Training: {name}  ({total_timesteps:,} steps)")
    print(f"{'=' * 60}")

    recorder = TrainingRecorder()
    controller.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
        callback=recorder,
    )
    print(f"  Done training {name}.")

    # --- save checkpoint immediately ---
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    controller.model.save(_model_path(name))
    print(f"[checkpoint] model saved  → {_model_zip_path(name)}")
    _save_recorder(name, recorder)

    return recorder


# ===========================================================================
# EVALUATION
# ===========================================================================


def evaluate_controller(
    name: str,
    controller,
    discrete: bool,
    n_episodes: int,
) -> dict:
    """
    Run `n_episodes` episodes on each of the two environments and return
    aggregated metrics.  Results are persisted immediately after completion.

    If a result file already exists for `name`, it is loaded and returned
    without running any episodes.
    """
    # --- check for existing eval ---
    cached = _load_eval(name)
    if cached is not None:
        return cached

    print(f"\n--- Evaluating: {name} ---")

    env_specs = [
        (
            "cristal",
            lambda: _wrap_eval_env(cristal_module.make_environment(discrete=discrete)),
        ),
        (
            "straight",
            lambda: _wrap_eval_env(straight_module.make_environment(discrete=discrete)),
        ),
    ]

    all_rewards: list[float] = []
    all_successes: list[bool] = []
    all_ctes: list[float] = []
    all_heading_errors: list[float] = []
    all_steerings: list[float] = []  # every per-step steering value

    for env_name, env_factory in env_specs:
        env = env_factory()

        for ep in range(n_episodes):
            obs, _ = env.reset()
            done = truncated = False

            ep_reward = 0.0
            ep_success = False
            ep_ctes: list[float] = []
            ep_heading_errors: list[float] = []
            ep_step_steerings: list[float] = []

            while not (done or truncated):
                action, _ = controller.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)

                ep_reward += float(reward)

                base_env = env.unwrapped
                ep_ctes.append(abs(base_env.state["cte"]))
                ep_heading_errors.append(abs(base_env.state["heading_error"]))

                raw = action
                ep_step_steerings.append(
                    float(raw[0]) if getattr(raw, "ndim", 0) > 0 else float(raw)
                )

                if isinstance(info, dict) and info.get("is_success", False):
                    ep_success = True

            all_rewards.append(ep_reward)
            all_successes.append(ep_success)
            all_ctes.append(float(np.mean(ep_ctes)) if ep_ctes else 0.0)
            all_heading_errors.append(
                float(np.mean(ep_heading_errors)) if ep_heading_errors else 0.0
            )
            all_steerings.extend(ep_step_steerings)

            print(
                f"  [{env_name}][ep {ep + 1:>2}/{n_episodes}]"
                f"  reward={ep_reward:>8.2f}"
                f"  cte={all_ctes[-1]:>5.3f}"
                f"  heading={all_heading_errors[-1]:>5.3f}"
                f"  success={ep_success}"
            )

        env.close()

    s_arr = np.array(all_steerings)
    mean_steering_rate = (
        float(np.mean(np.abs(np.diff(s_arr)))) if len(s_arr) > 1 else 0.0
    )

    metrics = {
        "mean_reward": float(np.mean(all_rewards)),
        "std_reward": float(np.std(all_rewards)),
        "success_rate": 100.0 * float(np.mean(all_successes)),
        "mean_cte": float(np.mean(all_ctes)),
        "mean_heading_error": float(np.mean(all_heading_errors)),
        "mean_steering_rate": mean_steering_rate,
    }

    print(
        f"\n  {name} → "
        f"reward={metrics['mean_reward']:.2f}±{metrics['std_reward']:.2f}  "
        f"success={metrics['success_rate']:.1f}%  "
        f"CTE={metrics['mean_cte']:.3f}m  "
        f"heading={metrics['mean_heading_error']:.3f}rad  "
        f"|Δδ|={metrics['mean_steering_rate']:.4f}"
    )

    # --- persist immediately ---
    _save_eval(name, metrics)

    return metrics


# ===========================================================================
# PLOTTING
# ===========================================================================


def _moving_average(values: list[float], window: int) -> np.ndarray:
    if len(values) < window:
        return np.array(values)
    return np.convolve(values, np.ones(window) / window, mode="valid")


def save_training_curves(
    recorders: dict[str, TrainingRecorder], output_dir: str
) -> None:
    """
    Save (or overwrite) two PNG files:
      fig_train_reward.png  —  mean episode reward (moving average)
      fig_train_cte.png     —  mean absolute CTE   (moving average)

    Called after every training run so the charts are always up to date.
    """
    colors = {"PPO": "#2196F3", "TRPO": "#4CAF50", "DQN": "#FF5722"}

    for metric_key, ylabel, title_suffix, filename in [
        (
            "reward_history",
            "Mean episode reward",
            "Mean episode reward during training",
            "fig_train_reward.png",
        ),
        (
            "cte_history",
            "Mean absolute CTE (m)",
            "Mean absolute cross-track error per rollout during training",
            "fig_train_cte.png",
        ),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4))

        for agent_name, recorder in recorders.items():
            history = getattr(recorder, metric_key)
            if not history:
                continue
            steps, values = zip(*history)
            smoothed = _moving_average(list(values), MOVING_AVG_WINDOW)
            smoothed_steps = list(steps)[len(steps) - len(smoothed) :]
            ax.plot(
                smoothed_steps,
                smoothed,
                label=agent_name,
                color=colors.get(agent_name),
                linewidth=1.8,
            )

        ax.set_xlabel("Environment steps")
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"{title_suffix}\n(moving average over {MOVING_AVG_WINDOW} episodes)"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        path = os.path.join(output_dir, filename)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[plot] saved → {path}")


# ===========================================================================
# TABLE SUMMARY
# ===========================================================================

ROW_ORDER = ["Clothoid Tentacles", "Pure Pursuit", "PPO", "TRPO", "DQN"]


def _load_all_evals() -> dict[str, dict]:
    """Collect every eval_*.json that has been written so far."""
    metrics = {}
    for name in ROW_ORDER:
        data = _load_eval(name)
        if data is not None:
            metrics[name] = data
    return metrics


def save_table_data(all_metrics: dict[str, dict], output_dir: str) -> None:
    """Write table_data.json and print a ready-to-copy LaTeX snippet."""
    path = os.path.join(output_dir, "table_data.json")
    with open(path, "w") as f:
        json.dump(all_metrics, f, indent=4)
    print(f"[table] saved → {path}")

    print("\n" + "=" * 72)
    print("LATEX TABLE VALUES")
    print("=" * 72)
    header = (
        f"{'Controller':<28}  {'Reward':>9}  {'Success%':>9}"
        f"  {'CTE(m)':>8}  {'Heading(rad)':>13}  {'|Δδ|':>8}"
    )
    print(header)
    print("-" * len(header))
    for name in ROW_ORDER:
        m = all_metrics.get(name)
        if m is None:
            continue
        print(
            f"{name:<28}  "
            f"{m['mean_reward']:>9.2f}  "
            f"{m['success_rate']:>8.1f}%  "
            f"{m['mean_cte']:>8.3f}  "
            f"{m['mean_heading_error']:>13.3f}  "
            f"{m['mean_steering_rate']:>8.4f}"
        )
    print("=" * 72)

    print("\nLaTeX midrule rows (copy-paste):\n")
    for name in ROW_ORDER:
        m = all_metrics.get(name)
        if m is None:
            continue
        print(
            f"        {name:<30} & "
            f"{m['mean_reward']:.2f} & "
            f"{m['success_rate']:.1f} & "
            f"{m['mean_cte']:.3f} & "
            f"{m['mean_heading_error']:.3f} \\\\"
        )


# ===========================================================================
# MAIN
# ===========================================================================


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Build training environments
    # ------------------------------------------------------------------
    print("\n[1/4] Building training environments …")
    vec_env_continuous = make_vec_env(discrete=False)
    vec_env_discrete = make_vec_env(discrete=True)

    # ------------------------------------------------------------------
    # 2. Train RL agents  (skipped per-agent if checkpoint exists)
    # ------------------------------------------------------------------
    print("\n[2/4] Training RL agents …")

    ppo_controller = ppo_module.PPOController(
        environment=vec_env_continuous,
        model_kwargs=dict(
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            ent_coef=0.0,
            verbose=0,
        ),
    )
    ppo_recorder = train_rl_agent("PPO", ppo_controller, TOTAL_TIMESTEPS)
    # Re-generate training curves after every completed (or loaded) agent
    save_training_curves({"PPO": ppo_recorder}, RESULTS_DIR)

    trpo_controller = trpo_module.TRPOController(
        environment=vec_env_continuous,
        model_kwargs=dict(
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_critic_updates=10,
            gamma=0.99,
            verbose=0,
        ),
    )
    trpo_recorder = train_rl_agent("TRPO", trpo_controller, TOTAL_TIMESTEPS)
    save_training_curves({"PPO": ppo_recorder, "TRPO": trpo_recorder}, RESULTS_DIR)

    dqn_controller = dqn_module.DQNController(
        environment=vec_env_discrete,
        model_kwargs=dict(
            learning_rate=0.0001,
            buffer_size=500_000,
            learning_starts=10_000,
            batch_size=256,
            gamma=0.99,
            exploration_fraction=0.3,
            exploration_final_eps=0.05,
            target_update_interval=2500,
            max_grad_norm=5.0,
            train_freq=4,
            verbose=0,
            policy_kwargs=dict(net_arch=[256, 256]),
        ),
    )
    dqn_recorder = train_rl_agent("DQN", dqn_controller, TOTAL_TIMESTEPS)
    save_training_curves(
        {"PPO": ppo_recorder, "TRPO": trpo_recorder, "DQN": dqn_recorder},
        RESULTS_DIR,
    )

    vec_env_continuous.close()
    vec_env_discrete.close()

    # ------------------------------------------------------------------
    # 3. Evaluate all five controllers  (skipped per-controller if eval
    #    file already exists)
    # ------------------------------------------------------------------
    print("\n[3/4] Evaluating all controllers …")

    # Classical controllers need a live env just to read physical constants
    # from unwrapped; we build a throwaway one for __init__ only.
    _tmp = _wrap_eval_env(cristal_module.make_environment(discrete=False))

    clothoids_controller = clothoids_module.ClothoidTentaclesController(
        environment=_tmp
    )
    purepursuit_controller = pp_module.PurePursuitController(
        environment=_tmp,
        lookahead_distance=5.0,
        target_velocity=5.0,
    )
    _tmp.close()

    all_metrics: dict[str, dict] = {}

    for name, controller, discrete in [
        ("Clothoid Tentacles", clothoids_controller, False),
        ("Pure Pursuit", purepursuit_controller, False),
        ("PPO", ppo_controller, False),
        ("TRPO", trpo_controller, False),
        ("DQN", dqn_controller, True),
    ]:
        all_metrics[name] = evaluate_controller(
            name=name,
            controller=controller,
            discrete=discrete,
            n_episodes=N_EVAL_EPISODES,
        )
        # Rewrite the summary table after every controller finishes
        save_table_data(all_metrics, RESULTS_DIR)

    # ------------------------------------------------------------------
    # 4. Final summary
    # ------------------------------------------------------------------
    print(f"\nAll done!  Results written to: {os.path.abspath(RESULTS_DIR)}/")
    print("  fig_train_reward.png  ← Figure: training reward curve")
    print("  fig_train_cte.png     ← Figure: training CTE curve")
    print("  table_data.json       ← All numbers for the LaTeX table")
    print(f"  checkpoints/          ← Trained model weights + recorder histories")


if __name__ == "__main__":
    main()
