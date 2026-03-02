import os

import dotenv
import hydra
import hydra.core
import hydra.core.hydra_config
import omegaconf
import stable_baselines3
import stable_baselines3.common
import stable_baselines3.common.callbacks
import stable_baselines3.common.logger
import stable_baselines3.common.monitor
import stable_baselines3.common.vec_env
import wandb
import wandb.integration
import wandb.integration.sb3

import src.helpers as helpers

print("[.env]:", dotenv.load_dotenv(dotenv_path=".env"))


def _make_env(env_config, reward_config, wrappers, monitor_path):
    env = hydra.utils.instantiate(env_config, render_mode="rgb_array")
    env = hydra.utils.instantiate(reward_config, environment=env)
    for wrapper in wrappers:
        env = hydra.utils.instantiate(wrapper, environment=env)
    env = stable_baselines3.common.monitor.Monitor(env=env, filename=monitor_path)
    return env


class _TrainMetricsCallback(stable_baselines3.common.callbacks.BaseCallback):
    def __init__(self):
        super().__init__()
        self._ctes = []
        self._heading_errors = []

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            self._ctes.append(abs(info["cte"]))
            self._heading_errors.append(abs(info["heading_error"]))
        return True

    def _on_rollout_end(self) -> None:
        if not self.model.ep_info_buffer:
            return
        rewards = [ep["r"] for ep in self.model.ep_info_buffer]
        lengths = [ep["l"] for ep in self.model.ep_info_buffer]
        metrics = {
            "train/mean_reward": sum(rewards) / len(rewards),
            "train/mean_ep_length": sum(lengths) / len(lengths),
            "train/mean_abs_cte": sum(self._ctes) / len(self._ctes),
            "train/mean_abs_heading_error": sum(self._heading_errors)
            / len(self._heading_errors),
            "global_step": self.num_timesteps,
        }
        self._ctes = []
        self._heading_errors = []
        wandb.log(metrics)


@hydra.main(version_base=None, config_path="configurations", config_name="train")
@helpers.prefill(key="wrappers", search_path="observations")
def train(configuration: omegaconf.DictConfig):
    print("[configuration]:", configuration)

    helpers.save_configuration(configuration, "train")

    output_directory = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    print("[output_directory]:", output_directory)

    env_configs = [
        ("cristal", configuration.cristal),
        ("straight", configuration.straight),
    ]

    train_environment = stable_baselines3.common.vec_env.DummyVecEnv(
        [
            lambda name=name, cfg=cfg: _make_env(
                cfg,
                configuration.reward,
                configuration.wrappers,
                os.path.join(output_directory, "logs", f"train_monitor_{name}.csv"),
            )
            for name, cfg in env_configs
        ]
    )
    train_environment = stable_baselines3.common.vec_env.VecVideoRecorder(
        train_environment,
        os.path.join(output_directory, "training_videos"),
        record_video_trigger=lambda step: step % 100_000 == 0,
        video_length=512,
    )

    eval_environment = stable_baselines3.common.vec_env.DummyVecEnv(
        [
            lambda name=name, cfg=cfg: _make_env(
                cfg,
                configuration.reward,
                configuration.wrappers,
                os.path.join(output_directory, "logs", f"eval_monitor_{name}.csv"),
            )
            for name, cfg in env_configs
        ]
    )
    eval_environment = stable_baselines3.common.vec_env.VecVideoRecorder(
        eval_environment,
        os.path.join(output_directory, "training_videos"),
        record_video_trigger=lambda step: step % 100_000 == 0,
        video_length=512,
    )

    authentication = wandb.login(
        key=os.environ.get("WANDB_API_KEY"), relogin=True, verify=True
    )

    print(f"[wandb]: authentication successful: {authentication}")

    run = wandb.init(
        project="research-project",
        config=omegaconf.OmegaConf.to_container(configuration, resolve=True),
        sync_tensorboard=True,
        save_code=True,
        dir=output_directory,
    )

    logger = stable_baselines3.common.logger.configure(
        os.path.join(output_directory, "logs"), ["stdout", "csv", "json", "tensorboard"]
    )
    controller = hydra.utils.instantiate(
        configuration.controller,
        environment=train_environment,
    )
    # NOTE: might not work with deterministic controllers as they don't have a .model
    controller.model.set_logger(logger)

    controller = controller.learn(
        total_timesteps=configuration.total_timesteps,
        progress_bar=True,
        callback=stable_baselines3.common.callbacks.CallbackList(
            [
                _TrainMetricsCallback(),
                stable_baselines3.common.callbacks.EvalCallback(
                    eval_env=eval_environment,
                    log_path=os.path.join(output_directory),
                    best_model_save_path=os.path.join(output_directory),
                    deterministic=True,
                ),
                wandb.integration.sb3.WandbCallback(
                    verbose=2,
                    model_save_path=os.path.join(output_directory, "models"),
                ),
            ]
        ),
    )


if __name__ == "__main__":
    train()
