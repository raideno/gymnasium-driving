import os
import wandb
import wandb.integration
import wandb.integration.sb3
import hydra
import dotenv
import omegaconf
import hydra.core
import hydra.core.hydra_config
import stable_baselines3
import stable_baselines3.common
import stable_baselines3.common.monitor
import stable_baselines3.common.logger
import stable_baselines3.common.vec_env
import stable_baselines3.common.callbacks

import src.helpers as helpers

print("[.env]:", dotenv.load_dotenv(dotenv_path=".env"))

@hydra.main(version_base=None, config_path="configurations", config_name="train")
@helpers.prefill(key="wrappers", search_path="observations")
def train(configuration: omegaconf.DictConfig):
    print("[configuration]:", configuration)
    
    helpers.save_configuration(configuration, "train")
    
    output_directory = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    print("[output_directory]:", output_directory)
        
    train_environment = hydra.utils.instantiate(configuration.train, render_mode="rgb_array")
    train_environment = hydra.utils.instantiate(
        configuration.reward,
        environment=train_environment,
    )
    for wrapper in configuration.wrappers:
        train_environment = hydra.utils.instantiate(
            wrapper,
            environment=train_environment
        )
    train_environment = stable_baselines3.common.monitor.Monitor(
        env=train_environment,
        filename=os.path.join(output_directory, "logs", "train_monitor.csv"),
    )
    train_environment = stable_baselines3.common.vec_env.DummyVecEnv([lambda: train_environment])
    train_environment = stable_baselines3.common.vec_env.VecVideoRecorder(
        train_environment,
        os.path.join(output_directory, "training_videos"),
        record_video_trigger=lambda step: step % 100_000 == 0,
        video_length=200,
    )
    
    eval_environment = hydra.utils.instantiate(configuration.eval, render_mode="rgb_array")
    eval_environment = hydra.utils.instantiate(
        configuration.reward,
        environment=eval_environment,
    )
    for wrapper in configuration.wrappers:
        eval_environment = hydra.utils.instantiate(
            wrapper,
            environment=eval_environment
        )
    eval_environment = stable_baselines3.common.monitor.Monitor(
        env=eval_environment,
        filename=os.path.join(output_directory, "logs", "eval_monitor.csv"),
    )
    eval_environment = stable_baselines3.common.vec_env.DummyVecEnv([lambda: eval_environment])
    eval_environment = stable_baselines3.common.vec_env.VecVideoRecorder(
        eval_environment,
        os.path.join(output_directory, "evaluation_videos"),
        record_video_trigger=lambda episode: episode % 100_000 == 0,
        video_length=200,
    )
    
    logger = stable_baselines3.common.logger.configure(
        os.path.join(output_directory, "logs"),
        ["stdout", "csv", "json", "tensorboard"]
    )
    
    controller = hydra.utils.instantiate(
        configuration.controller,
        environment=train_environment,
    )
    
    # NOTE: might not work with deterministic controllers as they don't have a .model
    controller.model.set_logger(logger)
    
    authentication = wandb.login(key=os.environ.get("WANDB_API_KEY"), relogin=True, verify=True)
    
    print(f"[wandb]: authentication successful: {authentication}")
    
    run = wandb.init(
        project="research-project",
        config=omegaconf.OmegaConf.to_container(configuration, resolve=True),
        # sync_tensorboard=True,
        save_code=True,
        dir=output_directory,
    )
    
    controller = controller.learn(
        total_timesteps=configuration.total_timesteps,
        progress_bar=True,
        callback=stable_baselines3.common.callbacks.CallbackList([
            stable_baselines3.common.callbacks.EvalCallback(
                eval_env=eval_environment,
                n_eval_episodes=5,
                eval_freq=10_000,
                # (str | None) – Path to a folder where the evaluations (evaluations.npz) will be saved. It will be updated at each evaluation.
                # NOTE: .npz is a numpy archive file that can be loaded with np.load() to access the evaluation results.
                log_path=os.path.join(output_directory),
                best_model_save_path=os.path.join(output_directory),
                deterministic=True,
                # (bool) – Whether the evaluation should use a stochastic or deterministic actions.
                # NOTE: if False, the agent will use the same action as during training (which might be stochastic if the policy is stochastic). Sampling.
                # If True, the agent will use the deterministic version of the policy (if it exists). Greedy.
                # deterministic,
            ),
            wandb.integration.sb3.WandbCallback(
                verbose=2,
                gradient_save_freq=100,
                model_save_path=os.path.join(output_directory, "models"),
                model_save_freq=50_000,
            )
        ]),
        # "log_interval": 100, # log every 100 episodes
    )
    
if __name__ == "__main__":
    train()
