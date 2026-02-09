import os
import csv
import hydra
import omegaconf
import stable_baselines3

from src.helpers import save_configuration

from evaluate import evaluate, plot_training_results

@hydra.main(version_base=None, config_path="configurations", config_name="train")
def main(configuration: omegaconf.DictConfig):
    save_configuration(configuration, "train")
    
    output_directory = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    print("[output_directory]:", output_directory)
    
        
    environment = hydra.utils.instantiate(configuration.environment)
    environment = hydra.utils.instantiate(
        configuration.reward,
        environment=environment,
    )
    environment = stable_baselines3.common.monitor.Monitor(
        env=environment,
        filename=os.path.join(output_directory, "logs", "monitor.csv"),
    )
    
    logger = stable_baselines3.common.logger.configure(
        os.path.join(output_directory, "logs"),
        ["stdout", "csv", "json", "tensorboard"]
    )
    
    controller = hydra.utils.instantiate(
        configuration.controller,
        environment=environment,
    )
    
    # NOTE: might not work with deterministic controllers as they don't have a .model
    controller.model.set_logger(logger)
    
    # TODO: make the evaluation on an environment with a static seed or a static obstacle
    evaluation_callback = stable_baselines3.common.callbacks.EvalCallback(
        eval_env=environment,
        n_eval_episodes=5,
        eval_freq=10_000,
        # (str | None) – Path to a folder where the evaluations (evaluations.npz) will be saved. It will be updated at each evaluation.
        # NOTE: .npz is a numpy archive file that can be loaded with np.load() to access the evaluation results.
        log_path=os.path.join(output_directory),
        best_model_save_path=os.path.join(output_directory),

        # (bool) – Whether the evaluation should use a stochastic or deterministic actions.
        # NOTE: if False, the agent will use the same action as during training (which might be stochastic if the policy is stochastic). Sampling.
        # If True, the agent will use the deterministic version of the policy (if it exists). Greedy.
        # deterministic,
    )
    
    controller = controller.learn(
        total_timesteps=configuration.total_timesteps,
        progress_bar=True,
        callback=stable_baselines3.common.callbacks.CallbackList([
            evaluation_callback,
        ]),
        # "log_interval": 100, # log every 100 episodes
    )
    
    os.makedirs(os.path.join(output_directory, "evaluations"), exist_ok=True)
    
    log_dir = os.path.join(output_directory, "logs")
    
    plot_training_results(log_dir, os.path.join(output_directory, "evaluations"))
    
    evaluate(
        controller.model,
        environment,
        output_path=os.path.join(output_directory, "evaluations"),
    )

if __name__ == "__main__":
    main()
