import os
import hydra
import omegaconf
import stable_baselines3

import src.helpers as helpers

from evaluate import evaluate, plot_training_results
                             
@hydra.main(version_base=None, config_path="configurations", config_name="train")
@helpers.prefill(key="wrappers", search_path="observations")
def main(configuration: omegaconf.DictConfig):
    print("[configuration]:", configuration)
    
    helpers.save_configuration(configuration, "train")
    
    output_directory = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    print("[output_directory]:", output_directory)
        
    train_environment = hydra.utils.instantiate(configuration.train)
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
    
    eval_environment = hydra.utils.instantiate(configuration.eval)
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
    
    evaluation_callback = stable_baselines3.common.callbacks.EvalCallback(
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
        eval_environment,
        output_path=os.path.join(output_directory, "evaluations"),
    )

if __name__ == "__main__":
    main()
