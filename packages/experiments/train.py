import hydra
import omegaconf

from evaluate import evaluate

@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(configuration : omegaconf.DictConfig) -> None:
    environment = hydra.utils.instantiate(configuration.environment)
    
    controller = hydra.utils.instantiate(
        configuration.controller,
        environment=environment,
        model_kwargs={
            "learning_rate": 1e-4,
            "buffer_size": 100_000,
            "learning_starts": 10_000,
            "batch_size": 64,
            "gamma": 0.99,
            "exploration_fraction": 0.3,
            "exploration_final_eps": 0.05,
            "verbose": 1,
        }
    )
    
    controller = controller.learn(
        total_timesteps=configuration.total_timesteps,
        progress_bar=True
    )
    
    evaluate(
        controller.model,
        environment,
    )
    
if __name__ == "__main__":
    main()
