import hydra
import omegaconf

from evaluate import evaluate

@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(configuration : omegaconf.DictConfig) -> None:
    environment = hydra.utils.instantiate(configuration.environment)
    
    controller = hydra.utils.instantiate(
        configuration.controller,
        environment=environment
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
