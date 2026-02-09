import os
import hydra
import omegaconf

import stable_baselines3
from src.helpers import save_configuration

from evaluate import evaluate

@hydra.main(version_base=None, config_path="configurations", config_name="test")
def main(configuration: omegaconf.DictConfig):
    print("[configuration]:", configuration)
    print("[output]:", hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    
    output_directory = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    save_configuration(configuration, "test")
    
    print(configuration.observations)
    
    environment = hydra.utils.instantiate(configuration.environment)
    environment = stable_baselines3.common.monitor.Monitor(
        env=environment,
        filename=os.path.join(output_directory, "logs", "monitor.csv"),
    )
    
    for wrapper in configuration.observations:
        print("[wrapper]:")
        print(wrapper)
        environment = hydra.utils.instantiate(
            wrapper,
            environment=environment
        )
        
    print(environment.observation_space)
    
if __name__ == "__main__":
    main()
