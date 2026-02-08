import hydra
import omegaconf

from src.helpers import save_configuration

from evaluate import evaluate

@hydra.main(version_base=None, config_path="configurations", config_name="test")
def main(configuration: omegaconf.DictConfig):
    print("[configuration]:", configuration)
    print("[output]:", hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    
    save_configuration(configuration, "test")
    
if __name__ == "__main__":
    main()
