import os

import hydra
import hydra.core
import hydra.core.hydra_config
import matplotlib.pyplot as plt
import omegaconf
import stable_baselines3

import helpers
import src.helpers as helpers
from evaluate import evaluate


@hydra.main(version_base=None, config_path="configurations", config_name="test")
@helpers.prefill(key="train.observations", search_path="observations")
@helpers.prefill(key="eval.observations", search_path="observations")
def test(configuration: omegaconf.DictConfig):
    print("[configuration]:", configuration)
    print("[config_path]:", helpers.get_config_path())
    print("[output]:", hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    # output_directory = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # helpers.save_configuration(configuration, "test")

    # print(configuration.observations)

    environment = hydra.utils.instantiate(
        configuration.straight, render_mode="rgb_array"
    )

    for i in range(10):
        environment.reset()
        plt.imshow(environment.render())
        plt.savefig(f"test_{i}.local.png")
    # environment = stable_baselines3.common.monitor.Monitor(
    #     env=environment,
    #     filename=os.path.join(output_directory, "logs", "monitor.csv"),
    # )

    # for wrapper in configuration.observations:
    #     print("[wrapper]:")
    #     print(wrapper)
    #     environment = hydra.utils.instantiate(
    #         wrapper,
    #         environment=environment
    #     )

    # print(environment.observation_space)


if __name__ == "__main__":
    test()
