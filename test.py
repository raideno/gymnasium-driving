import os

import hydra
import hydra.core
import hydra.core.hydra_config
import matplotlib.pyplot as plt
import omegaconf
import stable_baselines3

import src.helpers as helpers
from evaluate import evaluate


def _predict(controller, observation):
    action, _ = controller.predict(observation, deterministic=True)
    return action


@hydra.main(version_base=None, config_path="configurations", config_name="test")
@helpers.prefill(key="train.observations", search_path="observations")
@helpers.prefill(key="eval.observations", search_path="observations")
def test(configuration: omegaconf.DictConfig):
    print("[configuration]:", configuration)
    print("[config_path]:", helpers.get_config_path())
    print("[output]:", hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    output_directory = configuration.get("output_path", None)

    if output_directory is not None and os.path.exists(output_directory):
        target_configuration = helpers.load_configuration(
            output_directory=output_directory,
            expected_script="train",
        )

        best_model_path = os.path.join(output_directory, "best_model.zip")

        controller, environment, _ = helpers.instantiate_configuration(
            configuration=target_configuration,
            output_directory=output_directory,
            load_best_model=os.path.exists(best_model_path),
            base_dir="./configurations",
        )

        # observation, _ = environment.reset()
        # action = _predict(controller, observation)
        # print("[action]:", action)

    else:
        environment = hydra.utils.instantiate(
            configuration.straight, render_mode="rgb_array"
        )

        for i in range(10):
            environment.reset()
            plt.imshow(environment.render())
            plt.savefig(f"test_{i}.local.png")


if __name__ == "__main__":
    test()
