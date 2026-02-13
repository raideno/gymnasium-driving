from PIL import Image as PILImage
from IPython.display import display, clear_output

# NOTE: render_mode must be set to rgb_array

def preview(
    environment,
    clear: bool = True,
):
    image = environment.render()
    if clear:
        clear_output(wait=True)
    
    pil_img = PILImage.fromarray(image)
    
    display(pil_img)

import os
import json
import hydra
import omegaconf

def save_configuration(
    configuration: omegaconf.DictConfig,
    script: str | None = None
):
    output_directory = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    configuration = omegaconf.OmegaConf.to_container(configuration, resolve=True)
    
    with open(os.path.join(output_directory, "configuration.json"), "w") as file:
        json.dump(configuration, file, indent=4)
    
    if script is not None:
        with open(os.path.join(output_directory, "script"), "w") as file:
            file.write(script)

def load_configuration(
    output_directory: str,
    expected_script: str | None = None,
) -> omegaconf.DictConfig:
    with open(os.path.join(output_directory, "configuration.json"), "r") as file:
        configuration = json.load(file)
    
    if expected_script is not None:
        with open(os.path.join(output_directory, "script"), "r") as file:
            script = file.read()
        
        if script != expected_script:
            raise ValueError(f"Expected script does not match the one in the output directory. Expected: {expected_script}, Found: {script}")
    
    configuration = omegaconf.OmegaConf.create(configuration)
    
    return configuration

def instantiate_configuration(
    configuration: omegaconf.DictConfig,
    output_directory: str | None = None,
    load_best_model: bool = True,
):
    """
    Instantiate environment, reward wrapper, and controller from configuration.
    
    Args:
        configuration: Hydra configuration containing environment, reward, and controller specs
        output_directory: Path to directory containing saved models (optional)
        load_best_model: Whether to load the best model if it exists
    
    Returns:
        Tuple of (controller, environment)
    """
    train_environment = hydra.utils.instantiate(configuration.train)
    train_environment = hydra.utils.instantiate(
        configuration.reward,
        environment=train_environment,
    )
    
    eval_environment = hydra.utils.instantiate(configuration.train)
    eval_environment = hydra.utils.instantiate(
        configuration.reward,
        environment=eval_environment,
    )
    
    controller = hydra.utils.instantiate(
        configuration.controller,
        environment=train_environment,
    )
    
    if load_best_model and output_directory is not None:
        best_model_path = os.path.join(output_directory, "best_model.zip")
        if os.path.exists(best_model_path):
            controller.model = controller.model.load(best_model_path, env=train_environment)
    
    return controller, train_environment, eval_environment

def get_last_run_directory(
    base_directory: str,
    script: str | None = None,
):
    day_directories = sorted(os.listdir(base_directory))
    
    if not day_directories:
        raise ValueError(f"No runs found in base directory: {base_directory}")
    
    day_directory = os.path.join(base_directory, day_directories[-1])
    run_directories = sorted(os.listdir(day_directory))
    
    if not run_directories:
        raise ValueError(f"No runs found in day directory: {day_directory}")
    
    for run_dir in reversed(run_directories):
        full_path = os.path.join(day_directory, run_dir)
        
        if script is None:
            return full_path
        
        script_path = os.path.join(full_path, "script")
        if os.path.exists(script_path):
            with open(script_path, "r") as file:
                if file.read() == script:
                    return full_path
    
    raise ValueError(f"No matching run found for script: {script}")
