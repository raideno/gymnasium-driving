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
