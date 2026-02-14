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
import sys

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
    
    with open(os.path.join(output_directory, "command"), "w") as file:
        file.write(" ".join(sys.argv))

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
    base_dir: str = "../configurations"
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
    
    if "wrappers" in configuration.keys():
        apply_prefill(
            configuration=configuration,
            key="wrappers",
            search_path="observations",
            raise_if_missing=True,
            base_dir=base_dir
        )
        
        for wrapper in configuration.wrappers:
            train_environment = hydra.utils.instantiate(
                wrapper,
                environment=train_environment
            )
        
        for wrapper in configuration.wrappers:
            eval_environment = hydra.utils.instantiate(
                wrapper,
                environment=eval_environment
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

import os
import typing
import functools

import hydra
import omegaconf


def get_config_path() -> str:
    config_sources = hydra.core.hydra_config.HydraConfig.get().runtime.config_sources

    if not config_sources:
        raise ValueError("No configuration sources found.")

    for config_source in config_sources:
        if config_source.provider == "main":
            return config_source.path

    raise ValueError("Main configuration source not found.")


def _split_path(path: str) -> list[str]:
    if not isinstance(path, str) or not path.strip():
        raise ValueError("key must be a non-empty string")
    return [p for p in path.split(".") if p]


def _dot_to_fs_path(dot_path: str) -> str:
    return os.path.join(*_split_path(dot_path))


def _get_container_and_leaf(
    configuration: omegaconf.DictConfig, path: str
) -> tuple[typing.Any, str]:
    parts = _split_path(path)
    if len(parts) == 1:
        return configuration, parts[0]

    container: typing.Any = configuration
    for part in parts[:-1]:
        if not (isinstance(container, (dict, omegaconf.DictConfig)) and part in container):
            raise KeyError(f"Key path '{path}' not found (missing '{part}').")
        container = container[part]

    leaf = parts[-1]
    return container, leaf


def _load_prefill(
    configuration: omegaconf.DictConfig,
    key_path: str,
    base_dir: str,
    search_path: str | None,
) -> list[omegaconf.DictConfig]:
    container, leaf = _get_container_and_leaf(configuration, key_path)

    if not (isinstance(container, (dict, omegaconf.DictConfig)) and leaf in container):
        raise KeyError(f"Key '{key_path}' not found in configuration.")

    value = container[leaf]
    if not isinstance(value, (list, omegaconf.ListConfig)):
        raise ValueError(
            f"Expected a list for key '{key_path}', but got {type(value).__name__}."
        )

    relative_dir = _dot_to_fs_path(key_path) if search_path is None else search_path
    target_dir = os.path.join(base_dir, relative_dir)

    prefilled: list[omegaconf.DictConfig] = []
    for item in value:
        prefilled.append(
            omegaconf.OmegaConf.load(os.path.join(target_dir, f"{item}.yaml"))
        )

    return prefilled


def apply_prefill(
    configuration: omegaconf.DictConfig,
    key: str,
    *,
    search_path: str | None = None,
    raise_if_missing: bool = False,
    base_dir: str | None = None,
) -> omegaconf.DictConfig:
    """Apply the prefill operation directly (non-decorator API).

    Replaces the list at `key` with loaded configs.

    Args:
        configuration: Hydra/OmegaConf DictConfig to mutate.
        key: Dot-path key to prefill (e.g. "environments", "reward.observations").
        search_path: If None, loads from `<base_dir>/<key-as-path>/`.
                    If provided, loads from `<base_dir>/<search_path>/`.
        raise_if_missing: If True, missing keys raise; otherwise no-op.
        base_dir: If provided, overrides `get_config_path()` for locating YAMLs.

    Returns:
        The (mutated) configuration, for convenience.
    """
    effective_base_dir = get_config_path() if base_dir is None else base_dir

    try:
        container, leaf = _get_container_and_leaf(configuration, key)
    except KeyError:
        if raise_if_missing:
            raise
        return configuration

    if not (isinstance(container, (dict, omegaconf.DictConfig)) and leaf in container):
        if raise_if_missing:
            raise KeyError(f"Key '{key}' not found in configuration.")
        return configuration

    container[leaf] = _load_prefill(
        configuration=configuration,
        base_dir=effective_base_dir,
        key_path=key,
        search_path=search_path,
    )
    return configuration


def prefill(
    key: str,
    search_path: str | None = None,
    raise_if_missing: bool = False,
):
    """Decorator factory for pre-loading environment configurations.

    - `key` supports nested dot-path keys, e.g.:
        - @prefill("environments")
        - @prefill("reward.observations")

    - `search_path` controls where YAML files are loaded from:
        - If None (default): loads from `get_config_path()/<key-as-path>/`,
          where <key-as-path> is `key` with dots replaced by path separators.
          Example: key="reward.observations" => ".../reward/observations/<item>.yaml"
        - If provided: loads from `get_config_path()/search_path/<item>.yaml"

    - `raise_if_missing` controls missing key behavior:
        - If False (default): skip prefill when `key` is missing.
        - If True: raise an exception when `key` is missing.

    Replaces the list at `key` with loaded configs before calling the function.
    """

    def decorator(function: typing.Callable):
        @functools.wraps(function)
        def wrapper(configuration: omegaconf.DictConfig, *args, **kwargs):
            apply_prefill(
                configuration,
                key,
                search_path=search_path,
                raise_if_missing=raise_if_missing,
            )
            return function(configuration, *args, **kwargs)

        return wrapper

    return decorator