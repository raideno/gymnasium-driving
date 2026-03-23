# Training

For environment usage and extension details, see [`gymnasium_driving/README.md`](./gymnasium_driving/README.md).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

`pip install -e .` will look at the [`pyproject.toml`](./pyproject.toml) and install all the required libraries, there is also a [`requirements.txt`](./requirements.txt).

Then edit `.env` and set a valid `WANDB_API_KEY` (training logs are sent to Weights & Biases).

## Training architecture

Training entrypoint is `train.py`, configured by Hydra from `configurations/train.yaml`.

At runtime, the training pipeline is:

- Instantiate two environments (`cristal` and `straight`) from the selected environment config.
- Wrap each env with the configured reward wrapper.
- Apply observation wrappers from `wrappers=[...]` (resolved by `helpers.prefill` from `configurations/observations/`).
- Add `Monitor` logging and combine both envs in a `DummyVecEnv`.
- Record periodic videos with `VecVideoRecorder`.
- Instantiate the selected controller (`PPO`, `DQN`, `TRPO`) from `configurations/controller/*.yaml`.
- Train via Stable-Baselines3 with callbacks for custom metrics, evaluation, and model checkpointing.

## Run training

```bash
HYDRA_FULL_ERROR=1 .venv/bin/python train.py \
  environment=continuous \
  controller=ppo-continuous \
  reward=reward \
  total_timesteps=1000000 \
  wrappers=\[with_path_info,with_obstacles_info,with_base_info,with_road_info\]
```

Available config groups:

- `environment`: `continuous`, `discrete`
- `controller`: `ppo-continuous`, `ppo-discrete`, `dqn`, `trpo-discrete`
- `wrappers` items: `with_path_info`, `with_obstacles_info`, `with_base_info`, `with_road_info`

## Hydra usage and customization

Hydra composes config from `configurations/train.yaml` + overrides you pass on the CLI.

Useful patterns:

```bash
# Change algorithm hyperparameters
HYDRA_FULL_ERROR=1 .venv/bin/python train.py \
  environment=continuous \
  controller=ppo-continuous \
  reward=reward \
  total_timesteps=1000000 \
  wrappers=\[with_path_info,with_obstacles_info,with_base_info,with_road_info\] \
  controller.model_kwargs.learning_rate=0.0001 \
  controller.model_kwargs.batch_size=128

# Change training duration / seed
HYDRA_FULL_ERROR=1 .venv/bin/python train.py \
  environment=continuous \
  controller=ppo-continuous \
  reward=reward \
  wrappers=\[with_path_info,with_obstacles_info,with_base_info,with_road_info\] \
  total_timesteps=2000000 \
  seed=42

# Print fully resolved config
HYDRA_FULL_ERROR=1 .venv/bin/python train.py --cfg job --resolve
```

## Where to modify things

- Algorithm/hyperparameters: `configurations/controller/*.yaml` and `src/controllers/*.py`
- Scenario generation (roads, obstacles, spawn/goal): `src/environments/*.py`
- Observation schema: `gymnasium_driving/wrappers/observations/*.py` and `configurations/observations/*.yaml`
- Reward shaping: `gymnasium_driving/wrappers/rewards/reward.py` and `configurations/reward/reward.yaml`

To add a new wrapper/controller/environment option:

1. Implement code.
2. Add its Hydra YAML config in the matching `configurations/<group>/` folder.
3. Select it through CLI overrides.
