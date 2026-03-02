# Gymnasium Driving Environment

To install the environment and use it in your own project:
```bash
pip install git+https://github.com/raideno/gymnasium-driving-environment.git
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# TODO: remove the bicycle model dependency, see if there is any impact on performance.
.venv/bin/pip install git+https://github.com/winstxnhdw/KinematicBicycleModel.git
```

## Environment Development

You can find all the code related to the environment inside [`gymnasium_driving/`](./gymnasium_driving/) directory.

## RL Training

Make sure you have correctly setup the environment first.

```bash
HYDRA_FULL_ERROR=1 python train.py \
   	environment=discrete \
   	controller=dqn \
   	reward=path_progress \
   	total_timesteps=1000000 \
   	wrappers=\[with_path_info,with_obstacles_info,with_base_info,with_road_info\]
```

You can replace the environment, controller and the other parameters with the ones you want, you can find all available options inside the corresponding directory at [`configurations`](./configurations/).
