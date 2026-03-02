set shell := ["bash", "-cu"]

python := ".venv/bin/python"

setup:
    python -m venv .venv
    .venv/bin/pip install -e .
    .venv/bin/pip install git+https://github.com/winstxnhdw/KinematicBicycleModel.git

pack:
    npx repomix --ignore "notebooks/*" --output code.local.xml

train total_timesteps="1000000" controller="dqn" reward="path_progress":
    HYDRA_FULL_ERROR=1 {{ python }} train.py \
    	environment=discrete \
    	controller={{ controller }} \
    	reward={{ reward }} \
    	total_timesteps={{ total_timesteps }} \
    	wrappers=\[with_path_info,with_obstacles_info,with_base_info,with_road_info\]

evaluate output_path:
    HYDRA_FULL_ERROR=1 {{ python }} evaluate.py \
    	output_path={{ output_path }}

evaluate-controller output_path controller:
    HYDRA_FULL_ERROR=1 {{ python }} evaluate.py \
    	output_path={{ output_path }} \
    	controller={{ controller }}

clean:
    rm -rf outputs
