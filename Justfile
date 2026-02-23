set shell := ["bash","-cu"]

activate := "source .venv/bin/activate"

# npx repomix \
#     --ignore "\
#         **.ipynb,\
#         src/controllers/clothoids.py,\
#         gymnasium_driving/components/roads.py,\
#         gymnasium_driving/components/renderer.py,\
#         gymnasium_driving/components/performance.py
#     "

pack:
	npx repomix --ignore "notebooks/*" --output code.local.xml

train-straight:
	HYDRA_FULL_ERROR=1 python train.py \
		environment@train=straight \
		environment@eval=straight \
		train.discrete=true \
		eval.discrete=true \
		controller=dqn \
		reward=path_progress \
		total_timesteps=1000000 \
		wrappers=\[with_path_info,with_obstacles_info,with_base_info,with_road_info,random_obstacles\]

train-cristal:
	HYDRA_FULL_ERROR=1 python train.py \
		environment@train=cristal \
		environment@eval=cristal \
		train.discrete=true \
		eval.discrete=true \
		controller=dqn \
		reward=path_progress \
		total_timesteps=1000000 \
		wrappers=\[with_path_info,with_obstacles_info,with_base_info,with_road_info\]
