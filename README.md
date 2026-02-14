# Gymnasium Driving Environment

```bash
pip install git+https://github.com/raideno/gymnasium-driving-environment.git
```

```bash
pip install -e .
```

--- --- ---

**Train a path following RL algorithm:**
```bash
HYDRA_FULL_ERROR=1 python train.py \
    environment@train=cristal \
    environment@eval=cristal \
    train.discrete=true \
    train.number_of_obstacles=0 \
    eval.number_of_obstacles=0  \
    eval.discrete=True \
    total_timesteps=1000000 \
    reward=path_progress \
    controller=dqn
```

**Train a obstacle avoidance path following RL algorithm:**
```bash
HYDRA_FULL_ERROR=1 python train.py \
  environment@train=straight \
  controller=dqn \
  reward=path_progress_obstacles \
  total_timesteps=1000000 \
  train.discrete=true \
  environment@eval=straight \
  eval.discrete=true
```

--- --- ---

- PPO. Plusieurs type de trajectoire, une ligne direct et une sin.
- In isaac lab they use skrl library with ppo algorithm.

- A la prochaine réunion on selectionne sur quel environements je travaille.

- [ ] Zone de sécurité autour de l'obstacle pour la reward function. Selon la largeur de route, etc.

- [ ] Generalization.

--- ---

- [ ] Faire en sorte que l'algo generalie.
- [ ] Implementer les differentes autres trajectoires.
