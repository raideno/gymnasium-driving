# Gymnasium Driving Environment

To install the environment:
```bash
pip install git+https://github.com/raideno/gymnasium-driving-environment.git
```

## Environment Development

You can find all the code related to the environment inside [`gymnasium_driving/`](./gymnasium_driving/) directory.

## RL Training

First you need to have [Just](https://github.com/casey/just) installed in your computer.

1. Setup the python environment and required libraries:
```bash
just setup
```

2. Start a training:
```bash
just train-straight
# or
just train-cristal
```
