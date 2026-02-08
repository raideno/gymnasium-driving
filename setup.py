from setuptools import setup, find_packages

setup(
    name="experiments",
    version="0.1.0",
    packages=find_packages(include=["gymnasium_driving*"]),
    install_requires=[
        "gymnasium>=1.2.3",
        "numpy>=2.4.1",
        "ipython>=9.10.0",
        "pygame>=2.6.1",
        "scipy>=1.17.0",
        "matplotlib>=3.10.8",
        "stable-baselines3[extra]>=2.7.1",
        "hydra-core>=1.3.2",
    ],
    dependency_links=[
        "git+https://github.com/winstxnhdw/KinematicBicycleModel#egg=kbm",
    ],
)