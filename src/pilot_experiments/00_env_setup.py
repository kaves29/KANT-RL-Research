import numpy as np
import torch
import gymnasium as gym
import networkx
import kan
import ruptures
import codecarbon
import pingouin
import wandb
import statsmodels
from stable_baselines3 import PPO
from stable_baselines3 import __version__
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

env_hopper = make_vec_env("Hopper-v4", n_envs=4)
env_cheetah = make_vec_env("HalfCheetah-v4", n_envs=4)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print(f"Device: {device}")
print(f"Gym Version: {gym.__version__}")
print(f"SB3 Version: {__version__}")
print(f"Pytorch Version: {torch.__version__}")

