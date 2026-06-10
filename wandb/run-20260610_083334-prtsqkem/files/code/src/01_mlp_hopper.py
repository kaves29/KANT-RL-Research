import wandb
import torch
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

RAND_SEEDS = 10
for seed in range(RAND_SEEDS):
    # setting manual seeds to ensure reproducability
    torch.manual_seed(seed)
    np.random.seed(seed=seed)

    # organizing results in wandb with seed identifier for reproducibility
    run = wandb.init(
        project="kant_rl_project",
        name=f"MLP_Hopper_seed_{seed}",
        save_code=True
    )

    env = gym.make("Hopper-v4", )
    env.reset(seed=seed)

    model = PPO("MlpPolicy", env, verbose=0, seed=seed)
    model.learn(total_timesteps=1000000)

    model_path = f"models/mlp_hopper/mlp_hopper_{seed}" # seed is included with each file name to create unique files
    model.save(model_path)

    env.close()
    run.finish() # closing wandb file to stop and finalize recorded data
