import wandb
import torch
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from wandb.integration.sb3 import WandbCallback

RAND_SEEDS = 10
for seed in range(RAND_SEEDS):
    # setting manual seeds to ensure reproducability
    torch.manual_seed(seed)
    np.random.seed(seed=seed)

    # organizing results in wandb with seed identifier for reproducibility
    run = wandb.init(
        project="kant_rl_project",
        group="MLP_Hopper_seeds",
        name=f"MLP_Hopper_seed_{seed}",
        sync_tensorboard=True, 
        save_code=True
    )

    env = make_vec_env("Hopper-v4", n_envs=4, seed=seed)

    model = PPO("MlpPolicy", env, seed=seed, verbose=0, tensorboard_log=f"logs/pilot_experiment_logs/mlp_hopper_logs/mlp_hopper_seed_{seed}")
    model.learn(total_timesteps=1000000, callback=WandbCallback())

    model_path = f"models/pilot_models/mlp_hopper/mlp_hopper_{seed}" # seed is included with each file name to create unique files
    model.save(model_path)

    env.close()
    run.finish() # closing wandb file to stop and finalize recorded data
