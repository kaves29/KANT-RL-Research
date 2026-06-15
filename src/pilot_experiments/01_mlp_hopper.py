import os
import wandb
import torch
import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import __version__
from wandb.integration.sb3 import WandbCallback


# Configuration & Hardware Acceleration
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
NUM_SEEDS = 10
TOTAL_TIMESTEPS = 100000
NUM_ENVS = 4
ENVIRONMENT_NAME = "Hopper-v4"
LOG_DIR = f"/Users/shouryakaveti/VS_Projects/kan-vs-mlp-rl-research/logs/pilot_experiment_logs/mlp_hopper_logs"
MODEL_DIR = f"/Users/shouryakaveti/VS_Projects/kan-vs-mlp-rl-research/models/pilot_models/mlp_hopper"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

wandb.tensorboard.patch(root_logdir="logs/")

print(f"Beginning Pilot Experiment on Device: {DEVICE}")
print(f"Environment: {ENVIRONMENT_NAME} | Seeds: {NUM_SEEDS} \n")

for seed in range(NUM_SEEDS):
    # setting manual seeds to ensure reproducability
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Creating seed specific directories
    seed_log_dir = f"{LOG_DIR}/mlp_hopper_seed_{seed}"
    os.makedirs(seed_log_dir, exist_ok=True)

    # organizing results in wandb with seed identifier for reproducibility
    run = wandb.init(
        project="kant_rl_project",
        group="MLP_Hopper_seeds",
        name=f"MLP_Hopper_seed_{seed}",
        sync_tensorboard=True, 
        save_code=True,
        config={
            "environment": ENVIRONMENT_NAME,
            "total_timesteps": TOTAL_TIMESTEPS,
            "num_envs": NUM_ENVS,
            "seed": seed,
            "device": str(DEVICE),
            "architecture": "MLP (SB3 Default)",
            "torch_version": torch.__version__,
            "pykan_version": "0.2.8",
            "sb3_version": __version__
        }
    )

    env = make_vec_env(ENVIRONMENT_NAME, n_envs=NUM_ENVS, seed=seed)
    
    model = PPO("MlpPolicy", env, seed=seed, verbose=0, tensorboard_log=seed_log_dir, device="cpu")
    
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=WandbCallback())
        model.save(f"{MODEL_DIR}/mlp_hopper_{seed}")
    except Exception as e:
        print(f"CRITICAL: Code failure executing seed {seed}")
        print(f"Error: {e}")
    finally:
        env.close()
        run.finish() # closing wandb file to stop and finalize recorded data

print("All seeds executed successfully.")
