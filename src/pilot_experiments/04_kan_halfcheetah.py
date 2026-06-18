import os
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import wandb
import torch
from torch import nn
from kan import KAN
import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import __version__
from wandb.integration.sb3 import WandbCallback

# Configuration & Hardware Acceleration
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
NUM_SEEDS = 10
TOTAL_TIMESTEPS = 100000
NUM_ENVS = 4
INPUT_DIMS = 17
ENVIRONMENT_NAME = "HalfCheetah-v4"
LOG_DIR = f"/Users/shouryakaveti/VS_Projects/kan-vs-mlp-rl-research/logs/pilot_experiment_logs/kan_halfcheetah_logs"
MODEL_DIR = f"/Users/shouryakaveti/VS_Projects/kan-vs-mlp-rl-research/models/pilot_models/kan_halfcheetah"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

wandb.tensorboard.patch(root_logdir="logs/")

class CustomKan(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super(CustomKan, self).__init__()

        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        
        # Policy network
        self.policy_net = KAN(width=[INPUT_DIMS, 64, 64])
        self.policy_net.speed()

        # Value network
        self.value_net = KAN(width=[INPUT_DIMS, 64, 64])
        self.value_net.speed()

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.policy_net(features), self.value_net(features)
    
    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)
    
    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)
    
class CustomKanActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):

        super(CustomKanActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            *args,
            **kwargs,
        )
        self.ortho_init = False
    
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomKan(self.features_dim)

for seed in range(NUM_SEEDS):
    torch.manual_seed(seed)
    np.random.seed(seed)

    run = wandb.init(
        project="kant_rl_project",
        group="KAN_HalfCheetah_seeds",
        name=f"KAN_HalfCheetah_seed_{seed}",
        sync_tensorboard=True, 
        save_code=True,
        config={
            "environment": ENVIRONMENT_NAME,
            "total_timesteps": TOTAL_TIMESTEPS,
            "num_envs": NUM_ENVS,
            "seed": seed,
            "device": str(DEVICE),
            "architecture": f"KAN [{INPUT_DIMS}, 64, 64]",
            "torch_version": torch.__version__,
            "pykan_version": "0.2.8",
            "sb3_version": __version__
        }
    )

    env = make_vec_env(ENVIRONMENT_NAME, n_envs=NUM_ENVS, seed=seed)
    model = PPO(CustomKanActorCriticPolicy, env, verbose=0, tensorboard_log=f"{LOG_DIR}/kan_halfcheetah_seed_{seed}")

    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=WandbCallback())
        model.save(f"{MODEL_DIR}/kan_halfcheetah_seed_{seed}")
    except Exception as e:
        print(f"CRITICAL: Code failure executing seed {seed}")
        print(f"Error: {e}")
    finally:
        env.close()
        run.finish() # closing wandb file to stop and finalize recorded data

print("All seeds executed successfully.")

    