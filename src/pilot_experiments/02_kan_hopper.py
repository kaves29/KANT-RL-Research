import wandb
import torch
from torch import nn
from kan import KAN
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy
from wandb.integration.sb3 import WandbCallback
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

device = "mps" if torch.backends.mps.is_available() else "cpu"

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
        self.policy_net = KAN(width=[11, 64, 64]) # 11: Takes 11 dimensions from Hopper | 64 (middle): Outputs per hidden layer | 64 (last): Outputs for the entire network (must match last_layer_dim_pi)
        self.policy_net.speed()

        # Value network
        self.value_net = KAN(width=[11, 64, 64])
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

RAND_SEEDS = 10
for seed in range(RAND_SEEDS):
    torch.manual_seed(seed)
    np.random.seed(seed)

    run = wandb.init(
        project="kant_rl_project",
        group="KAN_Hopper_seeds",
        name=f"KAN_Hopper_seed_{seed}",
        sync_tensorboard=True, 
        save_code=True
    )

    env = make_vec_env("Hopper-v4", n_envs=4, seed=seed)

    model = PPO(CustomKanActorCriticPolicy, env, verbose=0, tensorboard_log=f"/Users/shouryakaveti/VS_Projects/kan-vs-mlp-rl-research/logs/pilot_experiment_logs/kan_hopper_logs/kan_hopper_seed_{seed}")
    model.learn(total_timesteps=1000000, callback=WandbCallback())

    model_path = f"/Users/shouryakaveti/VS_Projects/kan-vs-mlp-rl-research/models/pilot_models/kan_hopper/kan_hopper_seed_{seed}" # seed is included with each file name to create unique files
    model.save(model_path)

    env.close()
    run.finish() # closing wandb file to stop and finalize recorded data

    