import os
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import wandb
import torch
from torch import nn
from kan import KAN
import gymnasium as gym
import numpy as np
import statistics
import random

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import __version__
from wandb.integration.sb3 import WandbCallback

# Configuration & Hardware Acceleration
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
NUM_SEEDS = 5
TOTAL_TIMESTEPS = 50000
NUM_ENVS = 4
INPUT_DIMS = 4
ENVIRONMENT_NAME = "CartPole-v4"
LOG_DIR = f"logs/validation/cartpole_exp1a_kan"
MODEL_DIR = f"models/validation/cartpole_exp1a_kan"
EPSILON = 1e-2

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

wandb.tensorboard.patch(root_logdir="logs/")

def get_cartpole_obs(num_samples=1000):
    env = gym.make("CartPole-v4")
    observations = []

    for sample in range(num_samples):
        obs, info = env.reset()
        observations.append(obs)

        num_rand_rollout = 5
        for rollout in range(num_rand_rollout):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            observations.append(obs)

            if terminated or truncated:
                break

    env.close()
    return np.array(observations)

def compute_jaccard(seed_a, seed_b):
    intersection_count = 0
    union_count = 0

    for mask_idx in range(len(seed_a)):
        if(seed_a[mask_idx] == 1 and seed_b[mask_idx] == 1):
            intersection_count += 1
            union_count += 1

        if(seed_a[mask_idx] == 1 or seed_b[mask_idx] == 1):
            union_count += 1
    
    return intersection_count / union_count

def sparsity_score(kan_mask):
    active_splines = 0

    for mask_idx in range(len(kan_mask)):
        if(kan_mask[mask_idx] == 1):
            active_splines += 1
    
    return active_splines / len(kan_mask)
    

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
        group="Validation_Exp1A_KAN_CartPole_seeds",
        name=f"KAN_seed_{seed}",
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
    model = PPO(CustomKanActorCriticPolicy, env, verbose=0, tensorboard_log=f"{LOG_DIR}/kan_seed_{seed}")

    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=WandbCallback())
        model.save(f"{MODEL_DIR}/kan_seed_{seed}")
    except Exception as e:
        print(f"CRITICAL: Code failure executing seed {seed}")
        print(f"Error: {e}")
    finally:
        env.close()
        run.finish() # closing wandb file to stop and finalize recorded data

print("All seeds executed successfully.")

kan_topologies = []
for seed in range(NUM_SEEDS):
    seed_mask = []
    kan_model = torch.load(f"{MODEL_DIR}/kan_seed_{seed}")
    """ Determine an eplison value where if the spline variance is below
    eplison we 'prune' it with a binary mask by labeling it with a 0. If the 
    eplison value is above the thresold we label it 1. """
    
    obs = get_cartpole_obs(num_samples=1000)
    obs_tensor = torch.tensor(obs)

    output = kan_model(obs_tensor)

    for layer_idx, layer_tensor in enumerate(kan_model.edge_scores):
        out_dims, in_dims = kan_model.shape

        for output_node in range(out_dims):
            for input_node in range(in_dims):
                edge_score = layer_tensor[output_node, input_node].item()
                
                mask = (edge_score > EPSILON).int()
                seed_mask.append(mask)
    
    kan_topologies.append(seed_mask)

jaccard_scores = []
for seed_a_idx in range(NUM_SEEDS):
    for seed_b_idx in range(seed_a_idx + 1, NUM_SEEDS):
        seed_a = kan_topologies[seed_a_idx]
        seed_b = kan_topologies[seed_b_idx]
        comparison_result = [seed_a, seed_b, compute_jaccard(seed_a, seed_b)]
        jaccard_scores.append(comparison_result)

raw_jaccard_scores = []
for jaccard_score_idx in range(len(jaccard_scores)):
    raw_jaccard_scores.append(jaccard_scores[jaccard_score_idx][2])

mean_jaccard_score = statistics.mean(raw_jaccard_scores)

rand_mask_len = len(kan_topologies[0])
rand_kan_topologies = []
for mask_idx in range(len(kan_topologies)):
    mask_sparsity_score = sparsity_score(kan_topologies[mask_idx])

    num_active_splines = round(mask_sparsity_score * rand_mask_len)
    num_inactive_splines = rand_mask_len - num_active_splines
    random_mask = [1] * (num_active_splines) + [0] * (num_inactive_splines)
    random.shuffle(random_mask)

    rand_kan_topologies.append(random_mask)

rand_jaccard_scores = []
for seed_a_idx in range(len(rand_kan_topologies)):
    for seed_b_idx in range(seed_a_idx + 1, len(rand_kan_topologies)):
        seed_a = rand_kan_topologies[seed_a_idx]
        seed_b = rand_kan_topologies[seed_b_idx]
        comparison_result = [seed_a, seed_b, compute_jaccard(seed_a, seed_b)]
        rand_jaccard_scores.append(comparison_result)

raw_rand_jaccard_scores = []
for rand_jaccard_score_idx in range(len(rand_jaccard_scores)):
    raw_rand_jaccard_scores.append(rand_jaccard_scores[rand_jaccard_score_idx][2])

p_value = (np.array(raw_rand_jaccard_scores) >= mean_jaccard_score).sum() / len(rand_jaccard_scores)

intersection_count = 0
for position in range(len(kan_topologies[0])):
    ref_value = kan_topologies[0][position]
    all_seeds_agree = True

    for seed_idx in range(len(kan_topologies)):
        if(ref_value != kan_topologies[seed_idx][position]):
            all_seeds_agree = False
            break

    if(all_seeds_agree):
        intersection_count += 1

print(f"JACCARD P-VALUE = {p_value}")
print(f"INTERSECTION COUNT: {intersection_count}")