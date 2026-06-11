# KANT
### Kolmogorov-Arnold Network Aided Network Transfer for Cross-Architecture Topology Initialization via Deep Reinforcement Learning 

**Research Question:** 
Does the functionally critical topology discovered by Kolmogorov-Arnold Networks trained via a PPO algorithm during reinforcement learning training constitute a transferable inductive bias that improves MLP initialization beyond what architecture-agnostic pruning methods achieve?

**Centeral Claim:**
KANs are highly flexible function approximators with no explicit structure for modular reuse. Yet we hypothesize they consistently converge to stable functional substructures across independent training runs whose topology transfers meaningfully to a completely different architecture. In other words, KANs discover genuine casual topological strucutre which contains enough information about the environment/task such that when transferred to a randomly initialized MLP, the inductive bias accelerates learning speeds, increases sampling efficiency, allows for stable convergence and consistency across seeds, and increases generalization capabilities.

## Project Overview:

### Architectures
- **KAN** (Kolmogorov-Arnold Network): A neural architecture that parameterizes learnable spline functions on edges instead of using fixed node activations. In this work, KANs are used to investigate whether reinforcement learning consistently induces meaningful topological structures and importance patterns across independent training runs.

- **MLP** (Multilayer Perceptron): A standard feedforward neural network used as the recipient architecture for topology transfer. The goal is not to compare KANs and MLPs directly, but to determine whether structural priors discovered by KANs can be imposed on MLPs and subsequently improve reinforcement learning performance.

### Environments
- **Hopper-v4** — MuJoCo continuous control, 11-dimensional state space, 3-dimensional continuous action space. One-legged robot learning to hop forward. 

- **HalfCheetah-v4** — MuJoCo continuous control, 17-dimensional state space, 6-dimensional continuous action space. 2D cheetah robot learning to run forward. 

## Four Core Experiments:
### **Experiment 1: Causal Consistency of Functional KAN Topologies Across Random Seeds**

**1A.** (Cross-seed consistency) Train KANs across 10 independent random seeds per environment. Extract surviving connections per seed using fomal sparsity score definition. Compute the consistency score through the fraction of seeds that share identical surviving connections. Run a permutation test against random subgraph overlap to prove consistency is statistically significantly higher than chance.

**1B.** (Load-Bearing Verification via Connection Ablation) Take trained KAN agents and systematically remove each surviving connection. Measure performance drop after the removal of surviving splines; conversely, remove non-surviving connections one at a time and confirm no performance drop. Produce ranked casual importance scores of each connection in the network which will be required for experiment 3. The process above will only be executed on 5 randomly sampled networks from each environment to lower computational costs.

**Statistical Tests:** Permutation test for consistency, Cohen's d effect sizes for intervention ablation results, bootstrap confidence intervals.


### **Experiment 2: Temporal Emergence and Phase Transition of Functional KAN Topology During PPO Training**

Utilizing the same 10 random seeded KANs from each environement, we analyze the topological structure and reward through going back their training checkpoints. For every model, we should have saved checkpoints periodically every 100k timesteps or every N episodes. At each individual checkpoint we compute topology stability and use the `ruptures` library for change point detection for when topology stabilizes. Generate a topology stability curve and overlay it on top of a reward curve to test whether phase transition coincides with policy breakthrough when reward increases rapidly. Potentially measure metrics such as FLOPS. 


### **Experiment 3: Cross-Architecture Transfer of Causal KAN Topology for MLP Initialization in Reinforcement Learning**

**3A.** Extract the functional topology from trained KAN agents using causal importance scores from Experiment 1. Vary the topology transfer threshold across five levels:
* Top 20% of surviving connections by causal importance
* Top 40% of surviving connections by causal importance
* Top 60% of surviving connections by causal importance
* Top 80% of surviving connections by causal importance
* Full topology
For each threshold, initialize an MLP where only weights corresponding to transferred connections are active using binary connectivity mask, and randomly initialize weights in surviving positions. Train the model from scratch and measure learning speed defined as number of timesteps to reach a specific reward thresold, final performance at convergence, and AUC. Possibly run all through major perbutations to test generalization? 

**3B.** Compare the performance of the KANT MLP to various different variants of MLPs (10 random seeds for each variant): 
* Randomly initialized dense MLP
* Randomly initialized sparse MLP
* Randomly initialized iterative magnitude pruned MLP
* Winning ticket MLP
* KANT MLP (best performing KANT MLP topology thresold from 3a)

Possibly run all through major perbutations to test generalization?

Statistical tests: Two-way ANOVA, Linear Mixed-Effects Model with seed as random effect, Cohen's d effect sizes for all pairwise comparisons, bootstrap confidence intervals for transfer curve, and p-values.


### **Experiment 4: Structural vs Importance Contributions to Causal Topology Transfer**

Compare two KANT MLP initialization variants: Binary connectivity mask and Importance weighted initialization with 5 random seeded KANT MLPs each.

**Binary connectivity mask:** Transfers only structural information of which connections survive, which are then randomly initialized.

**Importance weighted initialization:** Utilizes casual importance scores to scale initial weights accordingly (higher importance score = higher initial weight in KANT MLP) (still brainstorming on how to convert).

Statistical tests: Paired t-tests between Approach 1 and Approach 2 at each threshold level. Cohen's d effect sizes.

## Novel Contributions:
* KANT method: First systematic method for extracting functional topology from KAN policies and transferring it as an initialization prior to MLPs in RL (PRIMARY CONTRIBUTION)
* Causal intervention ablation proving KAN surviving connections are load-bearing in RL settings
* Topology consistency study across independent random seeds in KAN-based RL policies (maybe need to remove, overclaiming)
* Structural versus importance-weighted connections via binary connectivity mask versus importance weighted initialization comparison

**Possible Addition:** Theoretical proof of sparsity emergence proving functional topology must emerge under PPO training in casual environment structure

## Tools & Stack
- Python / PyTorch
- pykan library
- gymnasium + gymnasium[mujoco]
- Stable Baselines 3 (PPO)
- Weights & Biases (experiment tracking)
- MacBook M4 Pro (primary training hardware)
- Google Colab (backup GPU)
- CodeCarbon (energy tracking)
- calflops or torchprofile (FLOPs measurement)(TBD)


**Next Goal: Finish pilot experiemnts successfully and begin main experiments** 

**Last updated June 11th**