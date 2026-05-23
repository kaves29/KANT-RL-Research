# The Sparse Oracle
### Emergent Sparsity Patterns in KAN Policies as Predictors of Robustness Under Distribution Shift in Reinforcement Learning

**Research Question:** When Kolmogorov-Arnold Network policies are trained 
via reinforcement learning, do they develop structured sparsity patterns 
that emerge predictably during training, remain topologically consistent 
across independent random seeds, correspond to physically meaningful state 
variables, and predict agent robustness to distribution shift — and does 
this emergent sparsity represent a fundamentally different generalization 
mechanism than the imposed sparsity achievable in equivalent MLP agents?

**Central Claim:** Emergent sparsity in KAN policies predicts robustness 
to distribution shift in ways that imposed sparsity cannot replicate.

---

## Project Overview
This project investigates whether internal sparsity patterns that develop
naturally in Kolmogorov-Arnold Networks during reinforcement learning
training serve as a reliable predictor of policy robustness under
distribution shift — and whether this emergent sparsity is fundamentally
distinct from sparsity artificially imposed on equivalent MLP agents via
iterative magnitude pruning with fine-tuning.

### Architectures
- **KAN** (Kolmogorov-Arnold Network) — MIT, April 2024. Learnable spline
  functions on edges instead of fixed activations on nodes.
- **MLP** (Multilayer Perceptron) — traditional baseline. Used for
  comparison and for the imposed sparsity baseline via iterative magnitude
  pruning with fine-tuning.

### Environments
- **Hopper-v4** — MuJoCo continuous control, 11-dimensional state space,
  3-dimensional continuous action space. One-legged robot learning to hop
  forward. Featured in original PPO paper (Schulman et al. 2017).
  Simple environment in complexity gradient.
- **HalfCheetah-v4** — MuJoCo continuous control, 17-dimensional state
  space, 6-dimensional continuous action space. 2D cheetah robot learning
  to run forward. Body-part specific state variables (thigh, shin, foot
  angles and velocities) enable precise topology consistency predictions
  against biomechanical ground truth. Complex environment in complexity
  gradient.

### Scale
40 total independent training runs — 10 seeds per condition
2 architectures × 2 environments × 10 seeds each

---

## Four Core Experiments

**1. Sparsity Emergence Dynamics**
Plot sparsity score against training episodes across all 10 seeds per
condition. Look for a phase transition where sparsity jumps sharply
corresponding to policy breakthrough moments. Standard deviation bands
plotted across all seeds.

**2. Topology Consistency Across Seeds**
Identify which specific spline connections survive across all 10
independent seeds per condition. Test whether surviving connections
correspond to state variables that are physically meaningful.
Pre-registered prediction for HalfCheetah: back thigh and back shin
splines will survive while foot splines go flat, consistent with
biomechanical analysis of 2D locomotion. Pre-registered prediction
for Hopper: hip and thigh splines will survive while foot splines
contribute less.

**3. Sparsity Predicts Robustness (Central Claim)**
Test all 40 trained agents under environment perturbations without
retraining. Two perturbation types per environment. Correlate
pre-tested sparsity score with post-perturbation performance drop
using Pearson correlation, p-values, and Cohen's d effect sizes.

**4. Emergent vs. Imposed Sparsity Comparison**
Prune MLP agents to match KAN sparsity levels using iterative magnitude
pruning with fine-tuning — the strongest available pruning baseline.
Compare robustness of naturally sparse KAN versus artificially sparse
MLP at equal sparsity levels.

---

## Distribution Shift Perturbations
- **Hopper-v4:** gravity modification, joint friction modification
- **HalfCheetah-v4:** gravity modification, joint friction modification

---

## Key Metrics
- Sparsity score (inactive splines / total splines, variance threshold ε)
- Sparsity emergence curve (sparsity score vs. training episodes)
- Topology consistency score (fraction of seeds sharing identical
  surviving connections)
- Robustness score (performance drop under perturbation)
- Pearson correlation: sparsity score vs. robustness score
- Cohen's d effect sizes for all statistical comparisons
- Reward mean and variance across 100 evaluation episodes per agent

---

## Tools & Stack
- Python / PyTorch
- pykan library
- gymnasium + gymnasium[mujoco]
- Stable Baselines 3 (PPO)
- Weights & Biases (experiment tracking)
- MacBook M4 Pro (primary training hardware)
- Google Colab (backup GPU)
- CodeCarbon (energy tracking)
- calflops or torchprofile (FLOPs measurement)

---

## Repository Structure
- `/notebooks` — experiment notebooks
  - `00_environment_setup.ipynb`
  - `01_mlp_hopper.ipynb`
  - `02_mlp_halfcheetah.ipynb`
  - `03_kan_hopper.ipynb`
  - `04_kan_halfcheetah.ipynb`
  - `05_sparsity_analysis.ipynb`
  - `06_topology_consistency.ipynb`
  - `07_robustness_testing.ipynb`
  - `08_imposed_sparsity_baseline.ipynb`
  - `09_comparative_statistics.ipynb`
- `/src` — MLP and KAN agent source code
- `/data` — Raw results from all 40 training runs
- `/figures` — Exported graphs and visualizations
- `/references` — Key citations and paper links

---

## Literature Foundation
- Liu et al. — KAN: Kolmogorov-Arnold Networks (2024)
- Schulman et al. — PPO Algorithms (2017)
- Igl et al. — Generalization in RL with Selective Noise Injection (2019)
- Cobbe et al. — Quantifying Generalization in RL (2019)
- Frankle and Carlin — The Lottery Ticket Hypothesis (2019)
- Doshi-Velez and Kim — Rigorous Science of Interpretable ML (2017)
- SPAN — Agile RL through Separable Neural Architectures (2026)
- Interpretable RL for Load Balancing using KAN (2025)
- RL Finetunes Small Subnetworks (2025)

---

*Freshman research project targeting ISEF 2027.*
*Pivot finalized May 23, 2026. No experimental code written yet.*
*Next milestone: finish literature review by May 31, begin PyTorch June 1.*