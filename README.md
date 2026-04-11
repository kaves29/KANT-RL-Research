# KAN vs. MLP in Reinforcement Learning
### Studying Functional Sparsity Emergence, Policy Interpretability, and Efficiency

**Research Question:** When trained via reinforcement learning, do 
Kolmogorov-Arnold Networks naturally develop functional sparsity — 
and if so, does that sparsity correlate with more stable, efficient, 
and interpretable policies compared to traditional MLPs?

---

## Project Overview
This project compares two neural network architectures:
- **MLP** (Multilayer Perceptron) — the traditional standard
- **KAN** (Kolmogorov-Arnold Network) — published by MIT, April 2024

Both agents are trained on two environments:
- CartPole-v1 (simple control task)
- LunarLander-v2 (complex control task)

20 total independent training runs (2 architectures × 2 environments × 5 runs each)

---

## Key Metrics
- Reward over time (learning curves)
- Convergence speed (episodes to stable performance)
- Parameter count (computational cost)
- Sparsity score (novel metric — % of inactive spline functions)
- Reward stability (mean ± variance across 100 evaluation episodes)
- Interpretability analysis (dominant spline extraction and annotation)

---

## Tools & Stack
- Python / PyTorch
- pykan library
- Gymnasium (CartPole, LunarLander environments)
- Stable Baselines 3
- Weights & Biases (experiment tracking)
- Google Colab (GPU training)

---

## Repository Structure
- `/notebooks` — Google Colab experiment notebooks
- `/src` — MLP and KAN agent source code
- `/data` — Raw results from all 20 training runs
- `/figures` — Exported graphs and visualizations
- `/references` — Key citations and paper links

---

*Sophmore research project targeting ISEF 2027.*
*Started April 2026.*