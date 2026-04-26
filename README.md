# GenAlign — A Framework for Aligning Generative Models with Reinforcement Learning

A modular JAX/Flax toolkit implementing the full RLHF alignment pipeline: Supervised Fine-Tuning (SFT), Reward Modeling, PPO-based RLHF, and Direct Preference Optimization (DPO). Designed for research into aligning large language models with human preferences.

---

## Overview

GenAlign provides clean, research-grade implementations of the four-stage alignment pipeline:

```
Stage 1: SFT          — fine-tune on curated demonstrations
Stage 2: Reward Model — learn scalar reward from preference pairs
Stage 3: RLHF/PPO     — optimise policy against reward model with PPO
Stage 4: DPO          — direct preference optimisation without RL
```

All components are implemented in **JAX + Flax** for hardware-accelerated training on GPUs/TPUs via `jit`, `vmap`, and `pmap`.

---

## Results

### RLHF/PPO vs DPO on Anthropic/hh-rlhf

| Method        | Win Rate | Mean Reward | KL from SFT | Toxicity | Train Time |
|---------------|----------|-------------|-------------|----------|------------|
| SFT Baseline  | 0.500    | 0.312       | 0.000       | 8.3%     | —          |
| RLHF/PPO      | **0.681**| **0.894**   | 0.342       | 3.1%     | 4.2 h      |
| DPO           | 0.664    | 0.847       | **0.218**   | **2.8%** | **1.8 h**  |
| Multi-obj PPO | **0.703**| **0.921**   | 0.389       | **1.9%** | 4.8 h      |

- RLHF/PPO achieves the highest win rate (+18.1 pp over baseline) but uses more memory and requires online rollouts.
- DPO is 2.3× faster, preserves language quality better (lower KL), and requires no reward model at inference.
- Multi-objective PPO (helpfulness 0.6 + safety 0.3 + conciseness 0.1) shows the best overall alignment.

Detailed results: [`results/evaluation_metrics.json`](results/evaluation_metrics.json) | [`results/alignment_comparison.json`](results/alignment_comparison.json)

---

## Project Structure

```
gen-align/
├── src/
│   ├── models/
│   │   ├── base.py             # Transformer backbone (GPT-2 style)
│   │   ├── policy.py           # Causal LM policy
│   │   ├── reward_model.py     # Scalar + multi-objective reward model
│   │   ├── value_model.py      # PPO critic
│   │   └── reference_model.py  # Frozen SFT reference for KL penalty
│   ├── alignment/
│   │   ├── losses.py           # All JAX loss functions
│   │   ├── sft.py              # SFT trainer
│   │   ├── reward_trainer.py   # Reward model trainer
│   │   ├── ppo.py              # PPO trainer (full RLHF loop)
│   │   └── dpo.py              # DPO trainer
│   ├── data/
│   │   ├── preference_dataset.py  # HuggingFace preference datasets
│   │   ├── synthetic_prefs.py     # LLM-as-judge synthetic preference gen
│   │   └── data_utils.py          # Tokenisation, batching, JSONL loader
│   ├── evaluation/
│   │   ├── metrics.py          # Win rate, perplexity, KL, Distinct-n
│   │   ├── evaluator.py        # Full evaluation pipeline
│   │   └── safety.py           # Toxicity scoring (heuristic + Detoxify)
│   └── utils/
│       ├── jax_utils.py        # Checkpointing, sharding, PRNGKey utils
│       ├── logging_utils.py    # CSV + W&B logging
│       └── visualization.py    # Training curves, reward distributions
├── scripts/
│   ├── train_sft.py
│   ├── train_reward_model.py
│   ├── train_ppo.py
│   ├── train_dpo.py
│   └── evaluate.py
├── tests/
│   ├── test_models.py
│   ├── test_alignment.py
│   └── test_evaluation.py
├── configs/
│   ├── sft_config.yaml
│   ├── reward_config.yaml
│   ├── ppo_config.yaml
│   └── dpo_config.yaml
└── results/
    ├── alignment_comparison.json
    ├── reward_scores.csv
    └── evaluation_metrics.json
```

---

## Installation

```bash
git clone https://github.com/sri-godugu/gen-align.git
cd gen-align
pip install -r requirements.txt
```

For GPU/TPU support, install the appropriate `jaxlib` variant from the [JAX installation guide](https://github.com/google/jax#installation).

---

## Usage

### Stage 1 — Supervised Fine-Tuning

```bash
python scripts/train_sft.py --config configs/sft_config.yaml
```

The SFT model is the starting point for all alignment methods. It is also used as the frozen reference policy for KL-divergence regularisation during RL training.

### Stage 2 — Reward Model

```bash
python scripts/train_reward_model.py --config configs/reward_config.yaml
```

Trains a Bradley-Terry reward model on `(chosen, rejected)` preference pairs. Supports single-objective and multi-objective reward heads.

### Stage 3 — RLHF/PPO

```bash
python scripts/train_ppo.py \
    --config        configs/ppo_config.yaml \
    --ref_params    outputs/sft/sft_params.pkl \
    --reward_params outputs/reward_model/reward_params.pkl
```

Full RLHF loop: rollout generation → reward + KL scoring → GAE advantage estimation → clipped PPO updates.

### Stage 4 — DPO (no reward model required)

```bash
python scripts/train_dpo.py \
    --config     configs/dpo_config.yaml \
    --ref_params outputs/sft/sft_params.pkl
```

### Evaluation

```bash
python scripts/evaluate.py \
    --ppo_params    outputs/ppo/ppo_policy_params.pkl \
    --dpo_params    outputs/dpo/dpo_policy_params.pkl \
    --ref_params    outputs/sft/sft_params.pkl \
    --reward_params outputs/reward_model/reward_params.pkl \
    --test_data     data/preference_test.jsonl \
    --output_dir    results/
```

---

## Implemented Algorithms

### Loss Functions (`src/alignment/losses.py`)

| Function | Formula | Use |
|---|---|---|
| `sft_loss` | Cross-entropy | Stage 1 |
| `reward_loss` | `-log σ(r_w − r_l)` | Stage 2 |
| `dpo_loss` | `-log σ(β·[log π(y_w)/π_ref(y_w) − log π(y_l)/π_ref(y_l)])` | Stage 4 |
| `ppo_policy_loss` | `E[min(r·A, clip(r,1±ε)·A)]` | Stage 3 |
| `ppo_value_loss` | Clipped MSE | Stage 3 |
| `gae_advantages` | GAE (Schulman 2015) | Stage 3 |
| `kl_penalty` | `log π_θ − log π_ref` | Stage 3 |

### Multi-Objective Rewards

```python
from src.models.reward_model import MultiObjectiveRewardModel

model = MultiObjectiveRewardModel(config, n_objectives=3)
# weights: [helpfulness, safety, conciseness]
reward = model.weighted_reward(input_ids, weights=jnp.array([0.6, 0.3, 0.1]))
```

### Synthetic Preference Generation

```python
from src.data.synthetic_prefs import build_synthetic_dataset

pairs = build_synthetic_dataset(prompts, n_pairs=500, output_path='data/synthetic.jsonl')
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Key Design Choices

- **JAX functional style**: all model parameters are explicit dicts; `jax.jit` + `jax.grad` for efficient compiled training.
- **Weight-tied LM head**: the output projection reuses the embedding matrix, halving memory usage for the policy.
- **Stop-gradient reference model**: `jax.lax.stop_gradient` on frozen reference params ensures zero-cost KL computation.
- **Clipped PPO + clipped value loss**: both actor and critic use clipping to prevent large updates.
- **GAE normalisation**: advantages are mean/std normalised per batch for stable gradients.

---

## References

- Ouyang et al., *Training language models to follow instructions with human feedback* (InstructGPT, 2022)
- Schulman et al., *Proximal Policy Optimization Algorithms* (2017)
- Schulman et al., *High-dimensional continuous control using generalized advantage estimation* (2015)
- Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* (2023)
- Stiennon et al., *Learning to summarize from human feedback* (2020)
