"""
Reward model: maps (prompt, response) → scalar reward.

Architecture: shared Transformer backbone + linear scalar head.
Trained on preference pairs (chosen > rejected) with Bradley-Terry loss.
"""
from __future__ import annotations
from typing import Optional
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax

from .base import ModelConfig, TransformerBackbone


class RewardModel(nn.Module):
    """Outputs a scalar reward for each sequence."""
    config: ModelConfig

    @nn.compact
    def __call__(self, input_ids: jnp.ndarray,
                  mask: Optional[jnp.ndarray] = None,
                  deterministic: bool = True) -> jnp.ndarray:
        """Returns rewards of shape (B,)."""
        h = TransformerBackbone(self.config)(input_ids, mask, deterministic)
        # Pool over the last non-padding token (or just last position)
        if mask is not None:
            # Use last valid token position
            lengths  = mask.sum(axis=-1).astype(jnp.int32) - 1
            pooled   = h[jnp.arange(h.shape[0]), lengths]
        else:
            pooled = h[:, -1, :]   # last position

        h2     = nn.Dense(self.config.hidden_size)(pooled)
        h2     = nn.tanh(h2)
        reward = nn.Dense(1, use_bias=False)(h2).squeeze(-1)
        return reward                                                   # (B,)

    def preference_loss(self, chosen_ids: jnp.ndarray,
                         rejected_ids: jnp.ndarray,
                         chosen_mask: Optional[jnp.ndarray] = None,
                         rejected_mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Bradley-Terry preference loss: -log σ(r_w - r_l)."""
        r_w = self(chosen_ids,   chosen_mask)
        r_l = self(rejected_ids, rejected_mask)
        loss = -jax.nn.log_sigmoid(r_w - r_l).mean()
        acc  = (r_w > r_l).mean()
        return loss, acc


class MultiObjectiveRewardModel(nn.Module):
    """
    Produces K separate reward scalars (e.g., task quality, safety, style).
    Final reward is a weighted sum: r = Σ w_k * r_k.
    """
    config:  ModelConfig
    n_objectives: int = 3

    @nn.compact
    def __call__(self, input_ids: jnp.ndarray,
                  mask: Optional[jnp.ndarray] = None,
                  deterministic: bool = True) -> jnp.ndarray:
        h = TransformerBackbone(self.config)(input_ids, mask, deterministic)
        pooled  = h[:, -1, :]
        rewards = nn.Dense(self.n_objectives, use_bias=False)(pooled)  # (B, K)
        return rewards

    def weighted_reward(self, input_ids: jnp.ndarray,
                         weights: jnp.ndarray,
                         mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """weights: (K,) → returns (B,) scalar."""
        rewards = self(input_ids, mask)
        return (rewards * weights[None, :]).sum(axis=-1)


def create_reward_train_state(config: ModelConfig,
                               learning_rate: float = 1e-4,
                               rng_key = None) -> train_state.TrainState:
    if rng_key is None:
        rng_key = jax.random.PRNGKey(1)
    model  = RewardModel(config)
    dummy  = jnp.ones((1, 4), dtype=jnp.int32)
    params = model.init(rng_key, dummy)
    tx     = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate),
    )
    return train_state.TrainState.create(apply_fn=model.apply,
                                          params=params, tx=tx)
