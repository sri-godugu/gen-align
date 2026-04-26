"""
Policy model: language model producing next-token log-probabilities.

Used as:
  - the SFT reference model (frozen)
  - the trainable policy in RLHF/PPO and DPO
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax

from .base import ModelConfig, TransformerBackbone


class PolicyModel(nn.Module):
    """Causal language model (GPT-2 style) for policy training."""
    config: ModelConfig

    @nn.compact
    def __call__(self, input_ids: jnp.ndarray,
                  mask: Optional[jnp.ndarray] = None,
                  deterministic: bool = True) -> jnp.ndarray:
        """Returns logits (B, T, vocab_size)."""
        h = TransformerBackbone(self.config)(input_ids, mask, deterministic)
        if self.config.tie_embeddings:
            # Weight-tied LM head
            embed = self.variables['params']['TransformerBackbone_0']['Embed_0']['embedding']
            logits = h @ embed.T
        else:
            logits = nn.Dense(self.config.vocab_size, use_bias=False)(h)
        return logits

    def log_probs_from_logits(self, logits: jnp.ndarray,
                               labels: jnp.ndarray) -> jnp.ndarray:
        """Per-token log-probs for given labels. Shape: (B, T)."""
        lp = jax.nn.log_softmax(logits, axis=-1)
        return jnp.take_along_axis(lp, labels[..., None], axis=-1).squeeze(-1)

    def sequence_log_prob(self, input_ids: jnp.ndarray,
                           mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Sum of log-probs for tokens [1..T] conditioned on [0..T-1].
        Returns (B,) scalar log-prob per sequence.
        """
        logits     = self(input_ids[:, :-1], mask, deterministic=True)
        labels     = input_ids[:, 1:]
        token_lp   = self.log_probs_from_logits(logits, labels)
        if mask is not None:
            token_lp = token_lp * mask[:, 1:]
        return token_lp.sum(axis=-1)


def create_policy_train_state(config: ModelConfig,
                               learning_rate: float = 1e-4,
                               weight_decay: float  = 0.01,
                               rng_key = None) -> train_state.TrainState:
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
    model  = PolicyModel(config)
    dummy  = jnp.ones((1, 4), dtype=jnp.int32)
    params = model.init(rng_key, dummy)
    tx     = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate, weight_decay=weight_decay),
    )
    return train_state.TrainState.create(apply_fn=model.apply,
                                          params=params, tx=tx)
