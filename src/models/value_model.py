"""
Value model (critic) for PPO.

Shares the same backbone as the policy but outputs a per-token
state value V(s_t) used for advantage estimation.
"""
from __future__ import annotations
from typing import Optional
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax

from .base import ModelConfig, TransformerBackbone


class ValueModel(nn.Module):
    """Per-token value head; used as the critic in PPO."""
    config: ModelConfig

    @nn.compact
    def __call__(self, input_ids: jnp.ndarray,
                  mask: Optional[jnp.ndarray] = None,
                  deterministic: bool = True) -> jnp.ndarray:
        """Returns values (B, T)."""
        h = TransformerBackbone(self.config)(input_ids, mask, deterministic)
        v = nn.Dense(1, use_bias=False)(h).squeeze(-1)   # (B, T)
        return v


def create_value_train_state(config: ModelConfig,
                              learning_rate: float = 1e-4,
                              rng_key = None) -> train_state.TrainState:
    if rng_key is None:
        rng_key = jax.random.PRNGKey(2)
    model  = ValueModel(config)
    dummy  = jnp.ones((1, 4), dtype=jnp.int32)
    params = model.init(rng_key, dummy)
    tx     = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate),
    )
    return train_state.TrainState.create(apply_fn=model.apply,
                                          params=params, tx=tx)
