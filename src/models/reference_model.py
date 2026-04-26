"""
Frozen reference model for KL-divergence penalty in RLHF/DPO.

The reference model is the SFT-pretrained policy kept frozen
during RL fine-tuning. It prevents the policy from deviating too
far from the original distribution.
"""
from __future__ import annotations
from typing import Dict, Optional
import jax
import jax.numpy as jnp

from .base import ModelConfig
from .policy import PolicyModel


class ReferenceModel:
    """
    Wraps a PolicyModel with frozen parameters.
    All calls are non-differentiable (jax.lax.stop_gradient applied).
    """

    def __init__(self, config: ModelConfig, params: Dict):
        self.config = config
        self.model  = PolicyModel(config)
        self.params = jax.lax.stop_gradient(params)

    @classmethod
    def from_sft_state(cls, sft_state, config: ModelConfig) -> 'ReferenceModel':
        """Create a reference model from a trained SFT TrainState."""
        return cls(config, sft_state.params)

    def log_probs(self, input_ids: jnp.ndarray,
                   mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Sequence log-probs under the reference policy. Returns (B,)."""
        params = jax.lax.stop_gradient(self.params)
        return self.model.apply(params, input_ids, mask,
                                 method=self.model.sequence_log_prob)

    def per_token_log_probs(self, input_ids: jnp.ndarray,
                             mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Per-token log-probs. Returns (B, T-1)."""
        params  = jax.lax.stop_gradient(self.params)
        logits  = self.model.apply(params, input_ids[:, :-1], mask)
        labels  = input_ids[:, 1:]
        lp      = jax.nn.log_softmax(logits, axis=-1)
        return jnp.take_along_axis(lp, labels[..., None], axis=-1).squeeze(-1)

    def kl_from_policy(self, policy_log_probs: jnp.ndarray,
                        input_ids: jnp.ndarray,
                        mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Per-token KL divergence D_KL(policy || reference).
        KL ≈ log π_θ(a|s) - log π_ref(a|s)
        Returns (B,) summed over tokens.
        """
        ref_lp = self.per_token_log_probs(input_ids, mask)
        kl     = policy_log_probs - ref_lp
        if mask is not None:
            kl = kl * mask[:, 1:]
        return kl.sum(axis=-1)
