"""
Base model definitions and shared Flax building blocks.

All GenAlign models share a common Transformer backbone (GPT-2 style)
implemented in Flax / JAX.  The backbone is subclassed to produce:
  - PolicyModel   : next-token log-probs
  - RewardModel   : scalar reward
  - ValueModel    : scalar value (critic for PPO)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import jax
import jax.numpy as jnp
import flax.linen as nn


# ── Config ────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    vocab_size:    int   = 50257    # GPT-2 default
    max_seq_len:   int   = 512
    hidden_size:   int   = 768
    n_layers:      int   = 12
    n_heads:       int   = 12
    mlp_ratio:     float = 4.0
    dropout:       float = 0.1
    layer_norm_eps: float = 1e-5
    tie_embeddings: bool  = True

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.n_heads


# ── Shared Transformer building blocks ───────────────────────────────────

class MultiHeadAttention(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None,
                  deterministic: bool = True) -> jnp.ndarray:
        B, T, C = x.shape
        H, D    = self.config.n_heads, self.config.head_dim

        qkv = nn.Dense(3 * C, use_bias=True)(x)                  # (B, T, 3C)
        q, k, v = jnp.split(qkv, 3, axis=-1)                     # each (B, T, C)
        q = q.reshape(B, T, H, D).transpose(0, 2, 1, 3)          # (B, H, T, D)
        k = k.reshape(B, T, H, D).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, H, D).transpose(0, 2, 1, 3)

        scale  = D ** -0.5
        attn   = (q @ k.transpose(0, 1, 3, 2)) * scale            # (B, H, T, T)

        # Causal mask
        causal = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
        attn   = jnp.where(causal, attn, jnp.finfo(jnp.float32).min)
        if mask is not None:
            attn = jnp.where(mask[:, None, None, :], attn,
                             jnp.finfo(jnp.float32).min)

        attn   = jax.nn.softmax(attn, axis=-1)
        attn   = nn.Dropout(rate=self.config.dropout)(attn, deterministic=deterministic)

        out    = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        return nn.Dense(C, use_bias=True)(out)


class FFN(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        hidden = int(self.config.hidden_size * self.config.mlp_ratio)
        h = nn.Dense(hidden)(x)
        h = nn.gelu(h)
        h = nn.Dense(self.config.hidden_size)(h)
        return nn.Dropout(rate=self.config.dropout)(h, deterministic=deterministic)


class TransformerBlock(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None,
                  deterministic: bool = True) -> jnp.ndarray:
        cfg = self.config
        # Pre-norm architecture
        h = nn.LayerNorm(epsilon=cfg.layer_norm_eps)(x)
        h = MultiHeadAttention(cfg)(h, mask, deterministic)
        h = nn.Dropout(rate=cfg.dropout)(h, deterministic=deterministic)
        x = x + h

        h = nn.LayerNorm(epsilon=cfg.layer_norm_eps)(x)
        h = FFN(cfg)(h, deterministic)
        x = x + h
        return x


class TransformerBackbone(nn.Module):
    """Shared GPT-2 style backbone used by all GenAlign models."""
    config: ModelConfig

    @nn.compact
    def __call__(self, input_ids: jnp.ndarray,
                  mask: Optional[jnp.ndarray] = None,
                  deterministic: bool = True) -> jnp.ndarray:
        cfg = self.config
        B, T = input_ids.shape

        tok_emb = nn.Embed(cfg.vocab_size, cfg.hidden_size)(input_ids)
        pos_ids = jnp.arange(T)[None, :]
        pos_emb = nn.Embed(cfg.max_seq_len, cfg.hidden_size)(pos_ids)
        x       = nn.Dropout(rate=cfg.dropout)(
            tok_emb + pos_emb, deterministic=deterministic)

        for _ in range(cfg.n_layers):
            x = TransformerBlock(cfg)(x, mask, deterministic)

        return nn.LayerNorm(epsilon=cfg.layer_norm_eps)(x)   # (B, T, H)
