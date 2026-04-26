"""
All alignment loss functions in JAX.

Implemented:
  sft_loss          — standard cross-entropy language modelling
  reward_loss       — Bradley-Terry preference ranking loss
  dpo_loss          — Direct Preference Optimization (Rafailov et al., 2023)
  ppo_policy_loss   — clipped PPO surrogate objective
  ppo_value_loss    — PPO value function MSE loss
  kl_penalty        — KL divergence penalty between policy and reference
  multi_obj_reward  — weighted multi-objective reward combination
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
from typing import Optional


# ── SFT loss ─────────────────────────────────────────────────────────────

def sft_loss(logits: jnp.ndarray,
              labels: jnp.ndarray,
              mask:   Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """
    Cross-entropy loss for next-token prediction.

    logits : (B, T, V)
    labels : (B, T)  — token ids shifted by 1 from input
    mask   : (B, T)  — 1 for real tokens, 0 for padding
    """
    B, T, V = logits.shape
    log_probs  = jax.nn.log_softmax(logits, axis=-1)              # (B, T, V)
    token_loss = -jnp.take_along_axis(
        log_probs, labels[..., None], axis=-1).squeeze(-1)        # (B, T)

    if mask is not None:
        token_loss = token_loss * mask
        return token_loss.sum() / (mask.sum() + 1e-8)
    return token_loss.mean()


# ── Reward model loss ─────────────────────────────────────────────────────

def reward_loss(r_chosen: jnp.ndarray,
                 r_rejected: jnp.ndarray) -> tuple:
    """
    Bradley-Terry preference loss.
    r_chosen, r_rejected : (B,) scalar rewards
    Returns (loss, accuracy)
    """
    loss = -jax.nn.log_sigmoid(r_chosen - r_rejected).mean()
    acc  = (r_chosen > r_rejected).astype(jnp.float32).mean()
    return loss, acc


# ── DPO loss ──────────────────────────────────────────────────────────────

def dpo_loss(policy_chosen_lp:   jnp.ndarray,
              policy_rejected_lp: jnp.ndarray,
              ref_chosen_lp:      jnp.ndarray,
              ref_rejected_lp:    jnp.ndarray,
              beta: float = 0.1) -> tuple:
    """
    Direct Preference Optimization loss (Rafailov et al., 2023).

    All inputs: (B,) sequence log-probs.

    L_DPO = -log σ(β * [log π_θ(y_w)/π_ref(y_w) - log π_θ(y_l)/π_ref(y_l)])
    """
    chosen_ratio   = policy_chosen_lp   - ref_chosen_lp
    rejected_ratio = policy_rejected_lp - ref_rejected_lp
    loss = -jax.nn.log_sigmoid(beta * (chosen_ratio - rejected_ratio)).mean()
    # Implicit reward margin
    reward_margin  = (chosen_ratio - rejected_ratio).mean()
    win_rate       = (chosen_ratio > rejected_ratio).astype(jnp.float32).mean()
    return loss, {'reward_margin': reward_margin, 'win_rate': win_rate}


# ── PPO losses ────────────────────────────────────────────────────────────

def ppo_policy_loss(log_probs:     jnp.ndarray,
                    old_log_probs: jnp.ndarray,
                    advantages:    jnp.ndarray,
                    clip_eps:      float = 0.2,
                    mask:          Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """
    Clipped PPO surrogate objective.

    log_probs, old_log_probs, advantages: (B, T)
    Returns scalar loss (negated for gradient ascent).
    """
    ratio    = jnp.exp(log_probs - old_log_probs)
    clipped  = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
    obj      = jnp.minimum(ratio * advantages, clipped * advantages)

    if mask is not None:
        obj  = obj * mask
        return -(obj.sum() / (mask.sum() + 1e-8))
    return -obj.mean()


def ppo_value_loss(values:       jnp.ndarray,
                   old_values:   jnp.ndarray,
                   returns:      jnp.ndarray,
                   clip_eps:     float = 0.2,
                   mask:         Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """Clipped value function loss."""
    v_clipped = old_values + jnp.clip(values - old_values, -clip_eps, clip_eps)
    loss_v1   = (values    - returns) ** 2
    loss_v2   = (v_clipped - returns) ** 2
    loss      = 0.5 * jnp.maximum(loss_v1, loss_v2)
    if mask is not None:
        loss = loss * mask
        return loss.sum() / (mask.sum() + 1e-8)
    return loss.mean()


def entropy_bonus(logits: jnp.ndarray,
                   mask:   Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """Entropy of the policy distribution for exploration bonus."""
    log_p = jax.nn.log_softmax(logits, axis=-1)
    p     = jax.nn.softmax(logits, axis=-1)
    ent   = -(p * log_p).sum(axis=-1)   # (B, T)
    if mask is not None:
        ent = ent * mask
        return ent.sum() / (mask.sum() + 1e-8)
    return ent.mean()


# ── KL penalty ────────────────────────────────────────────────────────────

def kl_penalty(policy_log_probs: jnp.ndarray,
                ref_log_probs:   jnp.ndarray,
                mask:            Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """
    Per-token KL: D_KL(policy || ref) = log π_θ - log π_ref.
    Returns scalar mean KL.
    """
    kl = policy_log_probs - ref_log_probs
    if mask is not None:
        kl = kl * mask
        return kl.sum() / (mask.sum() + 1e-8)
    return kl.mean()


# ── GAE advantage estimation ─────────────────────────────────────────────

def gae_advantages(rewards:  jnp.ndarray,
                    values:   jnp.ndarray,
                    dones:    jnp.ndarray,
                    gamma:    float = 0.99,
                    lam:      float = 0.95) -> tuple:
    """
    Generalised Advantage Estimation (Schulman et al., 2015).

    rewards, values, dones: (B, T)
    Returns (advantages, returns) both (B, T).
    """
    T         = rewards.shape[1]
    adv       = jnp.zeros_like(rewards)
    last_adv  = 0.0
    last_val  = values[:, -1]

    advs = []
    for t in reversed(range(T)):
        next_val  = values[:, t + 1] if t < T - 1 else last_val
        delta     = rewards[:, t] + gamma * next_val * (1 - dones[:, t]) - values[:, t]
        last_adv  = delta + gamma * lam * (1 - dones[:, t]) * last_adv
        advs.append(last_adv)

    advantages = jnp.stack(advs[::-1], axis=1)
    returns    = advantages + values
    # Normalise advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages, returns


# ── Multi-objective reward ────────────────────────────────────────────────

def multi_objective_reward(rewards:  jnp.ndarray,
                             weights: jnp.ndarray) -> jnp.ndarray:
    """
    Combine K reward dimensions into a scalar.
    rewards : (B, K)
    weights : (K,)
    Returns : (B,)
    """
    return (rewards * weights[None, :]).sum(axis=-1)
