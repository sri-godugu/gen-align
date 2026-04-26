"""
Evaluation metrics for aligned language models.

Metrics
-------
win_rate              — fraction of completions preferred over reference
reward_score          — mean scalar reward under the trained reward model
perplexity            — token-level perplexity under the policy
kl_from_reference     — mean KL divergence from the SFT baseline
diversity             — distinct-n n-gram diversity of generated outputs
"""
from __future__ import annotations
import math
from collections import Counter
from typing import Dict, List, Optional, Sequence

import numpy as np
import jax
import jax.numpy as jnp


# ── Win rate ──────────────────────────────────────────────────────────────────

def compute_win_rate(policy_rewards:    np.ndarray,
                     reference_rewards: np.ndarray) -> float:
    """
    Fraction of samples where policy reward > reference reward.

    Parameters
    ----------
    policy_rewards    : (N,)
    reference_rewards : (N,)
    """
    return float((policy_rewards > reference_rewards).mean())


# ── Reward score ──────────────────────────────────────────────────────────────

def compute_mean_reward(rewards: np.ndarray) -> Dict[str, float]:
    """Return mean, std, min, max of a reward array."""
    return {
        'mean': float(rewards.mean()),
        'std':  float(rewards.std()),
        'min':  float(rewards.min()),
        'max':  float(rewards.max()),
    }


# ── Perplexity ─────────────────────────────────────────────────────────────────

def compute_perplexity(logits:  np.ndarray,
                        labels:  np.ndarray,
                        mask:    Optional[np.ndarray] = None) -> float:
    """
    Token-level perplexity.

    logits : (B, T, V)
    labels : (B, T)
    mask   : (B, T)   1 for real tokens
    """
    log_probs  = jax.nn.log_softmax(jnp.array(logits), axis=-1)
    token_nlls = -jnp.take_along_axis(
        log_probs, jnp.array(labels)[..., None], axis=-1).squeeze(-1)

    if mask is not None:
        m = jnp.array(mask)
        nll = (token_nlls * m).sum() / (m.sum() + 1e-8)
    else:
        nll = token_nlls.mean()
    return float(jnp.exp(nll))


# ── KL divergence ─────────────────────────────────────────────────────────────

def compute_kl(policy_log_probs: np.ndarray,
                ref_log_probs:   np.ndarray,
                mask:            Optional[np.ndarray] = None) -> float:
    """
    Mean per-token KL divergence D_KL(policy || ref).

    policy_log_probs, ref_log_probs : (B, T)
    """
    kl = policy_log_probs - ref_log_probs
    if mask is not None:
        kl = kl * mask
        return float(kl.sum() / (mask.sum() + 1e-8))
    return float(kl.mean())


# ── Diversity (Distinct-n) ─────────────────────────────────────────────────────

def distinct_n(texts: List[str], n: int = 2) -> float:
    """
    Fraction of unique n-grams over all generated texts.
    Higher → more diverse generations.
    """
    ngrams_all, ngrams_unique = 0, set()
    for text in texts:
        tokens = text.split()
        grams  = zip(*[tokens[i:] for i in range(n)])
        for g in grams:
            ngrams_all    += 1
            ngrams_unique.add(g)
    return len(ngrams_unique) / max(ngrams_all, 1)


# ── Alignment delta ──────────────────────────────────────────────────────────

def alignment_delta(before_rewards: np.ndarray,
                     after_rewards:  np.ndarray) -> Dict[str, float]:
    """Improvement statistics after alignment training."""
    delta = after_rewards - before_rewards
    return {
        'mean_delta':  float(delta.mean()),
        'pct_improved': float((delta > 0).mean()),
        'win_rate':    compute_win_rate(after_rewards, before_rewards),
    }


# ── Aggregate report ─────────────────────────────────────────────────────────

def compute_all_metrics(policy_rewards:    np.ndarray,
                         reference_rewards: np.ndarray,
                         policy_kl:         Optional[np.ndarray] = None,
                         generations:       Optional[List[str]]  = None) -> Dict:
    report: Dict = {}
    report['win_rate']    = compute_win_rate(policy_rewards, reference_rewards)
    report['reward']      = compute_mean_reward(policy_rewards)
    report['ref_reward']  = compute_mean_reward(reference_rewards)
    report['delta']       = alignment_delta(reference_rewards, policy_rewards)
    if policy_kl is not None:
        report['mean_kl'] = float(policy_kl.mean())
    if generations:
        report['distinct_1'] = distinct_n(generations, 1)
        report['distinct_2'] = distinct_n(generations, 2)
    return report
