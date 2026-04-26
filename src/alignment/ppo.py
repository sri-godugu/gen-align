"""
PPO trainer for RLHF.

Implements the classic RLHF loop:
  1. Rollout — sample responses from the current policy.
  2. Score   — obtain rewards from the frozen reward model + KL penalty.
  3. Estimate advantages with GAE.
  4. Update  — several epochs of clipped PPO on the collected experience.

References
----------
Schulman et al., "Proximal Policy Optimization Algorithms" (2017).
Stiennon et al., "Learning to summarize from human feedback" (2020).
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax.training import train_state

from ..models.policy import PolicyModel, create_policy_train_state
from ..models.value_model import ValueModel, create_value_train_state
from ..models.reward_model import RewardModel
from ..models.reference_model import ReferenceModel
from ..models.base import ModelConfig
from .losses import (ppo_policy_loss, ppo_value_loss,
                     entropy_bonus, kl_penalty, gae_advantages)


@dataclass
class PPOConfig:
    learning_rate:    float = 1e-5
    value_lr:         float = 1e-4
    clip_eps:         float = 0.2
    value_clip_eps:   float = 0.2
    gamma:            float = 0.99
    lam:              float = 0.95
    kl_coef:          float = 0.1
    entropy_coef:     float = 0.01
    value_coef:       float = 0.5
    ppo_epochs:       int   = 4
    rollout_batch:    int   = 8
    max_new_tokens:   int   = 128
    max_steps:        int   = 1_000
    log_every:        int   = 10
    temperature:      float = 1.0


# ── JIT-compiled update steps ────────────────────────────────────────────────

@jax.jit
def _policy_update(policy_state: train_state.TrainState,
                   batch: Dict[str, jnp.ndarray],
                   cfg: PPOConfig) -> Tuple[train_state.TrainState, Dict]:
    input_ids    = batch['input_ids']
    old_lp       = batch['old_log_probs']
    advantages   = batch['advantages']
    mask         = batch.get('mask', None)
    logits_old   = batch.get('logits', None)

    def loss_fn(params):
        logits   = policy_state.apply_fn(params, input_ids[:, :-1], mask)
        labels   = input_ids[:, 1:]
        lp       = jax.nn.log_softmax(logits, axis=-1)
        token_lp = jnp.take_along_axis(lp, labels[..., None], axis=-1).squeeze(-1)

        pi_loss = ppo_policy_loss(token_lp, old_lp, advantages, cfg.clip_eps, mask)
        ent     = entropy_bonus(logits, mask)
        total   = pi_loss - cfg.entropy_coef * ent
        return total, {'policy_loss': pi_loss, 'entropy': ent}

    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(policy_state.params)
    policy_state = policy_state.apply_gradients(grads=grads)
    return policy_state, {**aux, 'total_loss': loss}


@jax.jit
def _value_update(value_state: train_state.TrainState,
                  batch: Dict[str, jnp.ndarray],
                  cfg: PPOConfig) -> Tuple[train_state.TrainState, Dict]:
    input_ids  = batch['input_ids']
    old_values = batch['old_values']
    returns    = batch['returns']
    mask       = batch.get('mask', None)

    def loss_fn(params):
        values = value_state.apply_fn(params, input_ids, mask)
        loss   = ppo_value_loss(values, old_values, returns, cfg.value_clip_eps, mask)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(value_state.params)
    value_state = value_state.apply_gradients(grads=grads)
    return value_state, {'value_loss': loss}


class PPOTrainer:
    """
    Full RLHF/PPO training loop.

    Parameters
    ----------
    model_config   : ModelConfig       — shared backbone config
    ppo_config     : PPOConfig
    reward_params  : dict              — frozen reward model params
    ref_params     : dict              — frozen SFT reference params
    rng_key        : jax PRNGKey
    """

    def __init__(self,
                 model_config:  ModelConfig,
                 ppo_config:    PPOConfig,
                 reward_params: Dict,
                 ref_params:    Dict,
                 rng_key        = None):
        self.cfg          = ppo_config
        self.model_config = model_config

        key = jax.random.PRNGKey(42) if rng_key is None else rng_key
        k1, k2 = jax.random.split(key)

        self.policy_state = create_policy_train_state(
            model_config, ppo_config.learning_rate, rng_key=k1)
        self.value_state  = create_value_train_state(
            model_config, ppo_config.value_lr, rng_key=k2)

        self.reward_model = RewardModel(model_config)
        self.reward_params = jax.lax.stop_gradient(reward_params)

        self.ref_model   = ReferenceModel(model_config, ref_params)
        self.history: Dict[str, list] = {
            'policy_loss': [], 'value_loss': [], 'reward': [],
            'kl': [], 'entropy': [], 'step': []
        }

    # ------------------------------------------------------------------
    def _score_responses(self,
                          input_ids: jnp.ndarray,
                          response_ids: jnp.ndarray) -> jnp.ndarray:
        """Concatenate prompt + response, score with reward model."""
        full_ids = jnp.concatenate([input_ids, response_ids], axis=1)
        rewards  = self.reward_model.apply(self.reward_params, full_ids)
        return rewards

    def _collect_rollout(self,
                          prompts: jnp.ndarray,
                          rng_key) -> Dict[str, jnp.ndarray]:
        """Greedy/temperature sampling — simplified rollout for illustration."""
        B, T_prompt = prompts.shape
        max_new     = self.cfg.max_new_tokens
        all_ids     = prompts

        old_log_probs_list = []

        for _ in range(max_new):
            logits   = self.policy_state.apply_fn(
                self.policy_state.params, all_ids)               # (B, T, V)
            next_lp  = jax.nn.log_softmax(logits[:, -1, :], axis=-1)  # (B, V)
            if self.cfg.temperature != 1.0:
                next_lp = next_lp / self.cfg.temperature
            rng_key, subkey = jax.random.split(rng_key)
            next_tok = jax.random.categorical(subkey, next_lp)  # (B,)
            old_log_probs_list.append(
                next_lp[jnp.arange(B), next_tok])               # (B,)
            all_ids  = jnp.concatenate(
                [all_ids, next_tok[:, None]], axis=1)

        response_ids  = all_ids[:, T_prompt:]                    # (B, max_new)
        old_log_probs = jnp.stack(old_log_probs_list, axis=1)   # (B, max_new)

        rewards  = self._score_responses(prompts, response_ids)  # (B,)
        kl_vals  = self.ref_model.kl_from_policy(
            old_log_probs, all_ids)                               # (B,)
        adjusted = rewards - self.cfg.kl_coef * kl_vals

        # Broadcast adjusted reward to (B, max_new) as final-step reward
        reward_seq = jnp.zeros_like(old_log_probs)
        reward_seq = reward_seq.at[:, -1].set(adjusted)

        values     = self.value_state.apply_fn(
            self.value_state.params, all_ids)[:, T_prompt:]      # (B, max_new)
        dones      = jnp.zeros_like(reward_seq)
        dones      = dones.at[:, -1].set(1.0)

        advantages, returns = gae_advantages(
            reward_seq, values, dones, self.cfg.gamma, self.cfg.lam)

        return {
            'input_ids':    all_ids,
            'old_log_probs': old_log_probs,
            'old_values':   values,
            'advantages':   advantages,
            'returns':      returns,
            'reward':       rewards,
            'kl':           kl_vals,
        }

    def train(self, prompt_loader: Iterator) -> Dict[str, list]:
        step   = 0
        rng    = jax.random.PRNGKey(0)
        t0     = time.time()

        while step < self.cfg.max_steps:
            for prompts in prompt_loader:
                prompts = jnp.array(prompts['input_ids'])
                rng, rollout_key = jax.random.split(rng)

                # Collect on-policy experience
                rollout = self._collect_rollout(prompts, rollout_key)

                # PPO update epochs
                for _ in range(self.cfg.ppo_epochs):
                    self.policy_state, p_metrics = _policy_update(
                        self.policy_state, rollout, self.cfg)
                    self.value_state,  v_metrics = _value_update(
                        self.value_state,  rollout, self.cfg)

                step += 1
                if step % self.cfg.log_every == 0:
                    mean_r  = float(rollout['reward'].mean())
                    mean_kl = float(rollout['kl'].mean())
                    elapsed = time.time() - t0
                    print(f'[PPO] step={step:4d}  '
                          f'reward={mean_r:.3f}  kl={mean_kl:.4f}  '
                          f'policy_loss={float(p_metrics["policy_loss"]):.4f}  '
                          f'value_loss={float(v_metrics["value_loss"]):.4f}  '
                          f'elapsed={elapsed:.1f}s')
                    self.history['policy_loss'].append(float(p_metrics['policy_loss']))
                    self.history['value_loss'].append(float(v_metrics['value_loss']))
                    self.history['reward'].append(mean_r)
                    self.history['kl'].append(mean_kl)
                    self.history['entropy'].append(float(p_metrics['entropy']))
                    self.history['step'].append(step)

                if step >= self.cfg.max_steps:
                    return self.history
        return self.history
