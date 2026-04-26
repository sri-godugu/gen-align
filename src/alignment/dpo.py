"""
Direct Preference Optimization (DPO) trainer.

DPO bypasses explicit reward modelling and optimises the policy
directly from preference pairs using a closed-form re-parameterisation
of the RLHF objective.

Reference: Rafailov et al., "Direct Preference Optimization:
           Your Language Model is Secretly a Reward Model" (2023).
"""
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Tuple

import jax
import jax.numpy as jnp
from flax.training import train_state

from ..models.policy import PolicyModel, create_policy_train_state
from ..models.reference_model import ReferenceModel
from ..models.base import ModelConfig
from .losses import dpo_loss


@dataclass
class DPOConfig:
    beta:          float = 0.1    # KL regularisation strength
    learning_rate: float = 5e-5
    weight_decay:  float = 0.01
    max_steps:     int   = 5_000
    log_every:     int   = 50
    eval_every:    int   = 500


def _sequence_log_prob(apply_fn, params,
                        input_ids: jnp.ndarray,
                        mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """(B,) sum of token log-probs for the sequence."""
    logits   = apply_fn(params, input_ids[:, :-1], mask)
    labels   = input_ids[:, 1:]
    lp       = jax.nn.log_softmax(logits, axis=-1)
    token_lp = jnp.take_along_axis(lp, labels[..., None], axis=-1).squeeze(-1)
    if mask is not None:
        token_lp = token_lp * mask[:, 1:]
    return token_lp.sum(axis=-1)


@jax.jit
def _train_step(policy_state: train_state.TrainState,
                ref_params:   Dict,
                apply_fn,
                batch:        Dict[str, jnp.ndarray],
                beta:         float) -> Tuple[train_state.TrainState, Dict]:
    chosen_ids    = batch['chosen_ids']
    rejected_ids  = batch['rejected_ids']
    chosen_mask   = batch.get('chosen_mask',   None)
    rejected_mask = batch.get('rejected_mask', None)

    # Reference log-probs (no gradient)
    ref_chosen_lp   = jax.lax.stop_gradient(
        _sequence_log_prob(apply_fn, ref_params, chosen_ids,   chosen_mask))
    ref_rejected_lp = jax.lax.stop_gradient(
        _sequence_log_prob(apply_fn, ref_params, rejected_ids, rejected_mask))

    def loss_fn(params):
        policy_chosen_lp   = _sequence_log_prob(apply_fn, params,
                                                 chosen_ids,   chosen_mask)
        policy_rejected_lp = _sequence_log_prob(apply_fn, params,
                                                 rejected_ids, rejected_mask)
        loss, aux = dpo_loss(policy_chosen_lp, policy_rejected_lp,
                             ref_chosen_lp,    ref_rejected_lp,
                             beta=beta)
        return loss, aux

    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(policy_state.params)
    policy_state = policy_state.apply_gradients(grads=grads)
    return policy_state, {'loss': loss, **aux}


@jax.jit
def _eval_step(policy_state: train_state.TrainState,
               ref_params:   Dict,
               apply_fn,
               batch:        Dict[str, jnp.ndarray],
               beta:         float) -> Dict:
    chosen_ids    = batch['chosen_ids']
    rejected_ids  = batch['rejected_ids']
    chosen_mask   = batch.get('chosen_mask',   None)
    rejected_mask = batch.get('rejected_mask', None)

    ref_chosen_lp   = _sequence_log_prob(apply_fn, ref_params, chosen_ids,   chosen_mask)
    ref_rejected_lp = _sequence_log_prob(apply_fn, ref_params, rejected_ids, rejected_mask)

    policy_chosen_lp   = _sequence_log_prob(apply_fn, policy_state.params,
                                             chosen_ids,   chosen_mask)
    policy_rejected_lp = _sequence_log_prob(apply_fn, policy_state.params,
                                             rejected_ids, rejected_mask)
    loss, aux = dpo_loss(policy_chosen_lp, policy_rejected_lp,
                         ref_chosen_lp,    ref_rejected_lp,
                         beta=beta)
    return {'loss': loss, **aux}


class DPOTrainer:
    """
    Trains a language model with DPO on preference pairs.

    Parameters
    ----------
    model_config : ModelConfig
    dpo_config   : DPOConfig
    ref_params   : dict   — frozen SFT reference parameters
    rng_key      : PRNGKey
    """

    def __init__(self,
                 model_config: ModelConfig,
                 dpo_config:   DPOConfig,
                 ref_params:   Dict,
                 rng_key       = None):
        self.cfg        = dpo_config
        self.ref_params = jax.lax.stop_gradient(ref_params)
        self.state      = create_policy_train_state(
            model_config, dpo_config.learning_rate, dpo_config.weight_decay, rng_key)
        self.apply_fn   = self.state.apply_fn
        self.history: Dict[str, list] = {
            'loss': [], 'reward_margin': [], 'win_rate': [],
            'eval_loss': [], 'eval_win_rate': [], 'step': []
        }

    # ------------------------------------------------------------------
    def train(self,
              train_loader: Iterator,
              eval_loader:  Optional[Iterator] = None) -> Dict[str, list]:
        step   = 0
        r_loss = 0.0
        r_margin = 0.0
        r_wr   = 0.0
        t0     = time.time()

        while True:
            for batch in train_loader:
                batch = {k: jnp.array(v) for k, v in batch.items()}
                self.state, metrics = _train_step(
                    self.state, self.ref_params, self.apply_fn, batch, self.cfg.beta)

                r_loss   += float(metrics['loss'])
                r_margin += float(metrics['reward_margin'])
                r_wr     += float(metrics['win_rate'])
                step     += 1

                if step % self.cfg.log_every == 0:
                    n = self.cfg.log_every
                    avg_l  = r_loss   / n
                    avg_m  = r_margin / n
                    avg_wr = r_wr     / n
                    r_loss = r_margin = r_wr = 0.0
                    elapsed = time.time() - t0
                    print(f'[DPO] step={step:5d}  loss={avg_l:.4f}  '
                          f'margin={avg_m:.4f}  win_rate={avg_wr:.3f}  '
                          f'elapsed={elapsed:.1f}s')
                    self.history['loss'].append(avg_l)
                    self.history['reward_margin'].append(avg_m)
                    self.history['win_rate'].append(avg_wr)
                    self.history['step'].append(step)

                if eval_loader is not None and step % self.cfg.eval_every == 0:
                    ev = self._evaluate(eval_loader)
                    self.history['eval_loss'].append(ev['loss'])
                    self.history['eval_win_rate'].append(ev['win_rate'])
                    print(f'[DPO] step={step:5d}  val_loss={ev["loss"]:.4f}  '
                          f'val_win_rate={ev["win_rate"]:.3f}')

                if 0 < self.cfg.max_steps <= step:
                    print(f'[DPO] Reached max_steps={self.cfg.max_steps}.')
                    return self.history
        return self.history

    def _evaluate(self, loader: Iterator) -> Dict[str, float]:
        tl, tw, n = 0.0, 0.0, 0
        for batch in loader:
            batch = {k: jnp.array(v) for k, v in batch.items()}
            m = _eval_step(self.state, self.ref_params, self.apply_fn, batch, self.cfg.beta)
            tl += float(m['loss'])
            tw += float(m['win_rate'])
            n  += 1
            if n >= 50:
                break
        return {'loss': tl / max(n, 1), 'win_rate': tw / max(n, 1)}

    @property
    def params(self):
        return self.state.params
