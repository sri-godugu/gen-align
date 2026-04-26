"""
Reward model trainer: learns a scalar reward from human preference pairs.

Training signal: Bradley-Terry loss on (chosen, rejected) pairs.
Supports both single-objective and multi-objective reward models.
"""
from __future__ import annotations
import time
from typing import Dict, Iterator, Optional, Tuple

import jax
import jax.numpy as jnp
from flax.training import train_state

from ..models.reward_model import RewardModel, create_reward_train_state
from ..models.base import ModelConfig
from .losses import reward_loss


@jax.jit
def _train_step(state: train_state.TrainState,
                batch: Dict[str, jnp.ndarray]) -> Tuple[train_state.TrainState, Dict]:
    chosen_ids    = batch['chosen_ids']
    rejected_ids  = batch['rejected_ids']
    chosen_mask   = batch.get('chosen_mask',   None)
    rejected_mask = batch.get('rejected_mask', None)

    def loss_fn(params):
        r_w = state.apply_fn(params, chosen_ids,   chosen_mask)
        r_l = state.apply_fn(params, rejected_ids, rejected_mask)
        loss, acc = reward_loss(r_w, r_l)
        return loss, acc

    (loss, acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, {'loss': loss, 'accuracy': acc}


@jax.jit
def _eval_step(state: train_state.TrainState,
               batch: Dict[str, jnp.ndarray]) -> Dict:
    chosen_ids    = batch['chosen_ids']
    rejected_ids  = batch['rejected_ids']
    chosen_mask   = batch.get('chosen_mask',   None)
    rejected_mask = batch.get('rejected_mask', None)
    r_w = state.apply_fn(state.params, chosen_ids,   chosen_mask)
    r_l = state.apply_fn(state.params, rejected_ids, rejected_mask)
    loss, acc = reward_loss(r_w, r_l)
    return {'loss': loss, 'accuracy': acc}


class RewardTrainer:
    """
    Trains a RewardModel on preference pairs.

    Parameters
    ----------
    config        : ModelConfig
    learning_rate : float
    max_steps     : int
    log_every     : int
    eval_every    : int
    """

    def __init__(self,
                 config:        ModelConfig,
                 learning_rate: float = 1e-4,
                 max_steps:     int   = 5_000,
                 log_every:     int   = 100,
                 eval_every:    int   = 500,
                 rng_key        = None):
        self.config     = config
        self.max_steps  = max_steps
        self.log_every  = log_every
        self.eval_every = eval_every
        self.state      = create_reward_train_state(config, learning_rate, rng_key)
        self.history: Dict[str, list] = {
            'train_loss': [], 'train_acc': [],
            'eval_loss':  [], 'eval_acc':  [],
            'step': []
        }

    # ------------------------------------------------------------------
    def train(self,
              train_loader: Iterator,
              eval_loader:  Optional[Iterator] = None) -> Dict[str, list]:
        step    = 0
        r_loss  = 0.0
        r_acc   = 0.0
        t0      = time.time()

        while True:
            for batch in train_loader:
                batch = {k: jnp.array(v) for k, v in batch.items()}
                self.state, metrics = _train_step(self.state, batch)
                r_loss += float(metrics['loss'])
                r_acc  += float(metrics['accuracy'])
                step   += 1

                if step % self.log_every == 0:
                    avg_loss = r_loss / self.log_every
                    avg_acc  = r_acc  / self.log_every
                    r_loss = r_acc = 0.0
                    elapsed = time.time() - t0
                    print(f'[RM] step={step:5d}  loss={avg_loss:.4f}  '
                          f'acc={avg_acc:.3f}  elapsed={elapsed:.1f}s')
                    self.history['train_loss'].append(avg_loss)
                    self.history['train_acc'].append(avg_acc)
                    self.history['step'].append(step)

                if eval_loader is not None and step % self.eval_every == 0:
                    vl, va = self._evaluate(eval_loader)
                    self.history['eval_loss'].append(vl)
                    self.history['eval_acc'].append(va)
                    print(f'[RM] step={step:5d}  val_loss={vl:.4f}  val_acc={va:.3f}')

                if 0 < self.max_steps <= step:
                    print(f'[RM] Reached max_steps={self.max_steps}.')
                    return self.history
        return self.history

    def _evaluate(self, loader: Iterator) -> Tuple[float, float]:
        tl, ta, n = 0.0, 0.0, 0
        for batch in loader:
            batch = {k: jnp.array(v) for k, v in batch.items()}
            m = _eval_step(self.state, batch)
            tl += float(m['loss'])
            ta += float(m['accuracy'])
            n  += 1
            if n >= 50:
                break
        return tl / max(n, 1), ta / max(n, 1)

    @property
    def params(self):
        return self.state.params
