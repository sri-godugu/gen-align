"""
Supervised Fine-Tuning (SFT) trainer.

Trains the policy on curated demonstration data using
standard cross-entropy next-token prediction.
"""
from __future__ import annotations
import time
from typing import Dict, Iterator, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from ..models.policy import PolicyModel, create_policy_train_state
from ..models.base import ModelConfig
from .losses import sft_loss


@jax.jit
def _train_step(state: train_state.TrainState,
                batch: Dict[str, jnp.ndarray]) -> Tuple[train_state.TrainState, Dict]:
    input_ids = batch['input_ids']
    mask      = batch.get('attention_mask', None)

    def loss_fn(params):
        logits = state.apply_fn(params, input_ids[:, :-1], mask)
        labels = input_ids[:, 1:]
        lbl_mask = mask[:, 1:] if mask is not None else None
        loss = sft_loss(logits, labels, lbl_mask)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, {'loss': loss}


@jax.jit
def _eval_step(state: train_state.TrainState,
               batch: Dict[str, jnp.ndarray]) -> Dict:
    input_ids = batch['input_ids']
    mask      = batch.get('attention_mask', None)
    logits    = state.apply_fn(state.params, input_ids[:, :-1], mask)
    labels    = input_ids[:, 1:]
    lbl_mask  = mask[:, 1:] if mask is not None else None
    loss      = sft_loss(logits, labels, lbl_mask)
    return {'loss': loss}


class SFTTrainer:
    """
    Fine-tunes a language model on demonstration data.

    Parameters
    ----------
    config       : ModelConfig
    learning_rate: float
    weight_decay : float
    max_steps    : int    — training steps; set -1 to run full epochs
    log_every    : int    — print metrics every N steps
    eval_every   : int    — run validation every N steps
    """

    def __init__(self,
                 config:        ModelConfig,
                 learning_rate: float = 2e-4,
                 weight_decay:  float = 0.01,
                 max_steps:     int   = 10_000,
                 log_every:     int   = 100,
                 eval_every:    int   = 500,
                 rng_key        = None):
        self.config        = config
        self.max_steps     = max_steps
        self.log_every     = log_every
        self.eval_every    = eval_every
        self.state         = create_policy_train_state(
            config, learning_rate, weight_decay, rng_key)
        self.history: Dict[str, list] = {'train_loss': [], 'eval_loss': [], 'step': []}

    # ------------------------------------------------------------------
    def train(self,
              train_loader: Iterator,
              eval_loader:  Optional[Iterator] = None) -> Dict[str, list]:
        step       = 0
        epoch      = 0
        running    = 0.0
        t0         = time.time()

        while True:
            epoch += 1
            for batch in train_loader:
                batch = {k: jnp.array(v) for k, v in batch.items()}
                self.state, metrics = _train_step(self.state, batch)
                running += float(metrics['loss'])
                step    += 1

                if step % self.log_every == 0:
                    avg = running / self.log_every
                    running = 0.0
                    elapsed = time.time() - t0
                    print(f'[SFT] step={step:6d}  loss={avg:.4f}  '
                          f'elapsed={elapsed:.1f}s')
                    self.history['train_loss'].append(avg)
                    self.history['step'].append(step)

                if eval_loader is not None and step % self.eval_every == 0:
                    val_loss = self._evaluate(eval_loader)
                    self.history['eval_loss'].append(val_loss)
                    print(f'[SFT] step={step:6d}  val_loss={val_loss:.4f}')

                if 0 < self.max_steps <= step:
                    print(f'[SFT] Reached max_steps={self.max_steps}.')
                    return self.history
        return self.history

    def _evaluate(self, loader: Iterator) -> float:
        total, n = 0.0, 0
        for batch in loader:
            batch = {k: jnp.array(v) for k, v in batch.items()}
            m = _eval_step(self.state, batch)
            total += float(m['loss'])
            n     += 1
            if n >= 50:         # cap eval at 50 batches
                break
        return total / max(n, 1)

    @property
    def params(self):
        return self.state.params
