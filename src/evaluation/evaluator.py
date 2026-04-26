"""
Full evaluation pipeline: runs a trained policy against a reference
model on a held-out preference test set and produces a summary report.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np
import jax
import jax.numpy as jnp

from ..models.policy import PolicyModel
from ..models.reward_model import RewardModel
from ..models.reference_model import ReferenceModel
from ..models.base import ModelConfig
from .metrics import compute_all_metrics, compute_perplexity
from .safety import SafetyScorer


class AlignmentEvaluator:
    """
    Evaluates an aligned policy against a frozen reference.

    Parameters
    ----------
    model_config    : ModelConfig
    policy_params   : dict      — trained policy parameters
    ref_params      : dict      — SFT reference parameters
    reward_params   : dict      — reward model parameters
    tokenizer       : optional tokenizer for decode
    """

    def __init__(self,
                 model_config:  ModelConfig,
                 policy_params: Dict,
                 ref_params:    Dict,
                 reward_params: Dict,
                 tokenizer      = None):
        self.config        = model_config
        self.policy_params = policy_params
        self.ref_params    = ref_params
        self.reward_params = reward_params
        self.tokenizer     = tokenizer

        self.policy  = PolicyModel(model_config)
        self.ref     = ReferenceModel(model_config, ref_params)
        self.reward  = RewardModel(model_config)
        self.safety  = SafetyScorer()

    # ------------------------------------------------------------------
    def evaluate(self,
                 test_loader:   Iterator,
                 max_batches:   int = 100,
                 output_path:   Optional[str] = None) -> Dict:
        policy_rewards, ref_rewards, kl_vals = [], [], []
        generations: List[str] = []

        t0 = time.time()
        for i, batch in enumerate(test_loader):
            if i >= max_batches:
                break
            batch = {k: jnp.array(v) for k, v in batch.items()}

            chosen_ids = batch['chosen_ids']
            mask       = batch.get('chosen_mask', None)

            # Policy reward
            pr = self.reward.apply(self.reward_params, chosen_ids, mask)
            policy_rewards.append(np.array(pr))

            # Reference reward (score same text under ref model — proxy comparison)
            rr = self.reward.apply(self.reward_params,
                                   batch['rejected_ids'],
                                   batch.get('rejected_mask', None))
            ref_rewards.append(np.array(rr))

            # KL from reference
            kl = self.ref.kl_from_policy(
                self._get_policy_lp(chosen_ids, mask), chosen_ids, mask)
            kl_vals.append(np.array(kl))

        policy_rewards = np.concatenate(policy_rewards)
        ref_rewards    = np.concatenate(ref_rewards)
        kl_array       = np.concatenate(kl_vals)

        elapsed = time.time() - t0
        report = compute_all_metrics(
            policy_rewards, ref_rewards, kl_array, generations or None)
        report['eval_time_s'] = round(elapsed, 2)
        report['n_samples']   = len(policy_rewards)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f'Evaluation report saved → {output_path}')

        self._print_report(report)
        return report

    def _get_policy_lp(self,
                        input_ids: jnp.ndarray,
                        mask:      Optional[jnp.ndarray]) -> jnp.ndarray:
        """Per-token log-probs from the trained policy."""
        logits   = self.policy.apply(self.policy_params, input_ids[:, :-1], mask)
        labels   = input_ids[:, 1:]
        lp       = jax.nn.log_softmax(logits, axis=-1)
        token_lp = jnp.take_along_axis(lp, labels[..., None], axis=-1).squeeze(-1)
        return token_lp

    @staticmethod
    def _print_report(report: Dict) -> None:
        print('\n' + '=' * 60)
        print('  Alignment Evaluation Summary')
        print('=' * 60)
        print(f"  Win rate         : {report['win_rate']:.3f}")
        print(f"  Policy reward    : {report['reward']['mean']:.4f} "
              f"± {report['reward']['std']:.4f}")
        print(f"  Ref    reward    : {report['ref_reward']['mean']:.4f} "
              f"± {report['ref_reward']['std']:.4f}")
        print(f"  Mean Δ reward    : {report['delta']['mean_delta']:.4f}")
        print(f"  % improved       : {report['delta']['pct_improved']:.2%}")
        if 'mean_kl' in report:
            print(f"  Mean KL (π||ref) : {report['mean_kl']:.4f}")
        if 'distinct_2' in report:
            print(f"  Distinct-2       : {report['distinct_2']:.4f}")
        print(f"  Samples          : {report['n_samples']}")
        print(f"  Eval time        : {report['eval_time_s']}s")
        print('=' * 60 + '\n')
