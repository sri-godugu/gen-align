"""
Evaluate an aligned policy and compare RLHF/PPO vs DPO methods.

Usage
-----
python scripts/evaluate.py \
    --ppo_params  outputs/ppo/ppo_policy_params.pkl \
    --dpo_params  outputs/dpo/dpo_policy_params.pkl \
    --ref_params  outputs/sft/sft_params.pkl \
    --reward_params outputs/reward_model/reward_params.pkl \
    --test_data   data/test_preferences.jsonl \
    --output_dir  results/
"""
import argparse
import json
from pathlib import Path

import numpy as np

from src.models.base import ModelConfig
from src.evaluation.evaluator import AlignmentEvaluator
from src.evaluation.metrics import compute_all_metrics
from src.data.data_utils import JSONLDataset
from src.utils.jax_utils import load_params
from src.utils.visualization import plot_rlhf_vs_dpo, plot_reward_distribution


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ppo_params',    required=False, default=None)
    p.add_argument('--dpo_params',    required=False, default=None)
    p.add_argument('--ref_params',    required=True)
    p.add_argument('--reward_params', required=True)
    p.add_argument('--test_data',     required=True)
    p.add_argument('--output_dir',    default='results')
    p.add_argument('--max_seq_len',   type=int, default=512)
    p.add_argument('--hidden_size',   type=int, default=768)
    p.add_argument('--n_layers',      type=int, default=12)
    p.add_argument('--n_heads',       type=int, default=12)
    return p.parse_args()


def _build_model_config(args) -> ModelConfig:
    return ModelConfig(
        vocab_size  = 50257,
        max_seq_len = args.max_seq_len,
        hidden_size = args.hidden_size,
        n_layers    = args.n_layers,
        n_heads     = args.n_heads,
    )


def main():
    args = parse_args()
    out  = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_cfg     = _build_model_config(args)
    ref_params    = load_params(args.ref_params)
    reward_params = load_params(args.reward_params)

    test_ds     = JSONLDataset(args.test_data, max_length=args.max_seq_len)
    test_loader = test_ds.get_dataloader(batch_size=16, shuffle=False)

    reports = {}

    for tag, param_path in [('ppo', args.ppo_params), ('dpo', args.dpo_params)]:
        if param_path is None or not Path(param_path).exists():
            continue
        policy_params = load_params(param_path)
        evaluator     = AlignmentEvaluator(
            model_cfg, policy_params, ref_params, reward_params)
        report = evaluator.evaluate(
            test_loader,
            max_batches=200,
            output_path=str(out / f'{tag}_eval.json'))
        reports[tag] = report

    # Comparison table
    if reports:
        comparison = {tag: {
            'win_rate':    r['win_rate'],
            'mean_reward': r['reward']['mean'],
            'mean_delta':  r['delta']['mean_delta'],
            'mean_kl':     r.get('mean_kl', None),
        } for tag, r in reports.items()}

        with open(out / 'alignment_comparison.json', 'w') as f:
            json.dump(comparison, f, indent=2)
        print('\nMethod comparison:')
        for tag, row in comparison.items():
            print(f'  {tag.upper():6s}  '
                  f'win_rate={row["win_rate"]:.3f}  '
                  f'reward={row["mean_reward"]:.4f}  '
                  f'delta={row["mean_delta"]:.4f}')

    print(f'\nResults saved to {out}/')


if __name__ == '__main__':
    main()
