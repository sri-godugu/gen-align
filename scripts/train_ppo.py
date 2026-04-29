"""
Run RLHF/PPO alignment training.

Usage
-----
python scripts/train_ppo.py --config configs/ppo_config.yaml \
    --ref_params   outputs/sft/sft_params.pkl \
    --reward_params outputs/reward_model/reward_params.pkl
"""
import argparse
import json
import yaml
from pathlib import Path

import jax

from src.models.base import ModelConfig
from src.alignment.ppo import PPOTrainer, PPOConfig
from src.data.data_utils import PromptDataset
from src.utils.jax_utils import load_params, save_checkpoint, save_params
from src.utils.visualization import plot_training_curves


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--config',        default='configs/ppo_config.yaml')
    args.add_argument('--ref_params',    default='outputs/sft/sft_params.pkl')
    args.add_argument('--reward_params', default='outputs/reward_model/reward_params.pkl')
    args.add_argument('--output_dir',    default='outputs/ppo')
    return args.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = ModelConfig(
        vocab_size  = cfg['model']['vocab_size'],
        max_seq_len = cfg['model']['max_seq_len'],
        hidden_size = cfg['model']['hidden_size'],
        n_layers    = cfg['model']['n_layers'],
        n_heads     = cfg['model']['n_heads'],
    )

    ppo_cfg = PPOConfig(
        learning_rate  = cfg['training']['policy_lr'],
        value_lr       = cfg['training']['value_lr'],
        clip_eps       = cfg['training'].get('clip_eps', 0.2),
        kl_coef        = cfg['training'].get('kl_coef', 0.1),
        entropy_coef   = cfg['training'].get('entropy_coef', 0.01),
        ppo_epochs     = cfg['training'].get('ppo_epochs', 4),
        rollout_batch  = cfg['training'].get('rollout_batch', 8),
        max_new_tokens = cfg['training'].get('max_new_tokens', 128),
        max_steps      = cfg['training']['max_steps'],
        log_every      = cfg['training'].get('log_every', 10),
    )

    ref_params    = load_params(args.ref_params)
    reward_params = load_params(args.reward_params)

    prompts_raw = cfg['data'].get('prompts', ['Tell me about space exploration.'] * 100)
    prompt_ds   = PromptDataset(prompts_raw,
                                 max_length=cfg['model']['max_seq_len'])
    prompt_loader = prompt_ds.get_dataloader(ppo_cfg.rollout_batch)

    trainer = PPOTrainer(
        model_config  = model_cfg,
        ppo_config    = ppo_cfg,
        reward_params = reward_params,
        ref_params    = ref_params,
    )

    print(f'[PPO] Starting RLHF on {jax.device_count()} device(s)...')
    history = trainer.train(prompt_loader)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    save_checkpoint(trainer.policy_state, str(out / 'checkpoints'))
    save_params(trainer.policy_state.params, str(out / 'ppo_policy_params.pkl'))

    with open(out / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    plot_training_curves(history, save_path=str(out / 'ppo_training.png'))
    print('[PPO] Training complete.')


if __name__ == '__main__':
    main()
