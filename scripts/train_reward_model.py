"""
Train the reward model on human preference pairs.

Usage
-----
python scripts/train_reward_model.py --config configs/reward_config.yaml
"""
import argparse
import json
import yaml
from pathlib import Path

import jax

from src.models.base import ModelConfig
from src.alignment.reward_trainer import RewardTrainer
from src.data.data_utils import JSONLDataset
from src.utils.jax_utils import save_checkpoint, save_params
from src.utils.visualization import plot_training_curves


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/reward_config.yaml')
    p.add_argument('--output_dir', default='outputs/reward_model')
    return p.parse_args()


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

    train_ds = JSONLDataset(cfg['data']['train_path'],
                             max_length=cfg['model']['max_seq_len'])
    eval_ds  = (JSONLDataset(cfg['data']['eval_path'],
                              max_length=cfg['model']['max_seq_len'])
                if cfg['data'].get('eval_path') else None)

    train_loader = train_ds.get_dataloader(cfg['training']['batch_size'])
    eval_loader  = (eval_ds.get_dataloader(cfg['training']['batch_size'], shuffle=False)
                    if eval_ds else None)

    trainer = RewardTrainer(
        config        = model_cfg,
        learning_rate = cfg['training']['learning_rate'],
        max_steps     = cfg['training']['max_steps'],
        log_every     = cfg['training'].get('log_every', 100),
        eval_every    = cfg['training'].get('eval_every', 500),
    )

    print(f'[RM] Training reward model on {jax.device_count()} device(s)...')
    history = trainer.train(train_loader, eval_loader)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    save_checkpoint(trainer.state, str(out / 'checkpoints'))
    save_params(trainer.params, str(out / 'reward_params.pkl'))

    with open(out / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    plot_training_curves(history, save_path=str(out / 'reward_training.png'))
    print('[RM] Training complete.')


if __name__ == '__main__':
    main()
