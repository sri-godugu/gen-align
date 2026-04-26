"""
Train a language model with Supervised Fine-Tuning (SFT).

Usage
-----
python scripts/train_sft.py --config configs/sft_config.yaml
"""
import argparse
import json
import yaml
from pathlib import Path

import jax
import jax.numpy as jnp

from src.models.base import ModelConfig
from src.alignment.sft import SFTTrainer
from src.data.data_utils import JSONLDataset, batch_encode
from src.utils.jax_utils import save_checkpoint, save_params
from src.utils.logging_utils import CSVLogger
from src.utils.visualization import plot_training_curves


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/sft_config.yaml')
    p.add_argument('--output_dir', default='outputs/sft')
    return p.parse_args()


def main():
    args   = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = ModelConfig(
        vocab_size   = cfg['model']['vocab_size'],
        max_seq_len  = cfg['model']['max_seq_len'],
        hidden_size  = cfg['model']['hidden_size'],
        n_layers     = cfg['model']['n_layers'],
        n_heads      = cfg['model']['n_heads'],
    )

    train_ds = JSONLDataset(
        cfg['data']['train_path'],
        max_length=cfg['model']['max_seq_len'])
    eval_ds  = JSONLDataset(
        cfg['data']['eval_path'],
        max_length=cfg['model']['max_seq_len']) if cfg['data'].get('eval_path') else None

    train_loader = train_ds.get_dataloader(cfg['training']['batch_size'], shuffle=True)
    eval_loader  = (eval_ds.get_dataloader(cfg['training']['batch_size'], shuffle=False)
                    if eval_ds else None)

    trainer = SFTTrainer(
        config        = model_cfg,
        learning_rate = cfg['training']['learning_rate'],
        weight_decay  = cfg['training'].get('weight_decay', 0.01),
        max_steps     = cfg['training']['max_steps'],
        log_every     = cfg['training'].get('log_every', 100),
        eval_every    = cfg['training'].get('eval_every', 500),
    )

    print(f'[SFT] Starting training on {jax.device_count()} device(s)...')
    history = trainer.train(train_loader, eval_loader)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    save_checkpoint(trainer.state, str(out / 'checkpoints'))
    save_params(trainer.params, str(out / 'sft_params.pkl'))

    with open(out / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    plot_training_curves(history, save_path=str(out / 'training_curve.png'))
    print('[SFT] Training complete.')


if __name__ == '__main__':
    main()
