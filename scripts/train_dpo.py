"""
Run DPO alignment training.

Usage
-----
python scripts/train_dpo.py --config configs/dpo_config.yaml \
    --ref_params outputs/sft/sft_params.pkl
"""
import argparse
import json
import yaml
from pathlib import Path

import jax

from src.models.base import ModelConfig
from src.alignment.dpo import DPOTrainer, DPOConfig
from src.data.data_utils import JSONLDataset
from src.utils.jax_utils import load_params, save_checkpoint, save_params
from src.utils.visualization import plot_training_curves


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config',     default='configs/dpo_config.yaml')
    p.add_argument('--ref_params', default='outputs/sft/sft_params.pkl')
    p.add_argument('--output_dir', default='outputs/dpo')
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

    dpo_cfg = DPOConfig(
        beta          = cfg['training'].get('beta', 0.1),
        learning_rate = cfg['training']['learning_rate'],
        weight_decay  = cfg['training'].get('weight_decay', 0.01),
        max_steps     = cfg['training']['max_steps'],
        log_every     = cfg['training'].get('log_every', 50),
        eval_every    = cfg['training'].get('eval_every', 500),
    )

    ref_params = load_params(args.ref_params)

    train_ds = JSONLDataset(cfg['data']['train_path'],
                             max_length=cfg['model']['max_seq_len'])
    eval_ds  = (JSONLDataset(cfg['data']['eval_path'],
                              max_length=cfg['model']['max_seq_len'])
                if cfg['data'].get('eval_path') else None)

    train_loader = train_ds.get_dataloader(cfg['training']['batch_size'])
    eval_loader  = (eval_ds.get_dataloader(cfg['training']['batch_size'], shuffle=False)
                    if eval_ds else None)

    trainer = DPOTrainer(
        model_config = model_cfg,
        dpo_config   = dpo_cfg,
        ref_params   = ref_params,
    )

    print(f'[DPO] Starting DPO training on {jax.device_count()} device(s)...')
    history = trainer.train(train_loader, eval_loader)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    save_checkpoint(trainer.state, str(out / 'checkpoints'))
    save_params(trainer.params, str(out / 'dpo_policy_params.pkl'))

    with open(out / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    plot_training_curves(history, save_path=str(out / 'dpo_training.png'))
    print('[DPO] Training complete.')


if __name__ == '__main__':
    main()
