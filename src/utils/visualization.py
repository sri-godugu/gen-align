"""
Training curve and reward visualisation utilities.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


def _smooth(values: Sequence[float], window: int = 5) -> np.ndarray:
    """Exponential moving average smoothing."""
    arr = np.array(values, dtype=float)
    if len(arr) < window:
        return arr
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    return np.convolve(arr, weights[::-1], mode='valid')


def plot_training_curves(history: Dict[str, List[float]],
                          save_path: Optional[str] = None,
                          smooth_window: int = 5) -> None:
    """
    Plot loss (and optionally other scalars) stored in a history dict.

    history keys → subplot titles; each value is a list of scalars.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('[visualization] matplotlib not installed — skipping plot.')
        return

    keys = [k for k in history if k != 'step']
    n    = len(keys)
    if n == 0:
        return

    steps = history.get('step', list(range(len(history[keys[0]]))))

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, key in zip(axes, keys):
        vals = np.array(history[key])
        s    = _smooth(vals, smooth_window)
        ax.plot(steps[:len(vals)], vals, alpha=0.3, color='steelblue')
        ax.plot(steps[:len(s)], s, color='steelblue', linewidth=2)
        ax.set_title(key.replace('_', ' ').title())
        ax.set_xlabel('Step')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f'Training curve saved → {save_path}')
    else:
        plt.show()
    plt.close(fig)


def plot_reward_distribution(policy_rewards:    np.ndarray,
                               reference_rewards: np.ndarray,
                               save_path:         Optional[str] = None) -> None:
    """Side-by-side histogram of policy vs. reference rewards."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('[visualization] matplotlib not installed — skipping plot.')
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    bins    = np.linspace(
        min(policy_rewards.min(), reference_rewards.min()),
        max(policy_rewards.max(), reference_rewards.max()),
        30)

    ax.hist(reference_rewards, bins=bins, alpha=0.6, label='Reference (SFT)',
            color='orange', edgecolor='white')
    ax.hist(policy_rewards,    bins=bins, alpha=0.6, label='Aligned policy',
            color='steelblue', edgecolor='white')
    ax.axvline(reference_rewards.mean(), color='darkorange', linestyle='--',
               linewidth=1.5, label=f'Ref mean={reference_rewards.mean():.3f}')
    ax.axvline(policy_rewards.mean(), color='navy', linestyle='--',
               linewidth=1.5, label=f'Policy mean={policy_rewards.mean():.3f}')

    ax.set_xlabel('Reward')
    ax.set_ylabel('Count')
    ax.set_title('Reward Distribution: Aligned Policy vs Reference')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f'Reward distribution plot saved → {save_path}')
    else:
        plt.show()
    plt.close(fig)


def plot_rlhf_vs_dpo(rlhf_history: Dict[str, List[float]],
                      dpo_history:  Dict[str, List[float]],
                      metric:       str = 'reward',
                      save_path:    Optional[str] = None) -> None:
    """Overlay a single metric from RLHF and DPO training runs."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(7, 4))

    for label, history, color in [('RLHF/PPO', rlhf_history, 'steelblue'),
                                    ('DPO',      dpo_history,  'tomato')]:
        if metric in history:
            vals  = np.array(history[metric])
            steps = history.get('step', list(range(len(vals))))
            s     = _smooth(vals)
            ax.plot(steps[:len(vals)], vals, alpha=0.25, color=color)
            ax.plot(steps[:len(s)],    s,    color=color, linewidth=2, label=label)

    ax.set_title(f'{metric.replace("_", " ").title()}: RLHF/PPO vs DPO')
    ax.set_xlabel('Step')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f'Comparison plot saved → {save_path}')
    else:
        plt.show()
    plt.close(fig)
