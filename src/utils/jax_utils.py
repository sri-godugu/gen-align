"""
JAX utility helpers: checkpointing, device sharding, and random key management.
"""
from __future__ import annotations
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from flax.training import train_state
from flax import serialization


# ── Checkpointing ─────────────────────────────────────────────────────────────

def save_checkpoint(state: train_state.TrainState,
                     path:  str,
                     step:  int = 0) -> None:
    """Serialise a Flax TrainState to disk using msgpack."""
    ckpt_dir = Path(path)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f'checkpoint_{step:06d}.msgpack'
    bytes_out = serialization.to_bytes(state)
    ckpt_path.write_bytes(bytes_out)
    print(f'Checkpoint saved → {ckpt_path}')


def load_checkpoint(state: train_state.TrainState,
                     path:  str) -> train_state.TrainState:
    """Load the latest checkpoint from a directory into an existing TrainState."""
    ckpt_dir = Path(path)
    ckpts    = sorted(ckpt_dir.glob('checkpoint_*.msgpack'))
    if not ckpts:
        raise FileNotFoundError(f'No checkpoints found in {ckpt_dir}')
    latest = ckpts[-1]
    state  = serialization.from_bytes(state, latest.read_bytes())
    print(f'Checkpoint loaded ← {latest}')
    return state


def save_params(params: Dict, path: str) -> None:
    """Save raw parameter dict with pickle (useful for reference/reward models)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(params, f)
    print(f'Params saved → {path}')


def load_params(path: str) -> Dict:
    with open(path, 'rb') as f:
        return pickle.load(f)


# ── Device utilities ──────────────────────────────────────────────────────────

def get_devices() -> Tuple[int, Any]:
    """Return (n_devices, devices) for the available JAX backend."""
    devices  = jax.devices()
    n        = len(devices)
    backend  = jax.default_backend()
    print(f'JAX backend: {backend}  |  devices: {n}')
    return n, devices


def replicate(tree: Any) -> Any:
    """Replicate a pytree across all available devices (for pmap)."""
    return jax.device_put_replicated(tree, jax.devices())


def unreplicate(tree: Any) -> Any:
    """Take the first device's copy of a replicated pytree."""
    return jax.tree_util.tree_map(lambda x: x[0], tree)


# ── PRNGKey management ────────────────────────────────────────────────────────

def make_keys(base_key, n: int):
    """Split a base PRNGKey into n independent subkeys."""
    return jax.random.split(base_key, n)


# ── Gradient utilities ────────────────────────────────────────────────────────

def global_norm(grads) -> jnp.ndarray:
    """Compute the global L2 norm of a gradient pytree."""
    leaves = jax.tree_util.tree_leaves(grads)
    return jnp.sqrt(sum(jnp.sum(g ** 2) for g in leaves))
