"""
Experiment logging: CSV and optional Weights & Biases integration.
"""
from __future__ import annotations
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class CSVLogger:
    """Appends scalar metrics to a CSV file every time log() is called."""

    def __init__(self, path: str, fields: List[str]):
        self.path   = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fields = ['timestamp', 'step'] + fields
        self._file  = None
        self._writer: Optional[csv.DictWriter] = None

    def __enter__(self):
        self._file   = open(self.path, 'w', newline='')
        self._writer = csv.DictWriter(self._file, fieldnames=self.fields,
                                       extrasaction='ignore')
        self._writer.writeheader()
        return self

    def __exit__(self, *_):
        if self._file:
            self._file.close()

    def log(self, step: int, metrics: Dict[str, Any]) -> None:
        row = {'timestamp': time.time(), 'step': step, **metrics}
        if self._writer is None:
            # Called outside context manager — open lazily
            with open(self.path, 'a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=self.fields, extrasaction='ignore')
                if self.path.stat().st_size == 0:
                    w.writeheader()
                w.writerow(row)
        else:
            self._writer.writerow(row)
            self._file.flush()


class WandbLogger:
    """
    Thin wrapper around wandb that gracefully degrades if wandb is absent.
    """

    def __init__(self, project: str, name: str, config: Dict = None):
        self._run = None
        try:
            import wandb
            self._run = wandb.init(project=project, name=name, config=config or {})
            print(f'[WandbLogger] Run started: {self._run.url}')
        except ImportError:
            print('[WandbLogger] wandb not installed — logging disabled.')

    def log(self, step: int, metrics: Dict[str, Any]) -> None:
        if self._run is not None:
            import wandb
            wandb.log({'step': step, **metrics})

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.finish()


class CompositeLogger:
    """Dispatches log() calls to both a CSVLogger and an optional WandbLogger."""

    def __init__(self, csv_path: str, fields: List[str],
                  wandb_project: Optional[str] = None,
                  wandb_name:    Optional[str] = None,
                  config:        Optional[Dict] = None):
        self.csv    = CSVLogger(csv_path, fields)
        self.wandb  = (WandbLogger(wandb_project, wandb_name, config)
                       if wandb_project else None)

    def log(self, step: int, metrics: Dict[str, Any]) -> None:
        self.csv.log(step, metrics)
        if self.wandb:
            self.wandb.log(step, metrics)

    def finish(self) -> None:
        if self.wandb:
            self.wandb.finish()
