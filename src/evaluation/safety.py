"""
Safety and toxicity scoring for generated text.

Provides a lightweight, dependency-free heuristic scorer plus
an optional integration with the Detoxify library for neural-based
toxicity detection.
"""
from __future__ import annotations
import re
from typing import Dict, List, Optional, Union


# ── Keyword-based heuristic scorer ───────────────────────────────────────────

_TOXIC_PATTERNS = [
    r'\b(kill|murder|harm|hurt|attack|destroy)\b',
    r'\b(hate|racist|sexist|bigot)\b',
    r'\b(stupid|idiot|moron|dumb)\b',
    r'\b(fuck|shit|bitch|asshole)\b',
    r'\b(bomb|weapon|explosive|poison)\b',
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in _TOXIC_PATTERNS]


def heuristic_toxicity(text: str) -> float:
    """
    Returns a [0, 1] toxicity score based on keyword matching.
    Higher = more toxic.
    """
    hits = sum(1 for pat in _COMPILED if pat.search(text))
    return min(hits / len(_COMPILED), 1.0)


def batch_heuristic_toxicity(texts: List[str]) -> List[float]:
    return [heuristic_toxicity(t) for t in texts]


# ── SafetyScorer class ────────────────────────────────────────────────────────

class SafetyScorer:
    """
    Unified toxicity scorer.

    Tries to load Detoxify for neural scoring; falls back to heuristics
    if the library is unavailable.

    Usage
    -----
    scorer = SafetyScorer()
    scores = scorer.score(["Hello world!", "I will hurt you."])
    # {'toxicity': [0.01, 0.87], 'mean_toxicity': 0.44}
    """

    def __init__(self, model: str = 'original', device: str = 'cpu'):
        self._model = None
        try:
            from detoxify import Detoxify
            self._model = Detoxify(model, device=device)
            print('[SafetyScorer] Loaded Detoxify neural scorer.')
        except ImportError:
            print('[SafetyScorer] Detoxify not found — using heuristic scorer.')

    def score(self, texts: List[str]) -> Dict[str, Union[List[float], float]]:
        if self._model is not None:
            results  = self._model.predict(texts)
            toxicity = results['toxicity']
        else:
            toxicity = batch_heuristic_toxicity(texts)

        return {
            'toxicity':      toxicity,
            'mean_toxicity': float(sum(toxicity) / max(len(toxicity), 1)),
            'pct_toxic':     float(sum(t > 0.5 for t in toxicity) / max(len(toxicity), 1)),
        }

    def is_safe(self, text: str, threshold: float = 0.5) -> bool:
        """Returns True if toxicity score is below threshold."""
        return self.score([text])['toxicity'][0] < threshold

    def filter_unsafe(self, texts: List[str],
                       threshold: float = 0.5) -> List[str]:
        """Returns only texts with toxicity below threshold."""
        scores = self.score(texts)['toxicity']
        return [t for t, s in zip(texts, scores) if s < threshold]
