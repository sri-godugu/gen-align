"""Unit tests for evaluation metrics and safety scorer."""
import pytest
import numpy as np

from src.evaluation.metrics import (
    compute_win_rate,
    compute_mean_reward,
    distinct_n,
    alignment_delta,
    compute_all_metrics,
)
from src.evaluation.safety import heuristic_toxicity, SafetyScorer


# ── Win rate ──────────────────────────────────────────────────────────────────

class TestWinRate:
    def test_all_wins(self):
        p = np.array([2.0, 3.0, 4.0])
        r = np.array([1.0, 2.0, 3.0])
        assert compute_win_rate(p, r) == 1.0

    def test_all_losses(self):
        p = np.array([0.0, 0.0])
        r = np.array([1.0, 1.0])
        assert compute_win_rate(p, r) == 0.0

    def test_half_wins(self):
        p = np.array([1.0, 0.0])
        r = np.array([0.0, 1.0])
        assert compute_win_rate(p, r) == 0.5


# ── Mean reward ───────────────────────────────────────────────────────────────

class TestMeanReward:
    def test_stats(self):
        r = np.array([1.0, 2.0, 3.0, 4.0])
        s = compute_mean_reward(r)
        assert s['mean'] == pytest.approx(2.5)
        assert s['min']  == 1.0
        assert s['max']  == 4.0


# ── Distinct-n ────────────────────────────────────────────────────────────────

class TestDistinctN:
    def test_identical_texts_low_diversity(self):
        texts = ['hello world'] * 10
        assert distinct_n(texts, 2) < 0.2

    def test_diverse_texts_high_diversity(self):
        texts = [f'word{i} word{i+1}' for i in range(20)]
        assert distinct_n(texts, 2) > 0.5

    def test_empty_returns_zero(self):
        assert distinct_n([], 2) == 0.0


# ── Alignment delta ───────────────────────────────────────────────────────────

class TestAlignmentDelta:
    def test_improvement(self):
        before = np.array([1.0, 1.0, 1.0])
        after  = np.array([2.0, 2.0, 2.0])
        d      = alignment_delta(before, after)
        assert d['mean_delta']   == pytest.approx(1.0)
        assert d['pct_improved'] == pytest.approx(1.0)
        assert d['win_rate']     == pytest.approx(1.0)


# ── Aggregate metrics ─────────────────────────────────────────────────────────

class TestComputeAllMetrics:
    def test_keys_present(self):
        p = np.random.randn(20) + 1.0
        r = np.random.randn(20)
        report = compute_all_metrics(p, r)
        for key in ('win_rate', 'reward', 'ref_reward', 'delta'):
            assert key in report

    def test_with_generations(self):
        p    = np.ones(5)
        r    = np.zeros(5)
        gens = ['the quick brown fox'] * 5
        report = compute_all_metrics(p, r, generations=gens)
        assert 'distinct_2' in report


# ── Safety scorer ─────────────────────────────────────────────────────────────

class TestSafetyScorer:
    def test_heuristic_clean_text(self):
        assert heuristic_toxicity('The weather is nice today.') == 0.0

    def test_heuristic_toxic_text(self):
        score = heuristic_toxicity('I want to kill you with a weapon.')
        assert score > 0.0

    def test_scorer_safe(self):
        scorer = SafetyScorer()
        assert scorer.is_safe('Hello, how are you?')

    def test_filter_unsafe(self):
        scorer = SafetyScorer()
        texts  = ['Hello!', 'I will hurt you with explosives.']
        safe   = scorer.filter_unsafe(texts, threshold=0.1)
        assert 'Hello!' in safe
