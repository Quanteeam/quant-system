"""Unit tests for active multi-factor portfolio construction."""
from __future__ import annotations

import pandas as pd

from portfolio import build_multifactor_portfolio


def test_top_n_selects_highest_scores():
    scores = pd.Series({"A": 3.0, "B": 2.0, "C": 1.0, "D": 0.5})
    weights = build_multifactor_portfolio(scores, top_n=2)
    assert set(weights.index) == {"A", "B"}


def test_weights_sum_to_one():
    scores = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0})
    weights = build_multifactor_portfolio(scores, top_n=3)
    assert abs(weights.sum() - 1.0) < 1e-9


def test_equal_weight():
    scores = pd.Series({"A": 3.0, "B": 2.0})
    weights = build_multifactor_portfolio(scores, top_n=2)
    assert abs(weights["A"] - 0.5) < 1e-9
    assert abs(weights["B"] - 0.5) < 1e-9


def test_top_n_larger_than_universe_uses_all():
    scores = pd.Series({"A": 1.0, "B": 2.0})
    weights = build_multifactor_portfolio(scores, top_n=10)
    assert len(weights) == 2
    assert abs(weights.sum() - 1.0) < 1e-9


def test_empty_scores_returns_empty():
    weights = build_multifactor_portfolio(pd.Series(dtype=float), top_n=5)
    assert weights.empty
