"""portfolio.py 단위 테스트."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio import build_event_portfolio, build_multifactor_portfolio, combine_sleeves


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


def test_combine_sleeves_correct_allocation():
    mf = pd.Series({"A": 1.0, "B": 1.0})   # 각 50% → * 0.4
    ev = pd.Series({"B": 1.0, "C": 1.0})   # 각 50% → * 0.6
    combined = combine_sleeves(mf, ev, mf_alloc=0.4, event_alloc=0.6)
    assert abs(combined["A"] - 0.4) < 1e-9
    assert abs(combined["B"] - 1.0) < 1e-9  # 0.4 + 0.6
    assert abs(combined["C"] - 0.6) < 1e-9


# -- Event portfolio (Phase 4) --

def _make_event_prices(n: int = 100) -> pd.DataFrame:
    """Rising price data for event portfolio tests."""
    dates = pd.date_range("2020-01-02", periods=n, freq="B")
    data = {("AAPL", "adj_close"): np.linspace(100, 120, n)}
    df = pd.DataFrame(data, index=dates)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["ticker", "field"])
    return df


def test_event_portfolio_takes_position():
    prices = _make_event_prices()
    signals = pd.DataFrame({
        "ticker": ["AAPL"],
        "date": [pd.Timestamp("2020-02-03")],
        "sue": [2.5],
    })
    w = build_event_portfolio(signals, prices, sue_threshold=1.5)
    assert w["AAPL"].sum() > 0  # 포지션 진입됨


def test_event_portfolio_below_threshold_no_position():
    prices = _make_event_prices()
    signals = pd.DataFrame({
        "ticker": ["AAPL"],
        "date": [pd.Timestamp("2020-02-03")],
        "sue": [0.5],  # threshold 미만
    })
    w = build_event_portfolio(signals, prices, sue_threshold=1.5)
    assert w["AAPL"].sum() == 0


def test_event_portfolio_stop_loss():
    """가격 하락 시 stop loss로 조기 exit."""
    dates = pd.date_range("2020-01-02", periods=100, freq="B")
    # 급락: 100 → 80 (−20%)
    data = {("AAPL", "adj_close"): np.concatenate([
        [100.0] * 25, np.linspace(100, 80, 75),
    ])}
    prices = pd.DataFrame(data, index=dates)
    prices.columns = pd.MultiIndex.from_tuples(prices.columns, names=["ticker", "field"])

    signals = pd.DataFrame({
        "ticker": ["AAPL"],
        "date": [pd.Timestamp("2020-01-02")],
        "sue": [2.0],
    })
    w = build_event_portfolio(signals, prices, sue_threshold=1.5,
                              max_holding_days=90, stop_loss=-0.10)
    # stop loss가 먼저 작동하므로 45일 미만 보유
    holding_days = (w["AAPL"] > 0).sum()
    assert holding_days < 90


def test_event_portfolio_max_holding():
    prices = _make_event_prices(n=200)
    signals = pd.DataFrame({
        "ticker": ["AAPL"],
        "date": [pd.Timestamp("2020-01-02")],
        "sue": [2.0],
    })
    w = build_event_portfolio(signals, prices, sue_threshold=1.5,
                              max_holding_days=30)
    holding_days = (w["AAPL"] > 0).sum()
    assert holding_days <= 31  # entry day + 30
