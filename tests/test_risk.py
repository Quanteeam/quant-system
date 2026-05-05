"""risk.py 단위 테스트."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risk import RiskEngine, RiskEvent


@pytest.fixture
def engine():
    return RiskEngine()


def test_position_limit_violation(engine):
    weights = pd.Series({"A": 0.05, "B": 0.02})  # A > 3%
    violations = engine.check_position_limits(weights)
    assert len(violations) == 1
    assert "A" in violations[0]


def test_position_limit_pass(engine):
    weights = pd.Series({"A": 0.02, "B": 0.01})
    assert engine.check_position_limits(weights) == []


def test_sector_limit_violation(engine):
    weights = pd.Series({"A": 0.20, "B": 0.15, "C": 0.10})
    sectors = pd.Series({"A": "Tech", "B": "Tech", "C": "Health"})
    violations = engine.check_sector_limits(weights, sectors)
    assert len(violations) == 1  # Tech: 35% > 30%


def test_sector_limit_pass(engine):
    weights = pd.Series({"A": 0.10, "B": 0.10, "C": 0.10})
    sectors = pd.Series({"A": "Tech", "B": "Health", "C": "Energy"})
    assert engine.check_sector_limits(weights, sectors) == []


def test_clean_weights_nan_inf(engine):
    w = pd.Series({"A": np.nan, "B": np.inf, "C": 0.5})
    cleaned = engine.clean_weights(w)
    assert cleaned["A"] == 0.0
    assert cleaned["B"] == 0.0
    assert cleaned["C"] == 0.5


def test_flag_extreme_returns(engine):
    ret = pd.DataFrame({"A": [0.01, 0.35, -0.02], "B": [-0.40, 0.01, 0.05]})
    flags = engine.flag_extreme_returns(ret)
    assert flags.iloc[1]["A"] is np.True_  # 35%
    assert flags.iloc[0]["B"] is np.True_  # -40%
    assert flags.iloc[0]["A"] is np.False_  # 1%


def _make_crashing_prices(n=252):
    """100 → 70 (−30%) crash for risk rule testing."""
    dates = pd.date_range("2020-01-02", periods=n, freq="B")
    # Flat then crash
    flat = [100.0] * (n // 2)
    crash = np.linspace(100, 65, n - n // 2).tolist()
    data = {("A", "adj_close"): flat + crash, ("SPY", "adj_close"): [100.0] * n}
    df = pd.DataFrame(data, index=dates)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["ticker", "field"])
    return df


def test_apply_risk_drawdown_halt():
    """DD -20% → system halt → weights zeroed."""
    prices = _make_crashing_prices()
    w = pd.DataFrame({"A": 1.0}, index=prices.index)  # 100% in crashing stock
    engine = RiskEngine()
    modified, events = engine.apply_risk_to_backtest(w, prices, cost_rate=0)
    # Should have risk events
    event_types = [e.event_type for e in events]
    assert "system_halt" in event_types or "reduce_50pct" in event_types
    # After halt, weights should be zero
    last_nonzero = (modified["A"] != 0).sum()
    assert last_nonzero < len(prices)


def test_apply_risk_no_events_when_flat():
    """Flat prices → no risk events."""
    dates = pd.date_range("2020-01-02", periods=100, freq="B")
    data = {("A", "adj_close"): [100.0] * 100}
    prices = pd.DataFrame(data, index=dates)
    prices.columns = pd.MultiIndex.from_tuples(prices.columns, names=["ticker", "field"])
    w = pd.DataFrame({"A": 0.5}, index=dates)
    engine = RiskEngine()
    _, events = engine.apply_risk_to_backtest(w, prices, cost_rate=0)
    assert len(events) == 0
