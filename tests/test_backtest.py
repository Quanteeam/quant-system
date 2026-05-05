"""backtest.py 단위 테스트."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest import BacktestEngine, BacktestResult, walk_forward_split


def _make_prices(n: int = 504) -> pd.DataFrame:
    """2년치 deterministic prices (A, B, SPY)."""
    dates = pd.date_range("2020-01-02", periods=n, freq="B")
    rng = np.random.default_rng(42)
    data = {}
    for ticker in ["A", "B", "SPY"]:
        p = [100.0]
        for _ in range(n - 1):
            p.append(p[-1] * (1 + 0.0003 + rng.normal(0, 0.01)))
        data[(ticker, "adj_close")] = p
    df = pd.DataFrame(data, index=dates)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["ticker", "field"])
    return df


def test_run_returns_backtest_result():
    prices = _make_prices()
    w = pd.DataFrame({"A": 0.5, "B": 0.5}, index=prices.index)
    result = BacktestEngine(prices, commission_bps=0, slippage_bps=0).run(w)
    assert isinstance(result, BacktestResult)
    assert len(result.equity_curve) == len(prices)


def test_commission_reduces_return():
    prices = _make_prices()
    # 턴오버가 발생하도록 매월 weight 전환
    w = pd.DataFrame({"A": 0.5, "B": 0.5}, index=prices.index)
    monthly = w.index.to_series().dt.month
    w.loc[monthly % 2 == 0, "A"] = 0.8
    w.loc[monthly % 2 == 0, "B"] = 0.2
    r_free = BacktestEngine(prices, commission_bps=0, slippage_bps=0).run(w)
    r_cost = BacktestEngine(prices, commission_bps=5, slippage_bps=50).run(w)
    assert r_free.total_return > r_cost.total_return


def test_calmar_positive():
    prices = _make_prices()
    w = pd.DataFrame({"A": 0.5, "B": 0.5}, index=prices.index)
    result = BacktestEngine(prices, commission_bps=0, slippage_bps=0).run(w)
    if result.cagr > 0:
        assert result.calmar > 0


def test_monthly_returns_not_empty():
    prices = _make_prices()
    w = pd.DataFrame({"A": 0.5, "B": 0.5}, index=prices.index)
    result = BacktestEngine(prices).run(w)
    assert len(result.monthly_returns) > 0


def test_walk_forward_split():
    dates = pd.date_range("2014-01-02", periods=252 * 8, freq="B")
    rng = np.random.default_rng(42)
    equity = pd.Series(
        (1 + 0.0003 + rng.normal(0, 0.01, len(dates))).cumprod() * 100_000,
        index=dates,
    )
    wf = walk_forward_split(equity, train_years=5, test_years=1)
    assert len(wf) > 0
    assert "train_sharpe" in wf[0]
    assert "test_sharpe" in wf[0]
