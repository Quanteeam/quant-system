"""Unit tests for active multi-factor calculations."""
from __future__ import annotations

import numpy as np
import pandas as pd

from factors import (
    _zscore,
    compute_composite,
    compute_lowvol,
    compute_momentum,
    compute_quality,
    compute_size,
    compute_value,
)


def _make_prices(tickers: list[str], n: int = 300, rising: bool = True) -> pd.DataFrame:
    dates = pd.date_range("2019-01-01", periods=n, freq="B")
    rng = np.random.default_rng(42)
    cols = pd.MultiIndex.from_tuples(
        [(ticker, "adj_close") for ticker in tickers],
        names=["ticker", "field"],
    )
    if rising:
        data = np.linspace(100, 150, n)[:, None] + rng.normal(0, 1, (n, len(tickers)))
    else:
        data = np.linspace(150, 100, n)[:, None] + rng.normal(0, 1, (n, len(tickers)))
    return pd.DataFrame(data, index=dates, columns=cols)


def _make_fundamentals(tickers: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market_cap": [1e9, 5e9, 20e9],
            "pe_ratio": [10.0, 20.0, 30.0],
            "pb_ratio": [1.0, 2.0, 4.0],
            "ev_ebitda": [8.0, 12.0, 18.0],
            "fcf_yield": [0.08, 0.04, 0.02],
            "roe": [0.20, 0.12, 0.08],
            "gross_margin": [0.45, 0.35, 0.25],
            "debt_equity": [0.3, 1.0, 2.5],
            "sector": ["Technology", "Technology", "Healthcare"],
        },
        index=tickers[:3],
    )


def test_momentum_returns_series():
    result = compute_momentum(_make_prices(["AAPL", "MSFT"]))
    assert isinstance(result, pd.Series)


def test_momentum_rising_positive():
    result = compute_momentum(_make_prices(["AAPL"], rising=True))
    assert result["AAPL"] > 0


def test_momentum_insufficient_data():
    result = compute_momentum(_make_prices(["AAPL"], n=50), lookback=252)
    assert result.empty


def test_size_smaller_cap_higher_score():
    fund = _make_fundamentals(["A", "B", "C"])
    result = compute_size(fund)
    assert result["A"] > result["B"] > result["C"]


def test_value_cheaper_stock_higher_score():
    fund = _make_fundamentals(["A", "B", "C"])
    result = compute_value(fund)
    assert result["A"] > result["C"]


def test_value_handles_negative_pe():
    fund = _make_fundamentals(["A", "B", "C"])
    fund.loc["A", "pe_ratio"] = -5.0
    result = compute_value(fund)
    assert not result.empty


def test_quality_higher_roe_higher_score():
    fund = _make_fundamentals(["A", "B", "C"])
    result = compute_quality(fund)
    assert result["A"] > result["C"]


def test_lowvol_returns_series():
    result = compute_lowvol(_make_prices(["AAPL", "MSFT"]), lookback=60)
    assert isinstance(result, pd.Series)
    assert len(result) == 2


def test_lowvol_insufficient_data():
    result = compute_lowvol(_make_prices(["AAPL"], n=30), lookback=60)
    assert result.empty


def test_composite_returns_series():
    prices = _make_prices(["A", "B", "C"])
    fund = _make_fundamentals(["A", "B", "C"])
    result = compute_composite(prices, fund)
    assert isinstance(result, pd.Series)
    assert len(result) > 0


def test_composite_without_fundamentals():
    prices = _make_prices(["A", "B"])
    result = compute_composite(prices, None)
    assert isinstance(result, pd.Series)
    assert len(result) > 0


def test_zscore_mean_zero():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    zscore = _zscore(series)
    assert abs(zscore.mean()) < 1e-9
    assert abs(zscore.std() - 1.0) < 1e-9
