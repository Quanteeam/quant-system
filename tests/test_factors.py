"""factors.py 단위 테스트 — compute_momentum."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factors import compute_momentum


def _make_prices(tickers: list[str], n: int = 300, rising: bool = True) -> pd.DataFrame:
    """테스트용 MultiIndex prices 생성 (adj_close만 포함)."""
    dates = pd.date_range("2019-01-01", periods=n, freq="B")
    rng = np.random.default_rng(42)
    cols = pd.MultiIndex.from_tuples(
        [(t, "adj_close") for t in tickers], names=["ticker", "field"]
    )
    if rising:
        data = np.linspace(100, 150, n)[:, None] + rng.normal(0, 1, (n, len(tickers)))
    else:
        data = np.linspace(150, 100, n)[:, None] + rng.normal(0, 1, (n, len(tickers)))
    return pd.DataFrame(data, index=dates, columns=cols)


def test_returns_series():
    prices = _make_prices(["AAPL", "MSFT"])
    result = compute_momentum(prices)
    assert isinstance(result, pd.Series)


def test_tickers_in_index():
    prices = _make_prices(["AAPL", "MSFT", "GOOG"])
    result = compute_momentum(prices)
    assert set(result.index).issubset({"AAPL", "MSFT", "GOOG"})


def test_insufficient_data_returns_empty():
    prices = _make_prices(["AAPL"], n=50)
    result = compute_momentum(prices, lookback=252, skip=21)
    assert result.empty


def test_rising_stock_positive_momentum():
    prices = _make_prices(["AAPL"], n=300, rising=True)
    result = compute_momentum(prices, lookback=252, skip=21)
    assert result["AAPL"] > 0


def test_falling_stock_negative_momentum():
    prices = _make_prices(["AAPL"], n=300, rising=False)
    result = compute_momentum(prices, lookback=252, skip=21)
    assert result["AAPL"] < 0
