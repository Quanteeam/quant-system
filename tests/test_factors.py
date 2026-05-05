"""factors.py 단위 테스트 — 5팩터 + composite."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factors import (
    compute_composite,
    compute_lowvol,
    compute_momentum,
    compute_quality,
    compute_size,
    compute_sue,
    compute_value,
    _zscore,
)


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


def _make_fundamentals(tickers: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "market_cap": [1e9, 5e9, 20e9],
        "pe_ratio": [10.0, 20.0, 30.0],
        "pb_ratio": [1.0, 2.0, 4.0],
        "ev_ebitda": [8.0, 12.0, 18.0],
        "fcf_yield": [0.08, 0.04, 0.02],
        "roe": [0.20, 0.12, 0.08],
        "gross_margin": [0.45, 0.35, 0.25],
        "debt_equity": [0.3, 1.0, 2.5],
        "sector": ["Technology", "Technology", "Healthcare"],
    }, index=tickers[:3])


# -- Momentum (기존) --

def test_momentum_returns_series():
    result = compute_momentum(_make_prices(["AAPL", "MSFT"]))
    assert isinstance(result, pd.Series)

def test_momentum_rising_positive():
    result = compute_momentum(_make_prices(["AAPL"], rising=True))
    assert result["AAPL"] > 0

def test_momentum_insufficient_data():
    result = compute_momentum(_make_prices(["AAPL"], n=50), lookback=252)
    assert result.empty


# -- Size --

def test_size_smaller_cap_higher_score():
    fund = _make_fundamentals(["A", "B", "C"])
    result = compute_size(fund)
    assert result["A"] > result["B"] > result["C"]  # A가 시총 가장 작음


# -- Value --

def test_value_cheaper_stock_higher_score():
    fund = _make_fundamentals(["A", "B", "C"])
    result = compute_value(fund)
    # A: PE 10, PB 1, EV/EBITDA 8, FCF 8% → 가장 저렴
    assert result["A"] > result["C"]

def test_value_handles_negative_pe():
    fund = _make_fundamentals(["A", "B", "C"])
    fund.loc["A", "pe_ratio"] = -5.0  # 적자
    result = compute_value(fund)
    assert not result.empty


# -- Quality --

def test_quality_higher_roe_higher_score():
    fund = _make_fundamentals(["A", "B", "C"])
    result = compute_quality(fund)
    assert result["A"] > result["C"]  # A: ROE 20%, 낮은 레버리지


# -- Low Vol --

def test_lowvol_returns_series():
    result = compute_lowvol(_make_prices(["AAPL", "MSFT"]), lookback=60)
    assert isinstance(result, pd.Series)
    assert len(result) == 2

def test_lowvol_insufficient_data():
    result = compute_lowvol(_make_prices(["AAPL"], n=30), lookback=60)
    assert result.empty


# -- Composite --

def test_composite_returns_series():
    prices = _make_prices(["A", "B", "C"])
    fund = _make_fundamentals(["A", "B", "C"])
    result = compute_composite(prices, fund)
    assert isinstance(result, pd.Series)
    assert len(result) > 0

def test_composite_without_fundamentals():
    """fundamentals=None이면 momentum + lowvol만 사용."""
    prices = _make_prices(["A", "B"])
    result = compute_composite(prices, None)
    assert isinstance(result, pd.Series)
    assert len(result) > 0


# -- SUE (Phase 4) --

def test_sue_positive_surprise():
    earnings = pd.DataFrame({
        "ticker": ["AAPL"] * 4,
        "date": pd.date_range("2020-03-31", periods=4, freq="QE"),
        "actual_eps": [1.0, 1.2, 1.5, 2.0],
        "estimate_eps": [0.9, 1.0, 1.3, 1.0],
    })
    result = compute_sue(earnings)
    assert not result.empty
    last = result.iloc[-1]
    assert last["sue"] > 0  # 마지막 이벤트: 큰 양의 서프라이즈

def test_sue_needs_at_least_two_events():
    earnings = pd.DataFrame({
        "ticker": ["AAPL"],
        "date": [pd.Timestamp("2020-03-31")],
        "actual_eps": [1.5],
        "estimate_eps": [1.0],
    })
    result = compute_sue(earnings)
    assert result.empty  # std 계산 불가

def test_sue_empty_input():
    result = compute_sue(pd.DataFrame(
        columns=["ticker", "date", "actual_eps", "estimate_eps"]))
    assert result.empty

def test_sue_no_lookahead():
    """i번째 SUE 계산에 미래 surprise 미포함 확인."""
    earnings = pd.DataFrame({
        "ticker": ["X"] * 4,
        "date": pd.date_range("2020-01-01", periods=4, freq="QE"),
        "actual_eps": [1.0, 1.2, 0.8, 100.0],  # 4번째에 극단적 서프라이즈
        "estimate_eps": [0.9, 0.9, 0.9, 0.9],
    })
    result = compute_sue(earnings)
    # i=1 시점: surprise = [0.1, 0.3], std > 0 → SUE 계산 가능
    sue_at_1 = result[result["date"] == earnings.iloc[1]["date"]]
    assert not sue_at_1.empty
    # i=1 SUE는 4번째 극단값(99.1)과 무관해야 함
    assert abs(sue_at_1.iloc[0]["sue"]) < 10  # 극단적이지 않음


# -- zscore --

def test_zscore_mean_zero():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    z = _zscore(s)
    assert abs(z.mean()) < 1e-9
    assert abs(z.std() - 1.0) < 1e-9
