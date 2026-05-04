"""data.py 단위 테스트 — yfinance 호출 없이 mock 사용."""
from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

import data as data_module
from data import _cache_key, load_prices


def _mock_raw(tickers: list[str], n: int = 10) -> pd.DataFrame:
    """yfinance download가 반환하는 형식 (field, ticker) MultiIndex 생성."""
    dates = pd.date_range("2020-01-02", periods=n, freq="B")
    fields = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    cols = pd.MultiIndex.from_product([fields, tickers])
    return pd.DataFrame(100.0, index=dates, columns=cols)


@pytest.fixture(autouse=True)
def redirect_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(data_module, "CACHE_DIR", tmp_path)


def test_load_prices_returns_multiindex():
    tickers = ["AAPL", "MSFT"]
    with patch("yfinance.download", return_value=_mock_raw(tickers)):
        df = load_prices(tickers, "2020-01-01", "2020-01-15")
    assert isinstance(df.columns, pd.MultiIndex)
    assert df.columns.names == ["ticker", "field"]


def test_load_prices_field_names_lowercase():
    tickers = ["AAPL"]
    with patch("yfinance.download", return_value=_mock_raw(tickers)):
        df = load_prices(tickers, "2020-01-01", "2020-01-15")
    fields = df["AAPL"].columns.tolist()
    assert "close" in fields
    assert "adj_close" in fields


def test_cache_hit_prevents_second_download():
    tickers = ["AAPL"]
    with patch("yfinance.download", return_value=_mock_raw(tickers)) as mock_dl:
        load_prices(tickers, "2020-01-01", "2020-01-15")
        load_prices(tickers, "2020-01-01", "2020-01-15")
    assert mock_dl.call_count == 1


def test_cache_key_order_independent():
    p1 = _cache_key(["AAPL", "MSFT"], "2020-01-01", "2020-12-31")
    p2 = _cache_key(["MSFT", "AAPL"], "2020-01-01", "2020-12-31")
    assert p1 == p2


def test_different_params_different_cache_key():
    p1 = _cache_key(["AAPL"], "2020-01-01", "2020-12-31")
    p2 = _cache_key(["AAPL"], "2021-01-01", "2021-12-31")
    assert p1 != p2
