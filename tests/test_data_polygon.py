"""data_polygon.py 단위 테스트 (mocked — API 키 불필요)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

import data_polygon


@pytest.fixture(autouse=True)
def clear_cache(tmp_path, monkeypatch):
    """테스트마다 캐시 디렉토리 격리."""
    monkeypatch.setattr(data_polygon, "CACHE_DIR", tmp_path / "polygon")


@pytest.fixture
def mock_client(monkeypatch):
    """Polygon REST client mock."""
    monkeypatch.setenv("POLYGON_API_KEY", "test_key")
    client = MagicMock()
    monkeypatch.setattr(data_polygon, "_get_client", lambda: client)
    return client


def _make_agg(ts_ms, o, h, l, c, v):
    agg = MagicMock()
    agg.timestamp = ts_ms
    agg.open = o
    agg.high = h
    agg.low = l
    agg.close = c
    agg.volume = v
    return agg


def test_load_prices_returns_multiindex(mock_client):
    ts1 = int(pd.Timestamp("2023-01-03").timestamp() * 1000)
    ts2 = int(pd.Timestamp("2023-01-04").timestamp() * 1000)
    mock_client.get_aggs.return_value = [
        _make_agg(ts1, 100, 105, 99, 102, 1e6),
        _make_agg(ts2, 102, 106, 101, 104, 1.2e6),
    ]
    result = data_polygon.load_prices(["AAPL"], "2023-01-01", "2023-01-05")
    assert isinstance(result.columns, pd.MultiIndex)
    assert result.columns.names == ["ticker", "field"]
    assert ("AAPL", "close") in result.columns


def test_load_prices_cache_hit(mock_client):
    ts1 = int(pd.Timestamp("2023-01-03").timestamp() * 1000)
    mock_client.get_aggs.return_value = [_make_agg(ts1, 100, 105, 99, 102, 1e6)]
    # First call
    data_polygon.load_prices(["AAPL"], "2023-01-01", "2023-01-05")
    # Second call → should not call API again
    mock_client.get_aggs.reset_mock()
    data_polygon.load_prices(["AAPL"], "2023-01-01", "2023-01-05")
    mock_client.get_aggs.assert_not_called()


def test_load_prices_empty_raises(mock_client):
    mock_client.get_aggs.return_value = []
    with pytest.raises(ValueError, match="No price data"):
        data_polygon.load_prices(["FAKE"], "2023-01-01", "2023-01-05")


def test_load_fundamentals_returns_dataframe(mock_client):
    fin_result = MagicMock()
    fin_result.results = [MagicMock(
        market_cap=1e9, pe_ratio=15.0, pb_ratio=2.0,
        ev_to_ebitda=10.0, free_cash_flow=5e7,
        return_on_equity=0.15, gross_margin=0.40,
        debt_to_equity=0.5, sector="Technology",
    )]
    mock_client.get_ticker_financials.return_value = fin_result
    result = data_polygon.load_fundamentals(["AAPL"])
    assert "AAPL" in result.index
    assert "market_cap" in result.columns
    assert result.loc["AAPL", "sector"] == "Technology"


def test_load_earnings_returns_dataframe(mock_client):
    e1 = MagicMock(actual_eps=1.5, estimated_eps=1.2, report_date="2023-01-25")
    e2 = MagicMock(actual_eps=1.8, estimated_eps=1.6, report_date="2023-04-25")
    mock_client.get_ticker_earnings.return_value = [e1, e2]
    result = data_polygon.load_earnings(["AAPL"])
    assert len(result) == 2
    assert set(result.columns) == {"ticker", "date", "actual_eps", "estimate_eps"}


def test_no_api_key_raises(monkeypatch):
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    # polygon 패키지 미설치 시 ImportError, 설치 시 EnvironmentError
    with pytest.raises((EnvironmentError, ImportError)):
        data_polygon._get_client()
