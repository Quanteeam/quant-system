"""data_sharadar.py unit tests."""
from __future__ import annotations

import pandas as pd
import pytest

import data_sharadar


@pytest.fixture(autouse=True)
def redirect_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(data_sharadar, "CACHE_DIR", tmp_path / "sharadar")


def test_load_prices_maps_sep_columns(monkeypatch):
    sample = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL"],
            "date": ["2023-01-03", "2023-01-04"],
            "open": [100.0, 101.0],
            "high": [103.0, 104.0],
            "low": [99.0, 100.0],
            "close": [102.0, 103.0],
            "closeunadj": [101.0, 102.0],
            "volume": [1_000_000, 1_100_000],
        }
    )
    monkeypatch.setattr(data_sharadar, "_fetch_table", lambda *args, **kwargs: sample)

    result = data_sharadar.load_prices(["AAPL"], "2023-01-01", "2023-01-05")

    assert isinstance(result.columns, pd.MultiIndex)
    assert ("AAPL", "adj_close") in result.columns
    assert ("AAPL", "close") in result.columns
    assert result.loc[pd.Timestamp("2023-01-03"), ("AAPL", "adj_close")] == 102.0
    assert result.loc[pd.Timestamp("2023-01-03"), ("AAPL", "close")] == 101.0


def test_select_fundamentals_snapshot_blocks_future_datekey():
    history = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL"],
            "dimension": ["ARQ", "ARQ"],
            "calendardate": pd.to_datetime(["2023-03-31", "2023-06-30"]),
            "datekey": pd.to_datetime(["2023-05-01", "2023-08-01"]),
            "marketcap": [100.0, 999.0],
            "pe1": [10.0, 50.0],
            "pb": [2.0, 9.0],
            "evebitda": [8.0, 20.0],
            "fcf": [5.0, 50.0],
            "roe": [0.10, 0.80],
            "grossmargin": [0.40, 0.95],
            "de": [0.50, 3.00],
            "sector": ["Technology", "Technology"],
        }
    )

    snapshot = data_sharadar.select_fundamentals_snapshot(
        history, ["AAPL"], pd.Timestamp("2023-07-01")
    )

    assert snapshot.loc["AAPL", "market_cap"] == 100.0
    assert snapshot.loc["AAPL", "pe_ratio"] == 10.0
    assert snapshot.loc["AAPL", "roe"] == 0.10


def test_select_fundamentals_snapshot_prefers_latest_known_row():
    history = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL", "MSFT"],
            "dimension": ["ARQ", "MRT", "ARQ"],
            "calendardate": pd.to_datetime(["2023-03-31", "2023-03-31", "2023-03-31"]),
            "datekey": pd.to_datetime(["2023-05-01", "2023-05-02", "2023-05-01"]),
            "marketcap": [100.0, 110.0, 200.0],
            "pe1": [10.0, 11.0, 20.0],
            "pb": [2.0, 2.2, 4.0],
            "evebitda": [8.0, 8.5, 12.0],
            "fcf": [5.0, 5.5, 8.0],
            "roe": [0.10, 0.11, 0.20],
            "grossmargin": [0.40, 0.42, 0.55],
            "de": [0.50, 0.45, 0.30],
            "sector": ["Technology", "Technology", "Technology"],
        }
    )

    snapshot = data_sharadar.select_fundamentals_snapshot(
        history, ["AAPL", "MSFT"], pd.Timestamp("2023-05-10")
    )

    assert set(snapshot.index) == {"AAPL", "MSFT"}
    assert snapshot.loc["AAPL", "market_cap"] == 110.0
    assert snapshot.loc["AAPL", "fcf_yield"] == pytest.approx(5.5 / 110.0)
