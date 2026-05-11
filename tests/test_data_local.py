"""Local parquet backend smoke tests."""
from __future__ import annotations

import pandas as pd
import pytest

from data_layer import local


@pytest.fixture
def local_bundle(tmp_path):
    base = tmp_path / "processed"
    (base / "sep" / "ticker=AAPL").mkdir(parents=True)
    (base / "sep" / "ticker=MSFT").mkdir(parents=True)

    common_dates = pd.to_datetime(["2023-01-03", "2023-01-04"])
    for ticker, offset in [("AAPL", 0), ("MSFT", 100)]:
        prices = pd.DataFrame(
            {
                "date": common_dates,
                "open": [100.0 + offset, 101.0 + offset],
                "high": [103.0 + offset, 104.0 + offset],
                "low": [99.0 + offset, 100.0 + offset],
                "close": [102.0 + offset, 103.0 + offset],
                "closeadj": [102.5 + offset, 103.5 + offset],
                "closeunadj": [101.5 + offset, 102.5 + offset],
                "volume": [1_000_000, 1_100_000],
            }
        )
        prices.to_parquet(base / "sep" / f"ticker={ticker}" / "data.parquet")

    pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL", "MSFT"],
            "table": ["SEP", "SF1", "SEP"],
            "sector": ["Technology", "Technology", "Technology"],
            "firstpricedate": ["2000-01-01", "2000-01-01", "2000-01-01"],
            "lastpricedate": ["2024-01-01", "2024-01-01", "2024-01-01"],
        }
    ).to_parquet(base / "tickers.parquet")

    pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL", "MSFT"],
            "dimension": ["ARQ", "ARQ", "ARQ"],
            "calendardate": pd.to_datetime(["2023-03-31", "2023-06-30", "2023-03-31"]),
            "datekey": pd.to_datetime(["2023-05-01", "2023-08-01", "2023-05-01"]),
            "reportperiod": pd.to_datetime(["2023-03-31", "2023-06-30", "2023-03-31"]),
            "marketcap": [100.0, 999.0, 200.0],
            "pe": [10.0, 50.0, 20.0],
            "pb": [2.0, 9.0, 4.0],
            "evebitda": [8.0, 20.0, 12.0],
            "fcf": [5.0, 50.0, 8.0],
            "roe": [0.10, 0.80, 0.20],
            "grossmargin": [0.40, 0.95, 0.55],
            "de": [0.50, 3.00, 0.30],
        }
    ).to_parquet(base / "sf1.parquet")

    return base


def test_load_prices_from_local_bundle(local_bundle):
    result = local.load_prices(["AAPL", "MSFT"], "2023-01-01", "2023-01-05", data_dir=str(local_bundle))

    assert isinstance(result.columns, pd.MultiIndex)
    assert ("AAPL", "adj_close") in result.columns
    assert ("AAPL", "close") in result.columns
    assert result.loc[pd.Timestamp("2023-01-03"), ("AAPL", "adj_close")] == 102.5
    assert result.loc[pd.Timestamp("2023-01-03"), ("AAPL", "close")] == 101.5


def test_load_ticker_metadata_filters_sep_rows(local_bundle):
    result = local.load_ticker_metadata(data_dir=str(local_bundle))

    assert result["ticker"].tolist() == ["AAPL", "MSFT"]
    assert pd.api.types.is_datetime64_any_dtype(result["firstpricedate"])


def test_local_pit_fundamentals_blocks_future_datekey_and_handles_missing_pe1(local_bundle):
    cache = local.load_quarterly_cache(["AAPL", "MSFT"], data_dir=str(local_bundle))
    snapshot = local.get_pit_fundamentals(
        cache,
        close_at_date=pd.Series(dtype=float),
        as_of_date=pd.Timestamp("2023-07-01"),
    )

    assert set(snapshot.index) == {"AAPL", "MSFT"}
    assert snapshot.loc["AAPL", "market_cap"] == 100.0
    assert snapshot.loc["AAPL", "pe_ratio"] == 10.0
    assert snapshot.loc["AAPL", "fcf_yield"] == pytest.approx(5.0 * 4 / 100.0)
    assert snapshot.loc["AAPL", "sector"] == "Technology"
