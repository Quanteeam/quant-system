"""Unified data backend selection for prices, fundamentals, PIT cache, earnings, and metadata."""
from __future__ import annotations

import pandas as pd

from core.config import DEFAULT_CONFIG

if DEFAULT_CONFIG.data.backend == "local":
    from data_layer.local import (
        get_pit_fundamentals,
        load_earnings,
        load_fundamentals,
        load_prices,
        load_quarterly_cache,
        load_ticker_metadata,
    )
elif DEFAULT_CONFIG.data.backend == "sharadar":
    from data_layer.sharadar import (
        get_pit_fundamentals,
        load_earnings,
        load_fundamentals,
        load_prices,
        load_quarterly_cache,
        load_ticker_metadata,
    )
else:
    from data_layer.yfinance_provider import (
        get_pit_fundamentals,
        load_earnings,
        load_fundamentals,
        load_prices,
        load_quarterly_cache,
    )

    def load_ticker_metadata() -> pd.DataFrame:
        return pd.DataFrame()


__all__ = [
    "get_pit_fundamentals",
    "load_earnings",
    "load_fundamentals",
    "load_prices",
    "load_quarterly_cache",
    "load_ticker_metadata",
]
