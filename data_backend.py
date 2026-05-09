"""Unified data backend selection for prices, fundamentals, PIT cache, earnings, and metadata."""
from __future__ import annotations

import pandas as pd

from config import DEFAULT_CONFIG

if DEFAULT_CONFIG.data.backend == "local":
    from data_local import (
        get_pit_fundamentals,
        load_earnings,
        load_fundamentals,
        load_prices,
        load_quarterly_cache,
        load_ticker_metadata,
    )
else:
    from data import (
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
