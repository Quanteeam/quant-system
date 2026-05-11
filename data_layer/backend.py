"""Unified data backend selection for prices, fundamentals, PIT cache, earnings, and metadata."""
from __future__ import annotations

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
    raise ValueError(
        f"Unsupported data backend '{DEFAULT_CONFIG.data.backend}'. "
        "Use QUANT_DATA_BACKEND=local or QUANT_DATA_BACKEND=sharadar."
    )


__all__ = [
    "get_pit_fundamentals",
    "load_earnings",
    "load_fundamentals",
    "load_prices",
    "load_quarterly_cache",
    "load_ticker_metadata",
]
