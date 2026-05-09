"""Shared strategy interfaces."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd


@dataclass(frozen=True)
class StrategyDefinition:
    key: str
    label: str
    description: str


@dataclass(frozen=True)
class StrategyDataAccess:
    get_prices: Callable[[tuple[str, ...], str, str], pd.DataFrame]
    get_earnings: Callable[[tuple[str, ...]], pd.DataFrame]
    get_quarterly: Callable[[tuple[str, ...]], dict]
    get_ticker_metadata: Callable[[], pd.DataFrame]
    ensure_benchmark_prices: Callable[[pd.DataFrame, str, str, list[str]], tuple[pd.DataFrame, list[str]]]
