"""yfinance 데이터 로딩 + 캐싱.

캐시: ~/.cache/quant-system/prices_{hash}.parquet
배치: 50종목씩, 실패 시 3회 retry
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path.home() / ".cache" / "quant-system"
BATCH_SIZE = 50
MAX_RETRIES = 3


def _cache_key(tickers: list[str], start: str, end: str) -> Path:
    payload = json.dumps({"tickers": sorted(tickers), "start": start, "end": end})
    h = hashlib.md5(payload.encode()).hexdigest()
    return CACHE_DIR / f"prices_{h}.parquet"


def _download_batch(batch: list[str], start: str, end: str) -> pd.DataFrame:
    for attempt in range(MAX_RETRIES):
        try:
            raw = yf.download(batch, start=start, end=end, auto_adjust=False, progress=False)
            if raw.empty:
                raise ValueError("Empty response from yfinance")
            return raw
        except Exception:
            if attempt == MAX_RETRIES - 1:
                raise


def _normalize_columns(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """yfinance (field, ticker) MultiIndex → (ticker, field), lowercase."""
    df = raw.copy()
    if not isinstance(df.columns, pd.MultiIndex):
        # Single ticker: flat columns like ['Open', 'Close', ...]
        ticker = tickers[0]
        df.columns = pd.MultiIndex.from_tuples(
            [(ticker, c.lower().replace(" ", "_")) for c in df.columns],
            names=["ticker", "field"],
        )
    else:
        df.columns = pd.MultiIndex.from_tuples(
            [(ticker, field.lower().replace(" ", "_")) for field, ticker in df.columns],
            names=["ticker", "field"],
        )
    return df.sort_index(axis=1)


def load_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """yfinance daily OHLCV.

    Returns:
        MultiIndex columns (ticker, field) DataFrame.
        field = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
    Cache:
        ~/.cache/quant-system/prices_{hash}.parquet
    """
    path = _cache_key(tickers, start, end)
    if path.exists():
        return pd.read_parquet(path)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    batches = [tickers[i:i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    frames = [_normalize_columns(_download_batch(b, start, end), b) for b in batches]
    df = pd.concat(frames, axis=1) if len(frames) > 1 else frames[0]
    df.to_parquet(path)
    return df
