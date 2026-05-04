"""yfinance 데이터 로딩 + 캐싱.

캐시: ~/.cache/quant-system/prices_{hash}.parquet
배치: 50종목씩, 실패 시 3회 retry
"""
from __future__ import annotations

import hashlib
import json
import time
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


# ---------------------------------------------------------------------------
# Fundamental data (Phase 2)
# ---------------------------------------------------------------------------
_FUND_CACHE_TTL = 86400  # 24시간


def load_fundamentals(tickers: list[str]) -> pd.DataFrame:
    """yfinance ticker.info에서 fundamental 데이터 로딩.

    Returns:
        DataFrame indexed by ticker. Columns:
        market_cap, pe_ratio, pb_ratio, ev_ebitda, fcf_yield,
        roe, gross_margin, debt_equity, sector

    ⚠ 현재 시점 스냅샷 → look-ahead bias. Phase 6에서 PIT 데이터로 교체.
    Cache: ~/.cache/quant-system/fund_{hash}.parquet (24h TTL)
    """
    h = hashlib.md5(json.dumps(sorted(tickers)).encode()).hexdigest()
    path = CACHE_DIR / f"fund_{h}.parquet"

    if path.exists() and (time.time() - path.stat().st_mtime) < _FUND_CACHE_TTL:
        return pd.read_parquet(path)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rows: dict[str, dict] = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            mc = info.get("marketCap") or 0
            rows[ticker] = {
                "market_cap": mc,
                "pe_ratio": info.get("trailingPE"),
                "pb_ratio": info.get("priceToBook"),
                "ev_ebitda": info.get("enterpriseToEbitda"),
                "fcf_yield": (info.get("freeCashflow") or 0) / mc if mc > 0 else None,
                "roe": info.get("returnOnEquity"),
                "gross_margin": info.get("grossMargins"),
                "debt_equity": info.get("debtToEquity"),
                "sector": info.get("sector", "Unknown"),
            }
        except Exception:
            continue

    df = pd.DataFrame(rows).T
    numeric_cols = [c for c in df.columns if c != "sector"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df.to_parquet(path)
    return df
