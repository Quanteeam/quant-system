"""Polygon.io ?곗씠??濡쒕뵫 (Phase 6).

Legacy free-data providers 대비 장점:
- Point-in-time (PIT) fundamental ??look-ahead bias ?쒓굅
- Survivorship-bias-free universe
- ??湲?earnings ?대젰

?섍꼍 蹂??
    POLYGON_API_KEY ??Polygon Stocks Starter ?댁긽 ?꾩슂 ($170+/??

?숈씪 ?명꽣?섏씠?? load_prices(), load_fundamentals(), load_earnings()
??data.py ?????紐⑤뱢??import?섎㈃ ??
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

CACHE_DIR = Path.home() / ".cache" / "quant-system" / "polygon"
_FUND_CACHE_TTL = 86400


def _get_client():
    """Polygon REST client (lazy import)."""
    try:
        from polygon import RESTClient
    except ImportError:
        raise ImportError("pip install polygon-api-client")
    key = os.environ.get("POLYGON_API_KEY")
    if not key:
        raise EnvironmentError("POLYGON_API_KEY ?섍꼍 蹂???꾩슂")
    return RESTClient(key)


def load_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Polygon daily OHLCV ??MultiIndex (ticker, field).

    Returns same format as data.load_prices() for drop-in replacement.
    """
    payload = json.dumps({"tickers": sorted(tickers), "start": start, "end": end})
    h = hashlib.md5(payload.encode()).hexdigest()
    path = CACHE_DIR / f"prices_{h}.parquet"

    if path.exists():
        return pd.read_parquet(path)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    client = _get_client()
    frames: list[pd.DataFrame] = []

    for ticker in tickers:
        aggs = client.get_aggs(ticker, 1, "day", start, end, limit=50000)
        if not aggs:
            continue
        rows = [{
            "date": pd.Timestamp(a.timestamp, unit="ms"),
            "open": a.open, "high": a.high, "low": a.low,
            "close": a.close, "adj_close": a.close,  # Polygon already adjusted
            "volume": a.volume,
        } for a in aggs]
        df = pd.DataFrame(rows).set_index("date")
        df.columns = pd.MultiIndex.from_tuples(
            [(ticker, f) for f in df.columns], names=["ticker", "field"]
        )
        frames.append(df)

    if not frames:
        raise ValueError("No price data from Polygon")
    result = pd.concat(frames, axis=1).sort_index()
    result.to_parquet(path)
    return result


def load_fundamentals(tickers: list[str]) -> pd.DataFrame:
    """Sharadar SF1 PIT fundamentals.

    Point-in-time: filedate 湲곗??쇰줈 ?대떦 ?쒖젏??怨듦컻???곗씠?곕쭔 ?ъ슜.
    ??look-ahead bias ?쒓굅.
    """
    h = hashlib.md5(json.dumps(sorted(tickers)).encode()).hexdigest()
    path = CACHE_DIR / f"fund_pit_{h}.parquet"

    if path.exists() and (time.time() - path.stat().st_mtime) < _FUND_CACHE_TTL:
        return pd.read_parquet(path)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    client = _get_client()
    rows: dict[str, dict] = {}

    for ticker in tickers:
        try:
            # Sharadar fundamentals (ARQ = quarterly, MRT = most recent trailing)
            fins = client.get_ticker_financials(ticker, limit=1)
            if not fins or not fins.results:
                continue
            f = fins.results[0]
            mc = getattr(f, "market_cap", 0) or 0
            rows[ticker] = {
                "market_cap": mc,
                "pe_ratio": getattr(f, "pe_ratio", None),
                "pb_ratio": getattr(f, "pb_ratio", None),
                "ev_ebitda": getattr(f, "ev_to_ebitda", None),
                "fcf_yield": (getattr(f, "free_cash_flow", 0) or 0) / mc if mc > 0 else None,
                "roe": getattr(f, "return_on_equity", None),
                "gross_margin": getattr(f, "gross_margin", None),
                "debt_equity": getattr(f, "debt_to_equity", None),
                "sector": getattr(f, "sector", "Unknown"),
            }
        except Exception:
            continue

    df = pd.DataFrame(rows).T
    numeric_cols = [c for c in df.columns if c != "sector"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df.to_parquet(path)
    return df


def load_earnings(tickers: list[str]) -> pd.DataFrame:
    """Polygon/Sharadar earnings ??actual vs estimate EPS (PIT).

    Sharadar SEP ?뚯씠釉??ъ슜: 怨쇨굅 ?꾩껜 遺꾧린 ?대젰 ?쒓났.
    """
    h = hashlib.md5(json.dumps(sorted(tickers)).encode()).hexdigest()
    path = CACHE_DIR / f"earnings_pit_{h}.parquet"

    if path.exists() and (time.time() - path.stat().st_mtime) < _FUND_CACHE_TTL:
        return pd.read_parquet(path)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    client = _get_client()
    rows: list[dict] = []

    for ticker in tickers:
        try:
            # Polygon earnings endpoint
            earnings = client.get_ticker_earnings(ticker, limit=50)
            if not earnings:
                continue
            for e in earnings:
                actual = getattr(e, "actual_eps", None)
                estimate = getattr(e, "estimated_eps", None)
                date = getattr(e, "report_date", None)
                if actual is None or estimate is None or date is None:
                    continue
                rows.append({
                    "ticker": ticker,
                    "date": pd.Timestamp(date).normalize(),
                    "actual_eps": float(actual),
                    "estimate_eps": float(estimate),
                })
        except Exception:
            continue

    df = (pd.DataFrame(rows) if rows
          else pd.DataFrame(columns=["ticker", "date", "actual_eps", "estimate_eps"]))
    df.to_parquet(path)
    return df
