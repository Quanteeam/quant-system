"""Nasdaq Data Link Sharadar data backend.

Loads SEP prices and SF1 fundamentals from Nasdaq Data Link, using SF1
``datekey`` as the point-in-time availability cutoff.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd


API_ROOT = "https://data.nasdaq.com/api/v3/datatables"
CACHE_DIR = Path.home() / ".cache" / "quant-system" / "sharadar"
_FUND_CACHE_TTL = 86400


def _get_api_key() -> str:
    key = os.environ.get("NASDAQ_DATA_LINK_API_KEY") or os.environ.get("QUANDL_API_KEY")
    if not key:
        raise EnvironmentError(
            "NASDAQ_DATA_LINK_API_KEY (or QUANDL_API_KEY) environment variable is required"
        )
    return key


def _request_json(table: str, params: dict[str, str]) -> dict:
    query = urllib.parse.urlencode(params, doseq=True)
    url = f"{API_ROOT}/{table}.json?{query}"
    with urllib.request.urlopen(url) as resp:  # nosec B310 - fixed HTTPS host
        return json.loads(resp.read().decode("utf-8"))


def _fetch_table(table: str, params: dict[str, str], paginate: bool = True) -> pd.DataFrame:
    base_params = dict(params)
    base_params["api_key"] = _get_api_key()

    rows: list[list] = []
    columns: list[str] | None = None
    cursor_id: str | None = None

    while True:
        req_params = dict(base_params)
        if cursor_id:
            req_params["qopts.cursor_id"] = cursor_id
        payload = _request_json(table, req_params)
        datatable = payload.get("datatable", {})
        if columns is None:
            columns = [c["name"] for c in datatable.get("columns", [])]
        rows.extend(datatable.get("data", []))
        cursor_id = datatable.get("next_cursor_id")
        if not paginate or not cursor_id:
            break

    if not columns:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=columns)


def _cache_path(prefix: str, payload: dict) -> Path:
    digest = hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return CACHE_DIR / f"{prefix}_{digest}.parquet"


def _first_present(frame: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for name in candidates:
        if name in frame.columns:
            return frame[name]
    return pd.Series(index=frame.index, dtype=float)


def _to_numeric(frame: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")


def load_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Load Sharadar SEP daily prices in (ticker, field) format."""
    payload = {"tickers": sorted(tickers), "start": start, "end": end}
    path = _cache_path("sep_prices", payload)
    if path.exists():
        return pd.read_parquet(path)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    params = {
        "ticker": ",".join(sorted(tickers)),
        "date.gte": start,
        "date.lte": end,
        "qopts.columns": "ticker,date,open,high,low,close,closeunadj,closeadj,volume,lastupdated",
    }
    df = _fetch_table("SHARADAR/SEP", params)
    if df.empty:
        raise ValueError("No price data from Sharadar SEP")

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    _to_numeric(df, ["open", "high", "low", "close", "closeunadj", "closeadj", "volume"])

    frames: list[pd.DataFrame] = []
    for ticker, grp in df.groupby("ticker"):
        grp = grp.sort_values("date").set_index("date")
        adjusted = grp["closeadj"] if "closeadj" in grp.columns else grp["close"]
        raw_close = grp["closeunadj"] if "closeunadj" in grp.columns else grp["close"]
        fields = pd.DataFrame(
            {
                "open": grp["open"],
                "high": grp["high"],
                "low": grp["low"],
                "close": raw_close,
                "adj_close": adjusted,
                "volume": grp["volume"],
            },
            index=grp.index,
        )
        fields.columns = pd.MultiIndex.from_tuples(
            [(ticker, c) for c in fields.columns], names=["ticker", "field"]
        )
        frames.append(fields)

    result = pd.concat(frames, axis=1).sort_index().sort_index(axis=1)
    result.to_parquet(path)
    return result


def load_ticker_metadata(tickers: list[str] | None = None) -> pd.DataFrame:
    """Load ticker metadata for the requested tickers."""
    tickers = sorted(set(tickers or []))
    if not tickers:
        return pd.DataFrame()

    payload = {"tickers": tickers}
    path = _cache_path("tickers", payload)
    if path.exists() and (time.time() - path.stat().st_mtime) < _FUND_CACHE_TTL:
        return pd.read_parquet(path)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    params = {"ticker": ",".join(tickers)}
    df = _fetch_table("SHARADAR/TICKERS", params)
    if df.empty:
        result = pd.DataFrame(index=pd.Index([], name="ticker"))
    else:
        if "table" in df.columns:
            df = df[df["table"].isin(["SF1", "SEP"])]
        keep = [c for c in ["ticker", "sector", "industry", "name", "exchange"] if c in df.columns]
        result = df[keep].drop_duplicates(subset=["ticker"]).set_index("ticker")
    result.to_parquet(path)
    return result


def load_fundamentals_history(
    tickers: list[str],
    dimensions: tuple[str, ...] = ("ARQ", "ART", "MRT"),
) -> pd.DataFrame:
    """Load raw SF1 history for the requested tickers."""
    payload = {"tickers": sorted(tickers), "dimensions": list(dimensions)}
    path = _cache_path("sf1_history", payload)
    if path.exists() and (time.time() - path.stat().st_mtime) < _FUND_CACHE_TTL:
        return pd.read_parquet(path)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    params = {
        "ticker": ",".join(sorted(tickers)),
        "dimension": ",".join(dimensions),
    }
    df = _fetch_table("SHARADAR/SF1", params)
    if df.empty:
        result = pd.DataFrame()
    else:
        for col in ["calendardate", "datekey", "reportperiod", "lastupdated"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col]).dt.normalize()
        meta = load_ticker_metadata(tickers)
        if not meta.empty:
            df = df.merge(meta.reset_index(), how="left", on="ticker", suffixes=("", "_meta"))
            if "sector_meta" in df.columns and "sector" not in df.columns:
                df = df.rename(columns={"sector_meta": "sector"})
        result = df.sort_values(["ticker", "datekey", "calendardate"]).reset_index(drop=True)
    result.to_parquet(path)
    return result


def select_fundamentals_snapshot(
    history: pd.DataFrame,
    tickers: list[str],
    as_of_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Build a point-in-time SF1 snapshot using only rows known by as_of_date."""
    empty = pd.DataFrame(
        columns=[
            "market_cap",
            "pe_ratio",
            "pb_ratio",
            "ev_ebitda",
            "fcf_yield",
            "roe",
            "gross_margin",
            "debt_equity",
            "sector",
        ]
    )
    if history.empty:
        return empty

    as_of = pd.Timestamp(as_of_date).normalize()
    hist = history[history["ticker"].isin(tickers)].copy()
    if "datekey" in hist.columns:
        hist = hist[hist["datekey"] <= as_of]
    if hist.empty:
        return empty

    priority = {"ARQ": 0, "ART": 1, "MRT": 2}
    hist["_dimension_priority"] = hist.get("dimension", pd.Series("", index=hist.index)).map(
        lambda value: priority.get(value, 99)
    )
    sort_cols = [
        c for c in ["ticker", "datekey", "calendardate", "_dimension_priority"] if c in hist.columns
    ]
    ascending_map = {
        "ticker": True,
        "datekey": False,
        "calendardate": False,
        "_dimension_priority": True,
    }
    hist = hist.sort_values(sort_cols, ascending=[ascending_map[c] for c in sort_cols])
    latest = hist.groupby("ticker", as_index=False).head(1).set_index("ticker")

    market_cap = pd.to_numeric(_first_present(latest, ["marketcap", "market_cap"]), errors="coerce")
    fcf = pd.to_numeric(_first_present(latest, ["fcf", "freecashflow", "free_cash_flow"]), errors="coerce")

    snapshot = pd.DataFrame(index=latest.index)
    snapshot["market_cap"] = market_cap
    snapshot["pe_ratio"] = _first_present(latest, ["pe1", "pe", "peratio", "priceearnings"])
    snapshot["pb_ratio"] = _first_present(latest, ["pb", "pbratio", "pricetobook"])
    snapshot["ev_ebitda"] = _first_present(latest, ["evebitda", "ev_ebitda"])
    snapshot["fcf_yield"] = pd.Series(index=latest.index, dtype=float)
    valid_mc = market_cap > 0
    snapshot.loc[valid_mc, "fcf_yield"] = fcf[valid_mc] / market_cap[valid_mc]
    snapshot["roe"] = _first_present(latest, ["roe", "roeq"])
    snapshot["gross_margin"] = _first_present(latest, ["grossmargin", "gross_margin"])
    snapshot["debt_equity"] = _first_present(latest, ["de", "debtequity", "debt_equity"])
    snapshot["sector"] = (
        latest["sector"] if "sector" in latest.columns else pd.Series("Unknown", index=latest.index)
    )

    numeric = [
        "market_cap",
        "pe_ratio",
        "pb_ratio",
        "ev_ebitda",
        "fcf_yield",
        "roe",
        "gross_margin",
        "debt_equity",
    ]
    snapshot[numeric] = snapshot[numeric].apply(pd.to_numeric, errors="coerce")
    return snapshot


def load_quarterly_cache(tickers: list[str]) -> dict:
    """Load Sharadar SF1 history in the common quarterly-cache shape."""
    return {"sf1": load_fundamentals_history(tickers)}


def get_pit_fundamentals(
    quarterly_cache: dict,
    close_at_date: pd.Series,
    as_of_date: pd.Timestamp,
    lag_days: int = 0,
) -> pd.DataFrame:
    """Return SF1 point-in-time fundamentals for the common strategy interface."""
    del close_at_date, lag_days
    history = quarterly_cache.get("sf1", pd.DataFrame())
    tickers = sorted(history["ticker"].dropna().unique().tolist()) if "ticker" in history.columns else []
    return select_fundamentals_snapshot(history, tickers, as_of_date)


def load_fundamentals(
    tickers: list[str],
    as_of_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return a point-in-time fundamentals snapshot."""
    history = load_fundamentals_history(tickers)
    return select_fundamentals_snapshot(history, tickers, as_of_date or pd.Timestamp.utcnow())


def load_earnings(tickers: list[str]) -> pd.DataFrame:
    """Sharadar backend currently does not expose PEAD earnings estimates."""
    del tickers
    return pd.DataFrame(columns=["ticker", "date", "actual_eps", "estimate_eps"])
