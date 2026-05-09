"""Local parquet-backed data loader for team-shared preprocessed Nasdaq data."""
from __future__ import annotations

from itertools import islice
from pathlib import Path

import pandas as pd


PRICE_COLUMNS = [
    "ticker",
    "date",
    "open",
    "high",
    "low",
    "close",
    "closeadj",
    "closeunadj",
    "volume",
]


def _resolve_data_dir(data_dir: str | None = None) -> Path:
    from core.config import DEFAULT_CONFIG

    base = Path(data_dir or DEFAULT_CONFIG.data.local_data_dir).expanduser()
    if not base.exists():
        raise FileNotFoundError(f"Local data directory not found: {base}")
    return base


def _empty_earnings() -> pd.DataFrame:
    return pd.DataFrame(columns=["ticker", "date", "actual_eps", "estimate_eps"])


def _chunked(values: list[str], size: int) -> list[list[str]]:
    return [values[i : i + size] for i in range(0, len(values), size)]


def load_ticker_metadata(data_dir: str | None = None) -> pd.DataFrame:
    base = _resolve_data_dir(data_dir)
    meta = pd.read_parquet(base / "tickers.parquet")
    if "table" in meta.columns:
        meta = meta[meta["table"].eq("SEP")].copy()
    for col in ("firstpricedate", "lastpricedate"):
        if col in meta.columns:
            meta[col] = pd.to_datetime(meta[col], errors="coerce")
    return meta.sort_values([c for c in ["ticker", "table"] if c in meta.columns]).reset_index(drop=True)


def load_prices(tickers: list[str], start: str, end: str, data_dir: str | None = None) -> pd.DataFrame:
    """Load local SEP parquet data in the same shape as data.load_prices()."""
    base = _resolve_data_dir(data_dir)
    tickers = sorted(set(tickers))
    if not tickers:
        raise ValueError("No tickers requested.")

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    frames: list[pd.DataFrame] = []
    for chunk in _chunked(tickers, 400):
        try:
            df = pd.read_parquet(
                base / "sep",
                filters=[("ticker", "in", chunk), ("date", ">=", start_ts), ("date", "<=", end_ts)],
                columns=PRICE_COLUMNS,
            )
        except Exception:
            rows: list[pd.DataFrame] = []
            for ticker in chunk:
                path = base / "sep" / f"ticker={ticker}" / "data.parquet"
                if not path.exists():
                    continue
                one = pd.read_parquet(path, columns=[c for c in PRICE_COLUMNS if c != "ticker"])
                if one.empty:
                    continue
                one["ticker"] = ticker
                rows.append(one)
            if not rows:
                continue
            df = pd.concat(rows, ignore_index=True)
            df["date"] = pd.to_datetime(df["date"])
            df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)].copy()

        if df.empty:
            continue
        frames.append(df)

    if not frames:
        raise ValueError("No local SEP data found for the requested tickers/date range.")

    raw = pd.concat(frames, ignore_index=True)
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.sort_values(["date", "ticker"])

    close_source = "closeunadj" if "closeunadj" in raw.columns else "close"
    adj_source = "closeadj" if "closeadj" in raw.columns else close_source
    field_map = {
        "open": "open",
        "high": "high",
        "low": "low",
        close_source: "close",
        adj_source: "adj_close",
        "volume": "volume",
    }

    wide_frames: list[pd.DataFrame] = []
    for source_col, field_name in field_map.items():
        if source_col not in raw.columns:
            continue
        wide = raw.pivot(index="date", columns="ticker", values=source_col).sort_index()
        wide.columns = pd.MultiIndex.from_product([wide.columns, [field_name]], names=["ticker", "field"])
        wide_frames.append(wide)

    out = pd.concat(wide_frames, axis=1).sort_index(axis=1)
    return out


def load_fundamentals(tickers: list[str], data_dir: str | None = None) -> pd.DataFrame:
    cache = load_quarterly_cache(tickers, data_dir=data_dir)
    return get_pit_fundamentals(cache, close_at_date=pd.Series(dtype=float), as_of_date=pd.Timestamp.today())


def load_quarterly_cache(tickers: list[str], data_dir: str | None = None) -> dict:
    """Load local SF1 history plus ticker metadata for PIT snapshots."""
    base = _resolve_data_dir(data_dir)
    tickers = sorted(set(tickers))

    sf1 = pd.read_parquet(base / "sf1.parquet")
    sf1 = sf1[sf1["ticker"].isin(tickers) & sf1["dimension"].isin(["ARQ", "MRQ"])].copy()
    sf1["datekey"] = pd.to_datetime(sf1["datekey"])
    sf1["calendardate"] = pd.to_datetime(sf1["calendardate"])
    sf1["reportperiod"] = pd.to_datetime(sf1["reportperiod"])
    sf1 = sf1.sort_values(["ticker", "datekey", "calendardate"])

    meta = load_ticker_metadata(data_dir=data_dir)
    meta = meta[meta["ticker"].isin(tickers)].copy()
    meta = meta.sort_values([c for c in ["ticker", "table"] if c in meta.columns]).drop_duplicates("ticker", keep="first")
    sector_map = meta.set_index("ticker")["sector"] if "sector" in meta.columns else pd.Series(dtype=object)

    return {"sf1": sf1, "sector_map": sector_map}


def get_pit_fundamentals(
    quarterly_cache: dict,
    close_at_date: pd.Series,
    as_of_date: pd.Timestamp,
    lag_days: int = 0,
) -> pd.DataFrame:
    """Use SF1 datekey as the point-in-time availability cutoff."""
    del close_at_date, lag_days

    sf1 = quarterly_cache.get("sf1", pd.DataFrame())
    if sf1.empty:
        return pd.DataFrame()

    eligible = sf1[sf1["datekey"] <= pd.Timestamp(as_of_date)].copy()
    if eligible.empty:
        return pd.DataFrame()

    latest = eligible.groupby("ticker", as_index=False).tail(1).set_index("ticker")
    market_cap = pd.to_numeric(latest.get("marketcap"), errors="coerce")
    fcf = pd.to_numeric(latest.get("fcf"), errors="coerce")

    out = pd.DataFrame(index=latest.index)
    out["market_cap"] = market_cap
    out["pe_ratio"] = pd.to_numeric(latest.get("pe1"), errors="coerce").combine_first(
        pd.to_numeric(latest.get("pe"), errors="coerce")
    )
    out["pb_ratio"] = pd.to_numeric(latest.get("pb"), errors="coerce")
    out["ev_ebitda"] = pd.to_numeric(latest.get("evebitda"), errors="coerce")
    out["fcf_yield"] = (fcf * 4 / market_cap).where(market_cap > 0)
    out["roe"] = pd.to_numeric(latest.get("roe"), errors="coerce")
    out["gross_margin"] = pd.to_numeric(latest.get("grossmargin"), errors="coerce")
    out["debt_equity"] = pd.to_numeric(latest.get("de"), errors="coerce")

    sector_map = quarterly_cache.get("sector_map", pd.Series(dtype=object))
    out["sector"] = sector_map.reindex(out.index).astype("object").fillna("Unknown")
    return out


def load_earnings(tickers: list[str], data_dir: str | None = None) -> pd.DataFrame:
    """Local processed bundle currently has no earnings surprise dataset."""
    del tickers, data_dir
    return _empty_earnings()
