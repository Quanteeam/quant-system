from __future__ import annotations

from typing import Iterable

import pandas as pd

BENCHMARK_TICKERS = ["SPY", "QQQ"]
MOMENTUM_SKIP_DAYS = 21

LEGACY_UNIVERSE_TICKERS = [
    "CRWD", "DDOG", "NET", "ZS", "HUBS", "PAYC", "FTNT", "SNAP",
    "MRVL", "SWKS", "MPWR", "ON", "ENTG", "MKSI",
    "ALGN", "DXCM", "HOLX", "TECH", "NBIX", "EXAS",
    "DECK", "POOL", "WSM", "DPZ", "WING", "BURL",
    "AXON", "GNRC", "TREX", "RBC", "FND", "SITE",
    "LPLA", "RGA", "EWBC", "KNSL", "WBS", "CFR",
    "TRGP", "AR", "CLF", "ATI", "GPK", "UFPI",
    "ELS", "AMH",
]

COMMON_STOCK_CATEGORIES = {
    "Domestic Common Stock",
    "Domestic Common Stock Primary Class",
    "Domestic Common Stock Secondary Class",
}

EXCHANGE_SCOPES = {
    "NASDAQ": {"NASDAQ"},
    "US Common": {"NASDAQ", "NYSE", "NYSEMKT", "BATS"},
}


def compute_load_start(
    eval_start: str | pd.Timestamp,
    momentum_lookback: int,
    lowvol_lookback: int,
    momentum_skip: int = MOMENTUM_SKIP_DAYS,
    buffer_days: int = 40,
) -> pd.Timestamp:
    """Return the earlier load start needed for factor warmup."""
    eval_start_ts = pd.Timestamp(eval_start)
    required_days = max(momentum_lookback + momentum_skip + 2, lowvol_lookback + 2)
    return (eval_start_ts - pd.offsets.BDay(required_days + buffer_days)).normalize()


def prepare_dynamic_metadata(
    metadata: pd.DataFrame,
    load_start: str | pd.Timestamp,
    eval_end: str | pd.Timestamp,
    exchange_scope: str = "NASDAQ",
) -> pd.DataFrame:
    if metadata is None or metadata.empty:
        return pd.DataFrame()

    meta = metadata.copy()
    if "table" in meta.columns:
        meta = meta[meta["table"].eq("SEP")].copy()
    if "category" in meta.columns:
        meta = meta[meta["category"].isin(COMMON_STOCK_CATEGORIES)].copy()

    exchanges = EXCHANGE_SCOPES.get(exchange_scope)
    if exchanges and "exchange" in meta.columns:
        meta = meta[meta["exchange"].isin(exchanges)].copy()

    for col in ("firstpricedate", "lastpricedate"):
        if col in meta.columns:
            meta[col] = pd.to_datetime(meta[col], errors="coerce")

    start_ts = pd.Timestamp(load_start)
    end_ts = pd.Timestamp(eval_end)
    if {"firstpricedate", "lastpricedate"}.issubset(meta.columns):
        meta = meta[
            meta["firstpricedate"].notna()
            & meta["lastpricedate"].notna()
            & (meta["firstpricedate"] <= end_ts)
            & (meta["lastpricedate"] >= start_ts)
        ].copy()

    meta = meta.sort_values([c for c in ["ticker", "lastpricedate", "table"] if c in meta.columns])
    meta = meta.drop_duplicates("ticker", keep="last")
    return meta.set_index("ticker", drop=False).sort_index()


def build_candidate_tickers(
    dynamic_metadata: pd.DataFrame,
    dynamic_universe: bool,
    legacy_tickers: Iterable[str] = LEGACY_UNIVERSE_TICKERS,
) -> list[str]:
    if dynamic_universe and dynamic_metadata is not None and not dynamic_metadata.empty:
        return sorted(dynamic_metadata.index.unique().tolist())
    return sorted(set(legacy_tickers))


def select_dynamic_universe(
    dynamic_metadata: pd.DataFrame,
    as_of_date: pd.Timestamp,
    prices_on_date: pd.Series,
    adv_on_date: pd.Series,
    history_on_date: pd.DataFrame,
    min_price: float,
    min_adv_usd: float,
    min_history_days: int,
) -> tuple[list[str], dict[str, int]]:
    if dynamic_metadata is None or dynamic_metadata.empty:
        return [], {
            "listed": 0,
            "price_ok": 0,
            "adv_ok": 0,
            "history_ok": 0,
            "eligible": 0,
        }

    as_of_date = pd.Timestamp(as_of_date)
    active = dynamic_metadata[
        (dynamic_metadata["firstpricedate"] <= as_of_date)
        & (dynamic_metadata["lastpricedate"] >= as_of_date)
    ]
    listed = set(active.index)

    prices_on_date = prices_on_date.dropna()
    adv_on_date = adv_on_date.dropna()
    history_counts = history_on_date.notna().sum(axis=0)

    price_ok = listed & set(prices_on_date[prices_on_date >= min_price].index)
    adv_ok = price_ok & set(adv_on_date[adv_on_date >= min_adv_usd].index)
    history_ok = adv_ok & set(history_counts[history_counts >= min_history_days].index)
    eligible = sorted(history_ok)

    return eligible, {
        "listed": len(listed),
        "price_ok": len(price_ok),
        "adv_ok": len(adv_ok),
        "history_ok": len(history_ok),
        "eligible": len(eligible),
    }
