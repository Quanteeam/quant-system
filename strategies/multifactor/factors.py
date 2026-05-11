"""Multi-factor calculations."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _get_close(prices: pd.DataFrame) -> pd.DataFrame:
    """Return adjusted close when available, otherwise raw close."""
    for field in ("adj_close", "close"):
        try:
            return prices.xs(field, level="field", axis=1)
        except KeyError:
            continue
    raise KeyError("prices must contain either adj_close or close")


def _zscore(series: pd.Series) -> pd.Series:
    """Cross-sectional z-score."""
    valid = series.dropna()
    if len(valid) < 2 or valid.std() == 0:
        return pd.Series(0.0, index=valid.index)
    return (valid - valid.mean()) / valid.std()


def _sector_neutral_zscore(raw: pd.Series, sectors: pd.Series) -> pd.Series:
    """Compute z-scores within each sector."""
    common = raw.dropna().index.intersection(sectors.dropna().index)
    if common.empty:
        return _zscore(raw)

    scores = raw[common]
    groups = sectors[common]
    mean = scores.groupby(groups).transform("mean")
    std = scores.groupby(groups).transform("std")
    return ((scores - mean) / std).fillna(0.0)


def compute_momentum(
    prices: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
) -> pd.Series:
    """Compute 12-1 style price momentum."""
    close = _get_close(prices)
    if len(close) < lookback + 1:
        return pd.Series(dtype=float)

    end_price = close.iloc[-(skip + 1)]
    start_price = close.iloc[-(lookback + 1)]
    return ((end_price / start_price) - 1.0).dropna()


def compute_size(fundamentals: pd.DataFrame) -> pd.Series:
    """Compute size factor as negative log market cap."""
    market_cap = fundamentals["market_cap"].dropna()
    market_cap = market_cap[market_cap > 0]
    return -np.log(market_cap)


def compute_value(fundamentals: pd.DataFrame) -> pd.Series:
    """Compute composite value score."""
    pe = fundamentals["pe_ratio"]
    pb = fundamentals["pb_ratio"]
    ev = fundamentals["ev_ebitda"]
    fcf = fundamentals["fcf_yield"]

    components = pd.DataFrame(
        {
            "earnings_yield": _zscore(1.0 / pe.where(pe > 0)),
            "book_yield": _zscore(1.0 / pb.where(pb > 0)),
            "ebitda_yield": _zscore(1.0 / ev.where(ev > 0)),
            "fcf_yield": _zscore(fcf),
        }
    )
    return components.mean(axis=1).dropna()


def compute_quality(fundamentals: pd.DataFrame) -> pd.Series:
    """Compute composite quality score."""
    debt_equity = fundamentals["debt_equity"]
    components = pd.DataFrame(
        {
            "roe": _zscore(fundamentals["roe"]),
            "gross_margin": _zscore(fundamentals["gross_margin"]),
            "low_leverage": _zscore(-debt_equity.where(debt_equity >= 0)),
        }
    )
    return components.mean(axis=1).dropna()


def compute_lowvol(prices: pd.DataFrame, lookback: int = 60) -> pd.Series:
    """Compute low-volatility score as negative annualized volatility."""
    close = _get_close(prices)
    if len(close) < lookback + 1:
        return pd.Series(dtype=float)

    daily_ret = close.iloc[-lookback:].pct_change().dropna()
    vol = daily_ret.std() * np.sqrt(252)
    return (-vol).dropna()


def compute_composite(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame | None,
    momentum_lookback: int = 252,
    lowvol_lookback: int = 60,
    sector_neutral: bool = True,
    weights: dict[str, float] | None = None,
) -> pd.Series:
    """Compute weighted multi-factor composite score."""
    if weights is None:
        weights = {
            "size": 0.2,
            "value": 0.2,
            "momentum": 0.2,
            "quality": 0.2,
            "lowvol": 0.2,
        }

    raw: dict[str, pd.Series] = {
        "momentum": compute_momentum(prices, lookback=momentum_lookback),
        "lowvol": compute_lowvol(prices, lookback=lowvol_lookback),
    }

    has_fundamentals = fundamentals is not None and not fundamentals.empty
    if has_fundamentals:
        raw["size"] = compute_size(fundamentals)
        raw["value"] = compute_value(fundamentals)
        raw["quality"] = compute_quality(fundamentals)

    sectors = (
        fundamentals["sector"]
        if has_fundamentals and "sector" in fundamentals.columns
        else None
    )

    scored: dict[str, pd.Series] = {}
    for name, series in raw.items():
        if series.empty:
            continue
        zscore = (
            _sector_neutral_zscore(series, sectors)
            if sector_neutral and sectors is not None
            else _zscore(series)
        )
        scored[name] = zscore * weights.get(name, 0.2)

    if not scored:
        return pd.Series(dtype=float)
    return pd.DataFrame(scored).sum(axis=1).dropna()
