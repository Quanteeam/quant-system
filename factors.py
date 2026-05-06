"""팩터 계산: 5팩터 + SUE + gap events. Sector neutral z-score 지원."""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_close(prices: pd.DataFrame) -> pd.DataFrame:
    """prices MultiIndex에서 close 추출 (adj_close 우선)."""
    for field in ("adj_close", "close"):
        try:
            return prices.xs(field, level="field", axis=1)
        except KeyError:
            continue
    raise KeyError("prices에 close 또는 adj_close 컬럼이 없습니다.")


def _zscore(s: pd.Series) -> pd.Series:
    """Cross-sectional z-score. NaN은 제외, 표준편차 0이면 0 반환."""
    valid = s.dropna()
    if len(valid) < 2 or valid.std() == 0:
        return pd.Series(0.0, index=valid.index)
    return (valid - valid.mean()) / valid.std()


def _sector_neutral_zscore(raw: pd.Series, sectors: pd.Series) -> pd.Series:
    """Sector 내 z-score. sector 정보 없는 종목은 전체 z-score 적용."""
    common = raw.dropna().index.intersection(sectors.dropna().index)
    if common.empty:
        return _zscore(raw)
    df = pd.DataFrame({"score": raw[common], "sector": sectors[common]})
    return df.groupby("sector")["score"].transform(
        lambda x: (x - x.mean()) / x.std() if len(x) > 1 and x.std() > 0 else 0.0
    )


# ---------------------------------------------------------------------------
# Individual factors
# ---------------------------------------------------------------------------

def compute_momentum(
    prices: pd.DataFrame, lookback: int = 252, skip: int = 21,
) -> pd.Series:
    """12-1 momentum: lookback일 수익, 직전 skip일 제외."""
    close = _get_close(prices)
    if len(close) < lookback + 1:
        return pd.Series(dtype=float)
    end_price = close.iloc[-(skip + 1)]
    start_price = close.iloc[-(lookback + 1)]
    return ((end_price / start_price) - 1.0).dropna()


def compute_size(fundamentals: pd.DataFrame) -> pd.Series:
    """Size factor: -log(market_cap). 소형주일수록 높은 점수."""
    mc = fundamentals["market_cap"].dropna()
    mc = mc[mc > 0]
    return -np.log(mc)


def compute_value(fundamentals: pd.DataFrame) -> pd.Series:
    """Value composite: z(earnings_yield) + z(book_yield) + z(1/EV_EBITDA) + z(fcf_yield).

    높을수록 저평가. 음수 PE/PB는 제외 (적자/음의 자본).
    """
    pe = fundamentals["pe_ratio"]
    pb = fundamentals["pb_ratio"]
    ev = fundamentals["ev_ebitda"]
    fcf = fundamentals["fcf_yield"]

    components = pd.DataFrame({
        "ey": _zscore(1.0 / pe.where(pe > 0)),
        "by": _zscore(1.0 / pb.where(pb > 0)),
        "ebit": _zscore(1.0 / ev.where(ev > 0)),
        "fcf": _zscore(fcf),
    })
    return components.mean(axis=1).dropna()


def compute_quality(fundamentals: pd.DataFrame) -> pd.Series:
    """Quality composite: z(ROE) + z(gross_margin) + z(-debt/equity).

    높을수록 고품질. 음수 D/E는 제외.
    """
    de = fundamentals["debt_equity"]
    components = pd.DataFrame({
        "roe": _zscore(fundamentals["roe"]),
        "gm": _zscore(fundamentals["gross_margin"]),
        "lev": _zscore(-de.where(de >= 0)),
    })
    return components.mean(axis=1).dropna()


def compute_lowvol(prices: pd.DataFrame, lookback: int = 60) -> pd.Series:
    """Low volatility: -annualized_vol(lookback days). 낮은 변동성 선호."""
    close = _get_close(prices)
    if len(close) < lookback + 1:
        return pd.Series(dtype=float)
    daily_ret = close.iloc[-lookback:].pct_change().dropna()
    vol = daily_ret.std() * np.sqrt(252)
    return (-vol).dropna()


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------

def compute_composite(
    prices: pd.DataFrame, fundamentals: pd.DataFrame | None,
    momentum_lookback: int = 252, lowvol_lookback: int = 60,
    sector_neutral: bool = True, weights: dict[str, float] | None = None,
) -> pd.Series:
    """5-factor weighted composite score. 높을수록 매수 후보."""
    if weights is None:
        weights = {"size": 0.2, "value": 0.2, "momentum": 0.2,
                   "quality": 0.2, "lowvol": 0.2}

    raw: dict[str, pd.Series] = {}
    raw["momentum"] = compute_momentum(prices, lookback=momentum_lookback)
    raw["lowvol"] = compute_lowvol(prices, lookback=lowvol_lookback)

    has_fund = fundamentals is not None and not fundamentals.empty
    if has_fund:
        raw["size"] = compute_size(fundamentals)
        raw["value"] = compute_value(fundamentals)
        raw["quality"] = compute_quality(fundamentals)

    sectors = (fundamentals["sector"]
               if has_fund and "sector" in fundamentals.columns else None)

    scored: dict[str, pd.Series] = {}
    for name, series in raw.items():
        if series.empty:
            continue
        z = (_sector_neutral_zscore(series, sectors)
             if sector_neutral and sectors is not None
             else _zscore(series))
        scored[name] = z * weights.get(name, 0.2)

    if not scored:
        return pd.Series(dtype=float)
    return pd.DataFrame(scored).sum(axis=1).dropna()


def compute_gap_events(
    prices: pd.DataFrame, gap_threshold: float = 0.05, volume_multiple: float = 1.5,
) -> pd.DataFrame:
    """Price gap-up 이벤트 (PEAD 보완). 수익률>5% + 거래량>1.5x 평균 → SUE proxy."""
    close = _get_close(prices)
    try:
        vol = prices.xs("volume", level="field", axis=1)
    except KeyError:
        return pd.DataFrame(columns=["ticker", "date", "sue"])
    ret = close.pct_change()
    high_vol = vol > vol.rolling(20).mean() * volume_multiple
    mask = (ret > gap_threshold) & high_vol
    mask = mask.iloc[21:]  # skip warmup
    if "SPY" in mask.columns:
        mask = mask.drop(columns=["SPY"])
    hits = mask.stack()
    hits = hits[hits]
    if hits.empty:
        return pd.DataFrame(columns=["ticker", "date", "sue"])
    rows = [{"ticker": t, "date": d, "sue": float(ret.loc[d, t] / 0.05)}
            for d, t in hits.index]
    return pd.DataFrame(rows)


def compute_sue(earnings: pd.DataFrame) -> pd.DataFrame:
    """SUE = (Actual - Estimate EPS) / std(past surprises). No look-ahead."""
    if earnings.empty:
        return pd.DataFrame(columns=["ticker", "date", "sue"])

    results: list[dict] = []
    for ticker, grp in earnings.groupby("ticker"):
        grp = grp.sort_values("date")
        surprises = (grp["actual_eps"] - grp["estimate_eps"]).values
        for i in range(1, len(grp)):
            # std는 0~i까지 (현재 포함) — 미래 데이터 사용 안 함
            past_std = float(pd.Series(surprises[: i + 1]).std())
            if past_std == 0 or np.isnan(past_std):
                continue
            results.append({
                "ticker": ticker,
                "date": grp.iloc[i]["date"],
                "sue": float(surprises[i] / past_std),
            })

    if not results:
        return pd.DataFrame(columns=["ticker", "date", "sue"])
    return pd.DataFrame(results)
