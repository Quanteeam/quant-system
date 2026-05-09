"""Trend Filter — 이평선 기반 포지션 크기 조절 (MaxDD 방어)."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class TrendFilterConfig:
    enable: bool = False
    ma_period: int = 200
    mode: str = "soft"        # "hard" (0%) or "soft" (50%)
    benchmark: str = "SPY"
    rf_annual: float = 0.04   # 현금 비중 무위험 수익률 (연 4%)


def apply_trend_filter(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    config: TrendFilterConfig,
) -> pd.DataFrame:
    """Trend filter 적용: benchmark MA 기준 포지션 축소.

    benchmark 종가 > MA → multiplier 1.0
    benchmark 종가 < MA → hard: 0.0, soft: 0.5
    """
    if not config.enable:
        return weights

    # benchmark 종가 추출
    bench = config.benchmark
    for field in ("adj_close", "close"):
        try:
            close = prices.xs(field, level="field", axis=1)
            break
        except KeyError:
            continue

    if bench not in close.columns:
        return weights

    bench_close = close[bench]
    ma = bench_close.rolling(config.ma_period, min_periods=config.ma_period).mean()

    # 각 날짜별 multiplier — T일 종가 신호 → T+1일 적용 (look-ahead 방지)
    above_ma = (bench_close >= ma).shift(1)
    if config.mode == "hard":
        multiplier = above_ma.astype(float)  # 1.0 or 0.0
    else:
        multiplier = above_ma.astype(float) * 0.5 + 0.5  # 1.0 or 0.5

    # weights에 맞춰 reindex
    multiplier = multiplier.reindex(weights.index, method="ffill").fillna(1.0)

    return weights.multiply(multiplier, axis=0)
