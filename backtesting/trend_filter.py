"""Trend Filter ???댄룊??湲곕컲 ?ъ????ш린 議곗젅 (MaxDD 諛⑹뼱)."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class TrendFilterConfig:
    enable: bool = False
    ma_period: int = 200
    mode: str = "soft"        # "hard" (0%) or "soft" (50%)
    benchmark: str = "SPY"
    rf_annual: float = 0.04   # ?꾧툑 鍮꾩쨷 臾댁쐞???섏씡瑜?(??4%)


def apply_trend_filter(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    config: TrendFilterConfig,
) -> pd.DataFrame:
    """Trend filter ?곸슜: benchmark MA 湲곗? ?ъ???異뺤냼.

    benchmark 醫낃? > MA ??multiplier 1.0
    benchmark 醫낃? < MA ??hard: 0.0, soft: 0.5
    """
    if not config.enable:
        return weights

    # benchmark 醫낃? 異붿텧
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

    # 媛??좎쭨蹂?multiplier ??T??醫낃? ?좏샇 ??T+1???곸슜 (look-ahead 諛⑹?)
    above_ma = (bench_close >= ma).shift(1)
    if config.mode == "hard":
        multiplier = above_ma.astype(float)  # 1.0 or 0.0
    else:
        multiplier = above_ma.astype(float) * 0.5 + 0.5  # 1.0 or 0.5

    # weights??留욎떠 reindex
    multiplier = multiplier.reindex(weights.index, method="ffill").fillna(1.0)

    return weights.multiply(multiplier, axis=0)
