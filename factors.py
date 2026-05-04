"""팩터 계산 모듈.

Phase 1: compute_momentum만 실제 구현. 나머지는 Phase 2~3 stub.
"""
from __future__ import annotations

import pandas as pd


def _get_close(prices: pd.DataFrame) -> pd.DataFrame:
    """prices MultiIndex에서 close 추출 (adj_close 우선)."""
    for field in ("adj_close", "close"):
        try:
            return prices.xs(field, level="field", axis=1)
        except KeyError:
            continue
    raise KeyError("prices에 close 또는 adj_close 컬럼이 없습니다.")


def compute_momentum(
    prices: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
) -> pd.Series:
    """12-1 momentum: lookback일 누적수익, 직전 skip일 제외.

    Args:
        prices: MultiIndex columns (ticker, field).
        lookback: 전체 lookback 기간 (일, 기본 252 ≈ 12개월).
        skip: 직전 제외 기간 (일, 기본 21 ≈ 1개월).

    Returns:
        ticker → momentum score. 데이터 부족 종목은 NaN 제외.
    """
    close = _get_close(prices)
    if len(close) < lookback + 1:
        return pd.Series(dtype=float)

    end_price = close.iloc[-(skip + 1)]       # t - skip
    start_price = close.iloc[-(lookback + 1)] # t - lookback
    momentum = (end_price / start_price) - 1.0
    return momentum.dropna()


def compute_size(prices: pd.DataFrame, market_caps: pd.Series) -> pd.Series:
    """Phase 2."""
    raise NotImplementedError("Phase 2")


def compute_value(*args, **kwargs) -> pd.Series:
    """Phase 2."""
    raise NotImplementedError("Phase 2")


def compute_quality(*args, **kwargs) -> pd.Series:
    """Phase 2."""
    raise NotImplementedError("Phase 2")


def compute_lowvol(prices: pd.DataFrame, lookback: int = 60) -> pd.Series:
    """Phase 2."""
    raise NotImplementedError("Phase 2")


def compute_sue(*args, **kwargs) -> pd.Series:
    """Phase 3."""
    raise NotImplementedError("Phase 3")
