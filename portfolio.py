"""포트폴리오 구성 모듈.

Phase 1: build_multifactor_portfolio (top N equal weight) + combine_sleeves.
Phase 4: build_event_portfolio stub.
"""
from __future__ import annotations

import pandas as pd


def build_multifactor_portfolio(
    scores: pd.Series,
    top_n: int = 20,
) -> pd.Series:
    """Top N equal weight.

    Args:
        scores: ticker → factor score. 높을수록 좋음.
        top_n: 보유 종목 수.

    Returns:
        ticker → weight. 합계 = 1.0.
    """
    valid = scores.dropna()
    if valid.empty:
        return pd.Series(dtype=float)
    top = valid.nlargest(min(top_n, len(valid)))
    weight = 1.0 / len(top)
    return pd.Series(weight, index=top.index)


def build_event_portfolio(*args, **kwargs) -> pd.Series:
    """Phase 4."""
    raise NotImplementedError("Phase 4")


def combine_sleeves(
    mf_weights: pd.Series,
    event_weights: pd.Series,
    mf_alloc: float = 0.4,
    event_alloc: float = 0.6,
) -> pd.Series:
    """두 sleeve를 alloc 비율로 통합. 동일 종목 weight 합산."""
    mf = mf_weights * mf_alloc
    ev = event_weights * event_alloc
    return mf.add(ev, fill_value=0.0)
