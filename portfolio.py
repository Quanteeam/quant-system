"""포트폴리오 구성 모듈.

build_multifactor_portfolio: top-N equal weight (multi-factor sleeve)
build_event_portfolio: PEAD event-driven (event sleeve)
combine_sleeves: 두 sleeve 통합
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


def build_event_portfolio(
    sue_signals: pd.DataFrame,
    prices: pd.DataFrame,
    quality_scores: pd.Series | None = None,
    value_scores: pd.Series | None = None,
    sue_threshold: float = 1.5,
    position_size: float = 0.015,
    max_stocks: int = 40,
    max_holding_days: int = 45,
    stop_loss: float = -0.10,
) -> pd.DataFrame:
    """PEAD event-driven portfolio weights.

    Entry: 실적 발표 D+1 (D+0 noise 회피).
    Exit (먼저 도달): D+max_holding_days / 다음 실적 D-3 / stop loss.
    Position: position_size per stock, max_stocks 동시 보유.

    Loose filter: quality_scores > 0 and value_scores > 0
    (sector-neutral z-score 기준, ≈ sector median 이상).

    ⚠ analyst revision / 직전 10일 다른 발표 필터는 Phase 6에서 추가.

    Returns: weights_history (index=date, columns=tickers, values=weight).
    """
    # Close prices 추출
    for field in ("adj_close", "close"):
        try:
            close = prices.xs(field, level="field", axis=1)
            break
        except KeyError:
            continue

    dates = close.index

    # SUE 양의 서프라이즈만, threshold 이상
    qualified = sue_signals[sue_signals["sue"] > sue_threshold].copy()

    # Loose filter
    if quality_scores is not None:
        q_pass = set(quality_scores[quality_scores > 0].index)
        qualified = qualified[qualified["ticker"].isin(q_pass)]
    if value_scores is not None:
        v_pass = set(value_scores[value_scores > 0].index)
        qualified = qualified[qualified["ticker"].isin(v_pass)]

    if qualified.empty:
        return pd.DataFrame(0.0, index=dates, columns=close.columns)

    # 다음 실적일 lookup (exit 계산용)
    earn_dates_map = sue_signals.groupby("ticker")["date"].apply(
        lambda x: sorted(x.tolist())
    ).to_dict()

    # 각 signal에 대해 entry/exit 결정
    positions: list[tuple[str, int, int]] = []  # (ticker, entry_idx, exit_idx)

    for _, sig in qualified.iterrows():
        ticker = sig["ticker"]
        sig_date = pd.Timestamp(sig["date"])

        if ticker not in close.columns:
            continue

        # Entry: 발표일 다음 첫 거래일 (D+1)
        entry_mask = dates > sig_date
        if not entry_mask.any():
            continue
        entry_idx = int(entry_mask.argmax())
        entry_price = close.iloc[entry_idx].get(ticker)
        if pd.isna(entry_price) or entry_price <= 0:
            continue

        # 다음 실적일 → D-3 exit
        future_earn = [pd.Timestamp(d) for d in earn_dates_map.get(ticker, [])
                       if pd.Timestamp(d) > sig_date]
        next_earn_exit = (future_earn[0] - pd.Timedelta(days=3)) if future_earn else None

        # Exit 결정: 날짜별 순회
        exit_idx = entry_idx
        for j in range(entry_idx, min(entry_idx + max_holding_days + 1, len(dates))):
            d = dates[j]
            # 다음 실적 D-3 도달
            if next_earn_exit is not None and d >= next_earn_exit:
                break
            # Stop loss
            p = close.iloc[j].get(ticker)
            if pd.notna(p) and (p / entry_price - 1) <= stop_loss:
                break
            exit_idx = j

        positions.append((ticker, entry_idx, exit_idx))

    # Daily weights 생성
    tickers_used = sorted(set(t for t, _, _ in positions))
    if not tickers_used:
        return pd.DataFrame(0.0, index=dates, columns=close.columns)

    weights = pd.DataFrame(0.0, index=dates, columns=tickers_used)
    for ticker, entry_idx, exit_idx in positions:
        # 동시 보유 한도 체크 (entry 시점)
        if (weights.iloc[entry_idx] > 0).sum() >= max_stocks:
            continue
        col = weights.columns.get_loc(ticker)
        weights.iloc[entry_idx: exit_idx + 1, col] = position_size

    return weights.reindex(columns=close.columns, fill_value=0.0)


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
