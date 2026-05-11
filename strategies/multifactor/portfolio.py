"""Portfolio construction for the active multi-factor workflow.

Event portfolio and sleeve-combination helpers are legacy utilities and are not
called by the current strategy runner.
"""
from __future__ import annotations

import pandas as pd


def build_multifactor_portfolio(
    scores: pd.Series,
    top_n: int = 20,
) -> pd.Series:
    """Top N equal weight.

    Args:
        scores: ticker ??factor score. ?믪쓣?섎줉 醫뗭쓬.
        top_n: 蹂댁쑀 醫낅ぉ ??

    Returns:
        ticker ??weight. ?⑷퀎 = 1.0.
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
    """Legacy event-driven portfolio weights.

    This helper is retained for historical tests only. The active strategy
    runner does not call it.

    Entry: ?ㅼ쟻 諛쒗몴 D+1 (D+0 noise ?뚰뵾).
    Exit (癒쇱? ?꾨떖): D+max_holding_days / ?ㅼ쓬 ?ㅼ쟻 D-3 / stop loss.
    Position: position_size per stock, max_stocks ?숈떆 蹂댁쑀.

    Loose filter: quality_scores > 0 and value_scores > 0
    (sector-neutral z-score 湲곗?, ??sector median ?댁긽).

    ??analyst revision / 吏곸쟾 10???ㅻⅨ 諛쒗몴 ?꾪꽣??Phase 6?먯꽌 異붽?.

    Returns: weights_history (index=date, columns=tickers, values=weight).
    """
    # Close prices 異붿텧
    for field in ("adj_close", "close"):
        try:
            close = prices.xs(field, level="field", axis=1)
            break
        except KeyError:
            continue

    dates = close.index

    # SUE ?묒쓽 ?쒗봽?쇱씠利덈쭔, threshold ?댁긽
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

    # ?ㅼ쓬 ?ㅼ쟻??lookup (exit 怨꾩궛??
    earn_dates_map = sue_signals.groupby("ticker")["date"].apply(
        lambda x: sorted(x.tolist())
    ).to_dict()

    # 媛?signal?????entry/exit 寃곗젙
    positions: list[tuple[str, int, int]] = []  # (ticker, entry_idx, exit_idx)

    for _, sig in qualified.iterrows():
        ticker = sig["ticker"]
        sig_date = pd.Timestamp(sig["date"])

        if ticker not in close.columns:
            continue

        # Entry: 諛쒗몴???ㅼ쓬 泥?嫄곕옒??(D+1)
        entry_mask = dates > sig_date
        if not entry_mask.any():
            continue
        entry_idx = int(entry_mask.argmax())
        entry_price = close.iloc[entry_idx].get(ticker)
        if pd.isna(entry_price) or entry_price <= 0:
            continue

        # ?ㅼ쓬 ?ㅼ쟻????D-3 exit
        future_earn = [pd.Timestamp(d) for d in earn_dates_map.get(ticker, [])
                       if pd.Timestamp(d) > sig_date]
        next_earn_exit = (future_earn[0] - pd.Timedelta(days=3)) if future_earn else None

        # Exit 寃곗젙: ?좎쭨蹂??쒗쉶
        exit_idx = entry_idx
        for j in range(entry_idx, min(entry_idx + max_holding_days + 1, len(dates))):
            d = dates[j]
            # ?ㅼ쓬 ?ㅼ쟻 D-3 ?꾨떖
            if next_earn_exit is not None and d >= next_earn_exit:
                break
            # Stop loss
            p = close.iloc[j].get(ticker)
            if pd.notna(p) and (p / entry_price - 1) <= stop_loss:
                break
            exit_idx = j

        positions.append((ticker, entry_idx, exit_idx))

    # Daily weights ?앹꽦
    tickers_used = sorted(set(t for t, _, _ in positions))
    if not tickers_used:
        return pd.DataFrame(0.0, index=dates, columns=close.columns)

    weights = pd.DataFrame(0.0, index=dates, columns=tickers_used)
    for ticker, entry_idx, exit_idx in positions:
        # ?숈떆 蹂댁쑀 ?쒕룄 泥댄겕 (entry ?쒖젏)
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
    """??sleeve瑜?alloc 鍮꾩쑉濡??듯빀. ?숈씪 醫낅ぉ weight ?⑹궛."""
    mf = mf_weights * mf_alloc
    ev = event_weights * event_alloc
    return mf.add(ev, fill_value=0.0)
