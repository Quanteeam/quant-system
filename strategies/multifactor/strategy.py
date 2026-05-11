"""Current multi-factor strategy runner.

This module owns the current factor model workflow so future strategies can live
beside it without editing the same files.
"""
from __future__ import annotations

import pandas as pd

from backtesting.costs import CostConfig
from backtesting.engine import BacktestEngine
from backtesting.trend_filter import TrendFilterConfig, apply_trend_filter
from core.universe import (
    BENCHMARK_TICKERS,
    LEGACY_UNIVERSE_TICKERS,
    MOMENTUM_SKIP_DAYS,
    build_candidate_tickers,
    compute_load_start,
    prepare_dynamic_metadata,
    select_dynamic_universe,
)
from strategies.base import StrategyDataAccess, StrategyDefinition
from strategies.multifactor.factors import compute_composite
from strategies.multifactor.portfolio import build_multifactor_portfolio
from data_layer.backend import get_pit_fundamentals


DEFINITION = StrategyDefinition(
    key="multifactor",
    label="Multi-factor",
    description="Current momentum, quality, value, size, and low-volatility strategy.",
)


def _to_dict(result):
    return {
        "equity": result.equity_curve,
        "drawdown": result.drawdown,
        "benchmark": result.benchmark_curve,
        "benchmark_curves": result.benchmark_curves,
        "benchmark_stats": result.benchmark_stats,
        "cagr": result.cagr,
        "sharpe": result.sharpe,
        "max_drawdown": result.max_drawdown,
        "calmar": result.calmar,
        "total_return": result.total_return,
        "monthly": result.monthly_returns,
        "annual_turnover": result.annual_turnover,
        "total_cost": result.total_cost,
        "cost_drag": result.cost_drag,
        "monthly_turnover": result.monthly_turnover,
        "monthly_entries": result.monthly_entries,
        "monthly_exits": result.monthly_exits,
        "average_holding_days": result.average_holding_days,
        "invested_ratio": result.invested_ratio,
    }


def run_backtest(
    data: StrategyDataAccess,
    start,
    end,
    top_n,
    mom_lb,
    vol_lb,
    sec_neutral,
    fw=None,
    comm_pct=0.00005,
    slip_pct=0.0005,
    dynamic_universe=False,
    exchange_scope="NASDAQ",
    min_price=5.0,
    min_adv_usd=5_000_000.0,
    tf_cfg: TrendFilterConfig | None = None,
):
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    load_start = compute_load_start(start_ts, momentum_lookback=mom_lb, lowvol_lookback=vol_lb)

    metadata = data.get_ticker_metadata() if dynamic_universe else pd.DataFrame()
    dynamic_meta = prepare_dynamic_metadata(metadata, load_start, end_ts, exchange_scope=exchange_scope)
    dynamic_active = dynamic_universe and not dynamic_meta.empty

    candidate_tickers = build_candidate_tickers(dynamic_meta, dynamic_active, LEGACY_UNIVERSE_TICKERS)
    load_tickers = tuple(sorted(set(candidate_tickers + BENCHMARK_TICKERS)))
    prices = data.get_prices(load_tickers, str(load_start.date()), end)
    prices, missing_benchmarks = data.ensure_benchmark_prices(
        prices, str(load_start.date()), end, BENCHMARK_TICKERS
    )

    close_field = "adj_close" if "adj_close" in prices.columns.get_level_values("field") else "close"
    close = prices.xs(close_field, level="field", axis=1)
    volume = prices.xs("volume", level="field", axis=1)

    investable_close = close[[ticker for ticker in candidate_tickers if ticker in close.columns]].copy()
    investable_volume = volume.reindex(columns=investable_close.columns)
    if investable_close.empty:
        return None

    qcache = data.get_quarterly(tuple(candidate_tickers))
    adv20 = (investable_close * investable_volume).rolling(20, min_periods=20).mean()
    min_history_days = max(mom_lb + MOMENTUM_SKIP_DAYS + 2, vol_lb + 2)
    factor_weights = fw or {
        "momentum": 0.3,
        "quality": 0.25,
        "value": 0.2,
        "size": 0.1,
        "lowvol": 0.15,
    }

    rebal = investable_close.index.to_series().groupby(pd.Grouper(freq="ME")).last().dropna().tolist()
    rows: list[tuple[pd.Timestamp, pd.Series]] = []
    universe_rows: list[dict] = []

    for rebalance_date in rebal:
        history = investable_close.loc[:rebalance_date].dropna(axis=1, how="all")
        if len(history) < min_history_days:
            continue

        if dynamic_active:
            eligible, counts = select_dynamic_universe(
                dynamic_meta,
                rebalance_date,
                investable_close.loc[rebalance_date],
                adv20.loc[rebalance_date],
                history,
                min_price=min_price,
                min_adv_usd=min_adv_usd,
                min_history_days=min_history_days,
            )
        else:
            eligible = [ticker for ticker in LEGACY_UNIVERSE_TICKERS if ticker in history.columns]
            counts = {
                "listed": len(eligible),
                "price_ok": len(eligible),
                "adv_ok": len(eligible),
                "history_ok": len(eligible),
                "eligible": len(eligible),
            }

        eligible = [ticker for ticker in eligible if ticker in history.columns]
        universe_row = {"date": rebalance_date, **counts, "selected": 0}
        if not eligible:
            universe_rows.append(universe_row)
            continue

        price_slice = prices.loc[:rebalance_date, [col for col in prices.columns if col[0] in eligible]]
        pit_fundamentals = get_pit_fundamentals(qcache, investable_close.loc[rebalance_date, eligible], rebalance_date)
        if not pit_fundamentals.empty:
            pit_fundamentals = pit_fundamentals.reindex(eligible)

        scores = compute_composite(
            price_slice,
            pit_fundamentals,
            momentum_lookback=mom_lb,
            lowvol_lookback=vol_lb,
            sector_neutral=sec_neutral,
            weights=factor_weights,
        )
        scores = scores.reindex(eligible).dropna()
        weights = build_multifactor_portfolio(scores, top_n=top_n)
        universe_row["selected"] = int(len(weights))
        universe_rows.append(universe_row)
        if not weights.empty:
            rows.append((rebalance_date, weights))

    if not rows:
        return None

    mf_wh = pd.DataFrame([weights for _, weights in rows], index=[d for d, _ in rows]).fillna(0.0)
    mf_wh = mf_wh.reindex(investable_close.index, method="ffill").fillna(0.0)
    mf_wh = mf_wh.reindex(columns=investable_close.columns, fill_value=0.0)

    cc = CostConfig(commission_pct=comm_pct, slippage_pct=slip_pct)
    engine = BacktestEngine(prices, cost_config=cc)

    out = {
        "weights": rows[-1][1].sort_values(ascending=False),
        "run_info": {
            "strategy_key": DEFINITION.key,
            "strategy_label": DEFINITION.label,
            "eval_start": start_ts,
            "eval_end": end_ts,
            "load_start": load_start,
            "dynamic_universe": dynamic_active,
            "requested_dynamic_universe": dynamic_universe,
            "exchange_scope": exchange_scope,
            "candidate_count": len(candidate_tickers),
            "warmup_days": len(investable_close.loc[: start_ts - pd.Timedelta(days=1)]),
            "min_history_days": min_history_days,
            "benchmarks": [ticker for ticker in BENCHMARK_TICKERS if ticker in close.columns],
            "missing_benchmarks": missing_benchmarks,
        },
        "universe_history": pd.DataFrame(universe_rows).set_index("date")
        if universe_rows
        else pd.DataFrame(),
    }

    if tf_cfg and tf_cfg.enable:
        out["mf_nofilter"] = _to_dict(engine.run(mf_wh, eval_start=start_ts))
        mf_filtered = apply_trend_filter(mf_wh, prices, tf_cfg)
        out["mf"] = _to_dict(engine.run(mf_filtered, eval_start=start_ts))
    else:
        out["mf"] = _to_dict(engine.run(mf_wh, eval_start=start_ts))

    return out


