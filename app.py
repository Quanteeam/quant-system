"""Streamlit UI for the quant system backtest, optimize, and robustness tabs."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from backtest import BacktestEngine
from config import DEFAULT_CONFIG
from factors import compute_composite, compute_gap_events, compute_quality, compute_sue, compute_value
from portfolio import build_event_portfolio, build_multifactor_portfolio
from risk import RiskEngine
from transaction_cost import CostConfig
from trend_filter import TrendFilterConfig, apply_trend_filter
from universe import (
    BENCHMARK_TICKERS,
    LEGACY_UNIVERSE_TICKERS,
    MOMENTUM_SKIP_DAYS,
    build_candidate_tickers,
    compute_load_start,
    prepare_dynamic_metadata,
    select_dynamic_universe,
)

from data_backend import get_pit_fundamentals, load_earnings, load_prices, load_quarterly_cache, load_ticker_metadata


@st.cache_data(show_spinner="Loading price data...")
def get_prices(tickers: tuple[str, ...], start: str, end: str):
    return load_prices(list(tickers), start, end)


@st.cache_data(show_spinner="Loading earnings data...")
def get_earnings(tickers: tuple[str, ...]):
    return load_earnings(list(tickers))


@st.cache_data(show_spinner="Loading quarterly PIT cache...")
def get_quarterly(tickers: tuple[str, ...]):
    return load_quarterly_cache(list(tickers))


@st.cache_data(show_spinner="Loading ticker metadata...")
def get_ticker_metadata():
    return load_ticker_metadata()


@st.cache_data(show_spinner=False)
def get_benchmark_fallback_prices(tickers: tuple[str, ...], start: str, end: str):
    try:
        from data import load_prices as load_yf_prices
    except Exception:
        return pd.DataFrame()
    try:
        return load_yf_prices(list(tickers), start, end)
    except Exception:
        return pd.DataFrame()



def ensure_benchmark_prices(prices: pd.DataFrame, start: str, end: str, required: list[str]):
    existing = set(prices.columns.get_level_values("ticker"))
    missing = tuple(ticker for ticker in required if ticker not in existing)
    if not missing:
        return prices, []
    fallback = get_benchmark_fallback_prices(missing, start, end)
    if fallback.empty:
        return prices, list(missing)
    merged = pd.concat([prices, fallback], axis=1).sort_index(axis=1)
    still_missing = [ticker for ticker in missing if ticker not in set(merged.columns.get_level_values("ticker"))]
    return merged, still_missing



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


@st.cache_data(show_spinner="Running backtest...")
def run_backtest(
    start,
    end,
    top_n,
    mom_lb,
    vol_lb,
    sec_neutral,
    ev_on,
    sue_th,
    max_hold,
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

    metadata = get_ticker_metadata() if dynamic_universe else pd.DataFrame()
    dynamic_meta = prepare_dynamic_metadata(metadata, load_start, end_ts, exchange_scope=exchange_scope)
    dynamic_active = dynamic_universe and not dynamic_meta.empty

    candidate_tickers = build_candidate_tickers(dynamic_meta, dynamic_active, LEGACY_UNIVERSE_TICKERS)
    load_tickers = tuple(sorted(set(candidate_tickers + BENCHMARK_TICKERS)))
    prices = get_prices(load_tickers, str(load_start.date()), end)
    prices, missing_benchmarks = ensure_benchmark_prices(prices, str(load_start.date()), end, BENCHMARK_TICKERS)

    close_field = "adj_close" if ("adj_close" in prices.columns.get_level_values("field")) else "close"
    close = prices.xs(close_field, level="field", axis=1)
    volume = prices.xs("volume", level="field", axis=1)

    investable_close = close[[t for t in candidate_tickers if t in close.columns]].copy()
    investable_volume = volume.reindex(columns=investable_close.columns)
    if investable_close.empty:
        return None

    qcache = get_quarterly(tuple(candidate_tickers))
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

    for d in rebal:
        history = investable_close.loc[:d].dropna(axis=1, how="all")
        if len(history) < min_history_days:
            continue

        if dynamic_active:
            eligible, counts = select_dynamic_universe(
                dynamic_meta,
                d,
                investable_close.loc[d],
                adv20.loc[d],
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
        universe_row = {"date": d, **counts, "selected": 0}
        if not eligible:
            universe_rows.append(universe_row)
            continue

        price_slice = prices.loc[:d, [col for col in prices.columns if col[0] in eligible]]
        pit_fundamentals = get_pit_fundamentals(qcache, investable_close.loc[d, eligible], d)
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
            rows.append((d, weights))

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
        "universe_history": pd.DataFrame(universe_rows).set_index("date") if universe_rows else pd.DataFrame(),
    }

    if tf_cfg and tf_cfg.enable:
        out["mf_nofilter"] = _to_dict(engine.run(mf_wh, eval_start=start_ts))
        mf_filtered = apply_trend_filter(mf_wh, prices, tf_cfg)
        out["mf"] = _to_dict(engine.run(mf_filtered, eval_start=start_ts))
    else:
        out["mf"] = _to_dict(engine.run(mf_wh, eval_start=start_ts))

    if ev_on:
        signal_prices = prices.loc[:, [col for col in prices.columns if col[0] in investable_close.columns]]
        sue = compute_sue(get_earnings(tuple(candidate_tickers)))
        gaps = compute_gap_events(signal_prices)
        sue = pd.concat([sue, gaps], ignore_index=True).drop_duplicates(subset=["ticker", "date"], keep="first")
        latest_fund = get_pit_fundamentals(qcache, investable_close.iloc[-1], end_ts)
        q_scores = compute_quality(latest_fund) if not latest_fund.empty else None
        v_scores = compute_value(latest_fund) if not latest_fund.empty else None
        ev_w = build_event_portfolio(sue, signal_prices, q_scores, v_scores, sue_threshold=sue_th, max_holding_days=max_hold)
        ev_w = ev_w.reindex(investable_close.index, method="ffill").fillna(0.0)
        out["event"] = _to_dict(engine.run(ev_w, eval_start=start_ts))

        cols = sorted(set(mf_wh.columns) | set(ev_w.columns))
        combined = (
            mf_wh.reindex(columns=cols, fill_value=0.0) * 0.4
            + ev_w.reindex(index=mf_wh.index, columns=cols, fill_value=0.0) * 0.6
        )
        risk_eng = RiskEngine()
        safe_w, risk_events = risk_eng.apply_risk_to_backtest(combined, prices)
        out["hybrid"] = _to_dict(engine.run(safe_w, eval_start=start_ts))
        out["risk_events"] = risk_events

    return out



def main():
    st.set_page_config(page_title="Quant System", layout="wide")
    st.title("Quant System")
    tab_bt, tab_opt, tab_rob = st.tabs(["Backtest", "Optimize", "Robustness"])

    bp = st.session_state.get("best_fw", {})
    def_mom = bp.get("momentum", 0.30)
    def_qual = bp.get("quality", 0.25)
    def_val = bp.get("value", 0.20)
    def_size = bp.get("size", 0.10)
    def_lvol = bp.get("lowvol", 0.15)
    bp_raw = st.session_state.get("best_params", {})
    cfg = DEFAULT_CONFIG

    with st.sidebar:
        st.header("Backtest Setup")
        start = str(st.date_input("Start Date", value=pd.Timestamp("2020-01-01")))
        end = str(st.date_input("End Date", value=pd.Timestamp("2024-12-31")))
        top_n = st.number_input("Top N", 5, 100, bp_raw.get("top_n", 20), step=1)
        mom_lb = st.number_input("Momentum Lookback", 60, 252, bp_raw.get("momentum_lb", 252), step=5)
        vol_lb = st.number_input("Low Vol Lookback", 20, 180, bp_raw.get("low_vol_lb", 60), step=5)
        sec_n = st.checkbox("Sector Neutral", True)

        st.divider()
        st.header("Universe")
        dynamic_default = cfg.data.backend == "local"
        dynamic_universe = st.checkbox("Use Dynamic Universe", value=dynamic_default)
        exchange_scope = st.selectbox("Exchange Scope", ["NASDAQ", "US Common"], index=0)
        min_price = st.number_input("Min Price ($)", min_value=1.0, max_value=100.0, value=float(cfg.universe.min_price_usd), step=1.0)
        min_adv_m = st.number_input("Min ADV ($M)", min_value=0.5, max_value=100.0, value=float(cfg.universe.min_adv_usd / 1_000_000), step=0.5)
        min_adv_usd = float(min_adv_m * 1_000_000)

        st.divider()
        st.header("Factor Weights")

        def _weight_row(label, default, key):
            c1, c2 = st.columns([3, 1])
            with c1:
                value = st.slider(label, 0.0, 1.0, default, 0.01, key=f"{key}_s")
            with c2:
                value = st.number_input("", 0.0, 1.0, value, 0.01, key=f"{key}_n", label_visibility="collapsed")
            return value

        w_mom = _weight_row("Momentum", def_mom, "mom")
        w_qual = _weight_row("Quality", def_qual, "qual")
        w_val = _weight_row("Value", def_val, "val")
        w_size = _weight_row("Size", def_size, "size")
        w_lvol = _weight_row("Low Vol", def_lvol, "lvol")

        w_total = round(w_mom + w_qual + w_val + w_size + w_lvol, 2)
        if abs(w_total - 1.0) < 0.01:
            st.success(f"Weights sum: {w_total:.2f}")
        else:
            st.error(f"Weights sum: {w_total:.2f} (must be 1.00)")

        if st.button("Normalize to 1.0") and w_total > 0:
            st.session_state["best_fw"] = {
                "momentum": w_mom / w_total,
                "quality": w_qual / w_total,
                "value": w_val / w_total,
                "size": w_size / w_total,
                "lowvol": w_lvol / w_total,
            }
            st.rerun()

        st.divider()
        st.header("Trend Filter")
        tf_on = st.checkbox("Enable Trend Filter", False)
        tf_ma = st.selectbox("MA Period", [100, 150, 200, 250], index=2) if tf_on else 200
        tf_mode = st.radio("Mode", ["soft", "hard"], horizontal=True) if tf_on else "soft"
        tf_bench = st.selectbox("Benchmark", ["SPY", "QQQ", "IWM"]) if tf_on else "SPY"

        st.divider()
        st.header("Transaction Cost")
        cost_preset = st.checkbox("Realistic Cost (IBKR)", True)
        if cost_preset:
            comm_pct = 0.00005
            slip_pct = 0.0005
            st.caption("Commission 0.005% + Slippage 0.05%")
        else:
            comm_pct = st.slider("Commission (%)", 0.0, 0.01, 0.00005, 0.00001, format="%.5f%%", key="comm")
            slip_pct = st.slider("Slippage (%)", 0.0, 0.05, 0.0005, 0.0001, format="%.4f%%", key="slip")
        cost_cfg = CostConfig(commission_pct=comm_pct, slippage_pct=slip_pct)

        st.divider()
        st.header("PEAD")
        ev_on = st.checkbox("Enable Hybrid 60/40", False)
        sue_th = st.slider("SUE Threshold", 0.3, 3.0, 1.0, 0.1) if ev_on else 1.0
        max_hold = st.slider("Max Holding", 15, 90, 60) if ev_on else 60
        run = st.button("Run", type="primary", use_container_width=True)

    with tab_opt:
        from app_optimize import render_optimize_tab

        render_optimize_tab()

    with tab_rob:
        from app_robustness import render_robustness_tab

        render_robustness_tab()

    tf_cfg = TrendFilterConfig(enable=tf_on, ma_period=tf_ma, mode=tf_mode, benchmark=tf_bench)
    with tab_bt:
        from app_backtest import render_backtest

        render_backtest(
            run,
            start,
            end,
            top_n,
            mom_lb,
            vol_lb,
            sec_n,
            ev_on,
            sue_th,
            max_hold,
            w_mom,
            w_qual,
            w_val,
            w_size,
            w_lvol,
            tf_cfg,
            run_backtest,
            cost_cfg,
            dynamic_universe,
            exchange_scope,
            min_price,
            min_adv_usd,
        )


if __name__ == "__main__":
    main()
