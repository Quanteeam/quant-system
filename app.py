"""Streamlit entrypoint for the quant system."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from backtesting.costs import CostConfig
from backtesting.trend_filter import TrendFilterConfig
from core.config import DEFAULT_CONFIG
from core.universe import BENCHMARK_TICKERS
from data_layer.backend import load_earnings, load_prices, load_quarterly_cache, load_ticker_metadata
from strategies.base import StrategyDataAccess
from strategies.registry import get_strategy, strategy_options


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
        from data_layer.yfinance_provider import load_prices as load_yf_prices
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


def _data_access() -> StrategyDataAccess:
    return StrategyDataAccess(
        get_prices=get_prices,
        get_earnings=get_earnings,
        get_quarterly=get_quarterly,
        get_ticker_metadata=get_ticker_metadata,
        ensure_benchmark_prices=ensure_benchmark_prices,
    )


@st.cache_data(show_spinner="Running backtest...")
def run_backtest(
    strategy_key,
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
    strategy = get_strategy(strategy_key)
    return strategy.run_backtest(
        data=_data_access(),
        start=start,
        end=end,
        top_n=top_n,
        mom_lb=mom_lb,
        vol_lb=vol_lb,
        sec_neutral=sec_neutral,
        ev_on=ev_on,
        sue_th=sue_th,
        max_hold=max_hold,
        fw=fw,
        comm_pct=comm_pct,
        slip_pct=slip_pct,
        dynamic_universe=dynamic_universe,
        exchange_scope=exchange_scope,
        min_price=min_price,
        min_adv_usd=min_adv_usd,
        tf_cfg=tf_cfg,
    )


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
        st.header("Model")
        options = strategy_options()
        labels = [option.label for option in options]
        selected_label = st.selectbox("Strategy", labels, index=0)
        strategy_key = next(option.key for option in options if option.label == selected_label)

        st.divider()
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
        min_price = st.number_input(
            "Min Price ($)", min_value=1.0, max_value=100.0,
            value=float(cfg.universe.min_price_usd), step=1.0,
        )
        min_adv_m = st.number_input(
            "Min ADV ($M)", min_value=0.5, max_value=100.0,
            value=float(cfg.universe.min_adv_usd / 1_000_000), step=0.5,
        )
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
        tf_bench = st.selectbox("Benchmark", BENCHMARK_TICKERS) if tf_on else "SPY"

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
        from ui.optimize import render_optimize_tab

        render_optimize_tab()

    with tab_rob:
        from ui.robustness import render_robustness_tab

        render_robustness_tab()

    tf_cfg = TrendFilterConfig(enable=tf_on, ma_period=tf_ma, mode=tf_mode, benchmark=tf_bench)
    with tab_bt:
        from ui.backtest import render_backtest

        render_backtest(
            run,
            strategy_key,
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
