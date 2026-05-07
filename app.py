"""Streamlit UI — Quant System (Backtest / Optimize / Robustness)."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import streamlit as st

from backtest import BacktestEngine
from data import load_earnings, load_fundamentals, load_prices
from data_sharadar import (
    load_fundamentals_history as load_sharadar_fundamentals_history,
    load_prices as load_sharadar_prices,
    select_fundamentals_snapshot,
)
from factors import compute_composite, compute_gap_events, compute_quality, compute_sue, compute_value
from portfolio import build_event_portfolio, build_multifactor_portfolio
from risk import RiskEngine
from trend_filter import TrendFilterConfig

UNIVERSE_TICKERS = [
    "CRWD", "DDOG", "NET", "ZS", "HUBS", "PAYC", "FTNT", "SNAP",
    "MRVL", "SWKS", "MPWR", "ON", "ENTG", "MKSI",
    "ALGN", "DXCM", "HOLX", "TECH", "NBIX", "EXAS",
    "DECK", "POOL", "WSM", "DPZ", "WING", "BURL",
    "AXON", "GNRC", "TREX", "RBC", "FND", "SITE",
    "LPLA", "RGA", "EWBC", "KNSL", "WBS", "CFR",
    "TRGP", "AR", "CLF", "ATI", "GPK", "UFPI",
    "ELS", "AMH",
]
ALL_TICKERS = UNIVERSE_TICKERS + ["SPY"]
DATA_BACKEND = os.environ.get("QUANT_DATA_BACKEND", "yfinance").lower()


@st.cache_data(show_spinner="가격 데이터 로딩 중...")
def get_prices(start: str, end: str):
    if DATA_BACKEND == "sharadar":
        return load_sharadar_prices(ALL_TICKERS, start, end)
    return load_prices(ALL_TICKERS, start, end)

@st.cache_data(show_spinner="펀더멘탈 로딩 중...")
def get_fundamentals():
    if DATA_BACKEND == "sharadar":
        return load_sharadar_fundamentals_history(UNIVERSE_TICKERS)
    return load_fundamentals(UNIVERSE_TICKERS)

@st.cache_data(show_spinner="실적 데이터 로딩 중...")
def get_earnings():
    return load_earnings(UNIVERSE_TICKERS)


def _to_dict(r):
    return {"equity": r.equity_curve, "drawdown": r.drawdown, "benchmark": r.benchmark_curve,
            "cagr": r.cagr, "sharpe": r.sharpe, "max_drawdown": r.max_drawdown,
            "calmar": r.calmar, "total_return": r.total_return, "monthly": r.monthly_returns}


@st.cache_data(show_spinner="백테스트 실행 중...")
def run_backtest(start, end, top_n, mom_lb, vol_lb, sec_neutral, ev_on, sue_th, max_hold,
                 fw=None):
    prices, fund = get_prices(start, end), get_fundamentals()
    for f in ("adj_close", "close"):
        try:
            close = prices.xs(f, level="field", axis=1); break
        except KeyError:
            continue
    uni = close[[t for t in UNIVERSE_TICKERS if t in close.columns]]
    rebal = uni.resample("ME").last().index

    rows = []
    for d in rebal:
        h = uni.loc[:d].dropna(axis=1, how="all")
        if len(h) < mom_lb + 5: continue
        vt = h.columns.tolist()
        sp = prices.loc[:d, [c for c in prices.columns if c[0] in vt]]
        if DATA_BACKEND == "sharadar":
            sf = select_fundamentals_snapshot(fund, vt, d)
        else:
            sf = fund.loc[fund.index.isin(vt)]
        fweights = fw or {"momentum": 0.3, "quality": 0.25, "value": 0.2, "size": 0.1, "lowvol": 0.15}
        sc = compute_composite(sp, sf, momentum_lookback=mom_lb, lowvol_lookback=vol_lb,
                               sector_neutral=sec_neutral, weights=fweights)
        w = build_multifactor_portfolio(sc, top_n=top_n)
        if not w.empty: rows.append((d, w))
    if not rows: return None

    mf_wh = pd.DataFrame([w for _, w in rows], index=[d for d, _ in rows]).fillna(0)
    mf_wh = mf_wh.reindex(uni.index, method="ffill").fillna(0)
    engine = BacktestEngine(prices, commission_bps=1, slippage_bps=30)
    out = {"mf": _to_dict(engine.run(mf_wh)), "weights": rows[-1][1]}

    if ev_on:
        sue = compute_sue(get_earnings())
        gaps = compute_gap_events(prices)
        sue = pd.concat([sue, gaps], ignore_index=True).drop_duplicates(
            subset=["ticker", "date"], keep="first")
        latest_fund = (
            select_fundamentals_snapshot(fund, UNIVERSE_TICKERS, prices.index[-1])
            if DATA_BACKEND == "sharadar"
            else fund
        )
        q = compute_quality(latest_fund) if not latest_fund.empty else None
        v = compute_value(latest_fund) if not latest_fund.empty else None
        ev_w = build_event_portfolio(sue, prices, q, v, sue_threshold=sue_th, max_holding_days=max_hold)
        out["event"] = _to_dict(engine.run(ev_w.reindex(uni.index, method="ffill").fillna(0)))
        cols = sorted(set(mf_wh.columns) | set(ev_w.columns))
        combined = (mf_wh.reindex(columns=cols, fill_value=0) * 0.4
                    + ev_w.reindex(index=mf_wh.index, columns=cols, fill_value=0) * 0.6)
        risk_eng = RiskEngine()
        safe_w, risk_events = risk_eng.apply_risk_to_backtest(combined, prices)
        out["hybrid"] = _to_dict(engine.run(safe_w))
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

    with st.sidebar:
        st.header("Multi-factor")
        start = str(st.date_input("시작일", value=pd.Timestamp("2020-01-01")))
        end = str(st.date_input("종료일", value=pd.Timestamp("2024-12-31")))
        top_n = st.number_input("Top N", 5, 50, bp_raw.get("top_n", 20), step=1)
        mom_lb = st.number_input("Momentum LB", 60, 252, bp_raw.get("momentum_lb", 252), step=5)
        vol_lb = st.number_input("Low Vol LB", 20, 180, bp_raw.get("low_vol_lb", 60), step=5)
        sec_n = st.checkbox("Sector Neutral", True)
        st.divider()
        st.header("Factor Weights")

        def _weight_row(label, default, key):
            c1, c2 = st.columns([3, 1])
            with c1:
                v = st.slider(label, 0.0, 1.0, default, 0.01, key=f"{key}_s")
            with c2:
                v = st.number_input("", 0.0, 1.0, v, 0.01, key=f"{key}_n",
                                    label_visibility="collapsed")
            return v

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
                "momentum": w_mom/w_total, "quality": w_qual/w_total,
                "value": w_val/w_total, "size": w_size/w_total, "lowvol": w_lvol/w_total}
            st.rerun()

        st.divider()
        st.header("Trend Filter")
        tf_on = st.checkbox("Enable Trend Filter", False)
        tf_ma = st.selectbox("MA Period", [100, 150, 200, 250], index=2) if tf_on else 200
        tf_mode = st.radio("Mode", ["soft", "hard"], horizontal=True) if tf_on else "soft"
        tf_bench = st.selectbox("Benchmark", ["SPY", "QQQ", "IWM"]) if tf_on else "SPY"
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
        render_backtest(run, start, end, top_n, mom_lb, vol_lb, sec_n,
                        ev_on, sue_th, max_hold, w_mom, w_qual, w_val, w_size, w_lvol,
                        tf_cfg, run_backtest, get_prices)

if __name__ == "__main__":
    main()
