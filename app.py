"""Streamlit UI — Quant System Phase 5 (4-baseline + risk engine)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backtest import BacktestEngine
from data import load_earnings, load_fundamentals, load_prices
from factors import compute_composite, compute_quality, compute_sue, compute_value
from portfolio import build_event_portfolio, build_multifactor_portfolio
from risk import RiskEngine

UNIVERSE_TICKERS = [
    "DVA", "NWSA", "NWS", "PNR", "REG", "RHI", "SJM", "LKQ", "MHK", "IPG",
    "HII", "AIZ", "BWA", "RL", "VFC", "FRT", "UHS", "MOS", "NDSN", "SEE",
    "LNC", "AOS", "WHR", "DXC", "HSIC", "GL", "HBI", "CPB", "CAG", "TPR",
    "TAP", "IVZ", "FLS", "MAS", "LEG", "ALK", "KIM", "PVH", "NI", "LEN",
    "PHM", "NRG", "WBA", "FOXA", "FOX", "NWL", "IRM", "AIV", "BIO", "JNPR",
]
ALL_TICKERS = UNIVERSE_TICKERS + ["SPY"]


@st.cache_data(show_spinner="가격 데이터 로딩 중...")
def get_prices(start: str, end: str):
    return load_prices(ALL_TICKERS, start, end)

@st.cache_data(show_spinner="펀더멘탈 로딩 중...")
def get_fundamentals():
    return load_fundamentals(UNIVERSE_TICKERS)

@st.cache_data(show_spinner="실적 데이터 로딩 중...")
def get_earnings():
    return load_earnings(UNIVERSE_TICKERS)


def _to_dict(r):
    return {"equity": r.equity_curve, "drawdown": r.drawdown, "benchmark": r.benchmark_curve,
            "cagr": r.cagr, "sharpe": r.sharpe, "max_drawdown": r.max_drawdown,
            "calmar": r.calmar, "total_return": r.total_return, "monthly": r.monthly_returns}


@st.cache_data(show_spinner="백테스트 실행 중...")
def run_backtest(start, end, top_n, mom_lb, vol_lb, sec_neutral, ev_on, sue_th, max_hold):
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
        sf = fund.loc[fund.index.isin(vt)]
        sc = compute_composite(sp, sf, momentum_lookback=mom_lb, lowvol_lookback=vol_lb, sector_neutral=sec_neutral)
        w = build_multifactor_portfolio(sc, top_n=top_n)
        if not w.empty: rows.append((d, w))
    if not rows: return None

    mf_wh = pd.DataFrame([w for _, w in rows], index=[d for d, _ in rows]).fillna(0)
    mf_wh = mf_wh.reindex(uni.index, method="ffill").fillna(0)
    engine = BacktestEngine(prices, commission_bps=1, slippage_bps=30)
    out = {"mf": _to_dict(engine.run(mf_wh)), "weights": rows[-1][1]}

    if ev_on:
        sue = compute_sue(get_earnings())
        q = compute_quality(fund) if not fund.empty else None
        v = compute_value(fund) if not fund.empty else None
        ev_w = build_event_portfolio(sue, prices, q, v, sue_threshold=sue_th, max_holding_days=max_hold)
        out["event"] = _to_dict(engine.run(ev_w.reindex(uni.index, method="ffill").fillna(0)))
        # Hybrid 40/60 + risk
        cols = sorted(set(mf_wh.columns) | set(ev_w.columns))
        combined = (mf_wh.reindex(columns=cols, fill_value=0) * 0.4
                    + ev_w.reindex(index=mf_wh.index, columns=cols, fill_value=0) * 0.6)
        risk_eng = RiskEngine()
        safe_w, risk_events = risk_eng.apply_risk_to_backtest(combined, prices)
        out["hybrid"] = _to_dict(engine.run(safe_w))
        out["risk_events"] = risk_events
    return out


def _f(v, pct=True):
    return f"{v*100:.2f}%" if pct else f"{v:.2f}"


def main():
    st.set_page_config(page_title="Quant System", layout="wide")
    st.title("Quant System — 4 Baseline Comparison")

    with st.sidebar:
        st.header("Multi-factor")
        start = str(st.date_input("시작일", value=pd.Timestamp("2020-01-01")))
        end = str(st.date_input("종료일", value=pd.Timestamp("2024-12-31")))
        top_n = st.slider("Top N", 10, 50, 20)
        mom_lb = st.slider("Momentum LB", 60, 252, 252)
        vol_lb = st.slider("Low Vol LB", 20, 120, 60)
        sec_n = st.checkbox("Sector Neutral", True)
        st.divider()
        st.header("PEAD")
        ev_on = st.checkbox("Enable Hybrid 60/40", False)
        sue_th = st.slider("SUE Threshold", 0.5, 3.0, 1.5) if ev_on else 1.5
        max_hold = st.slider("Max Holding", 15, 90, 45) if ev_on else 45
        run = st.button("Run", type="primary", use_container_width=True)

    if not run:
        st.info("파라미터 설정 후 **Run**을 클릭하세요."); return
    res = run_backtest(start, end, top_n, mom_lb, vol_lb, sec_n, ev_on, sue_th, max_hold)
    if not res:
        st.error("데이터 부족"); return

    # Active result
    active = res.get("hybrid", res["mf"])

    # Comparison metrics
    baselines = {"Multi-factor": res["mf"]}
    if "event" in res: baselines["PEAD Only"] = res["event"]
    if "hybrid" in res: baselines["Hybrid 40/60"] = res["hybrid"]
    cols = st.columns(len(baselines))
    for col, (name, r) in zip(cols, baselines.items()):
        col.subheader(name)
        col.metric("CAGR", _f(r["cagr"]))
        col.metric("Sharpe", _f(r["sharpe"], False))
        col.metric("Max DD", _f(r["max_drawdown"]))
        col.metric("Calmar", _f(r["calmar"], False))

    # Equity curves
    fig = go.Figure()
    colors = {"Multi-factor": "#22c55e", "PEAD Only": "#3b82f6", "Hybrid 40/60": "#f59e0b"}
    for name, r in baselines.items():
        eq = r["equity"]
        fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name=name, line=dict(color=colors.get(name, "#fff"))))
    bench = res["mf"]["benchmark"]
    fig.add_trace(go.Scatter(x=bench.index, y=bench.values, name="SPY", line=dict(color="#94a3b8", dash="dash")))
    fig.update_layout(title="Equity Curves (4 Baseline)", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # Drawdown
    fig2 = go.Figure()
    dd = active["drawdown"] * 100
    fig2.add_trace(go.Scatter(x=dd.index, y=dd.values, fill="tozeroy", name="Drawdown",
                              line=dict(color="#ef4444"), fillcolor="rgba(239,68,68,0.2)"))
    fig2.update_layout(title="Drawdown (%)", template="plotly_dark")
    st.plotly_chart(fig2, use_container_width=True)

    # Monthly heatmap
    monthly = active["monthly"]
    if not monthly.empty:
        mdf = monthly.to_frame("r")
        mdf["y"], mdf["m"] = mdf.index.year, mdf.index.month
        pv = mdf.pivot_table(values="r", index="y", columns="m", aggfunc="first")
        ml = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        pv.columns = [ml[m-1] for m in pv.columns]
        fig3 = go.Figure(data=go.Heatmap(z=pv.values*100, x=pv.columns, y=pv.index.astype(str),
                                         colorscale="RdYlGn", zmid=0,
                                         text=[[f"{v:.1f}" if not np.isnan(v) else "" for v in row] for row in pv.values*100],
                                         texttemplate="%{text}%"))
        fig3.update_layout(title="Monthly Returns (%)", template="plotly_dark")
        st.plotly_chart(fig3, use_container_width=True)

    # Risk events
    if "risk_events" in res and res["risk_events"]:
        st.subheader("Risk Events")
        ev_df = pd.DataFrame([{"Date": e.date, "Type": e.event_type, "Detail": e.detail} for e in res["risk_events"]])
        st.dataframe(ev_df, use_container_width=True, hide_index=True)

    # Weights
    st.subheader("최근 포트폴리오")
    wd = res["weights"].reset_index(); wd.columns = ["Ticker", "Weight"]
    wd["Weight"] = wd["Weight"].map(lambda x: f"{x*100:.2f}%")
    st.dataframe(wd, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
