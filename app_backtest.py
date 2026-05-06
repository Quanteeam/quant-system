"""Streamlit Backtest 탭 렌더링."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backtest import BacktestEngine, _cagr, _drawdown, _sharpe, _calmar
from trend_filter import TrendFilterConfig


def _f(v, pct=True):
    return f"{v*100:.2f}%" if pct else f"{v:.2f}"


def apply_trend_filter_to_results(res, get_prices_fn, start, end, tf_cfg):
    """Trend filter를 백테스트 결과에 후처리로 적용."""
    prices = get_prices_fn(start, end)
    res["mf_nofilter"] = res["mf"]

    eq_orig = res["mf"]["equity"]
    for field in ("adj_close", "close"):
        try:
            close = prices.xs(field, level="field", axis=1)
            break
        except KeyError:
            continue

    bench = tf_cfg.benchmark
    if bench not in close.columns:
        return res

    bench_close = close[bench].reindex(eq_orig.index, method="ffill")
    ma = bench_close.rolling(tf_cfg.ma_period, min_periods=tf_cfg.ma_period).mean()
    above = bench_close >= ma

    if tf_cfg.mode == "hard":
        mult = above.astype(float)
    else:
        mult = above.astype(float) * 0.5 + 0.5
    mult = mult.fillna(1.0)

    daily_ret = eq_orig.pct_change().fillna(0)
    rf_daily = tf_cfg.rf_annual / 252
    filtered_ret = daily_ret * mult + rf_daily * (1 - mult)
    filtered_eq = (1 + filtered_ret).cumprod() * eq_orig.iloc[0]

    dd = _drawdown(filtered_eq)
    cagr_val = _cagr(filtered_eq)
    max_dd = float(dd.min())
    monthly = filtered_eq.resample("ME").last().pct_change().dropna()

    res["mf"] = {
        "equity": filtered_eq, "drawdown": dd,
        "benchmark": res["mf_nofilter"]["benchmark"],
        "cagr": cagr_val, "sharpe": _sharpe(filtered_ret),
        "max_drawdown": max_dd, "calmar": _calmar(cagr_val, max_dd),
        "total_return": float(filtered_eq.iloc[-1] / filtered_eq.iloc[0] - 1),
        "monthly": monthly,
    }
    return res


def render_backtest(run, start, end, top_n, mom_lb, vol_lb, sec_n,
                    ev_on, sue_th, max_hold, w_mom, w_qual, w_val, w_size, w_lvol,
                    tf_cfg, run_backtest_fn, get_prices_fn):
    if not run:
        st.info("파라미터 설정 후 **Run**을 클릭하세요.")
        return
    fw = {"momentum": w_mom, "quality": w_qual, "value": w_val,
          "size": w_size, "lowvol": w_lvol}
    res = run_backtest_fn(start, end, top_n, mom_lb, vol_lb, sec_n,
                          ev_on, sue_th, max_hold, fw=fw)
    if not res:
        st.error("데이터 부족")
        return

    if tf_cfg and tf_cfg.enable:
        res = apply_trend_filter_to_results(res, get_prices_fn, start, end, tf_cfg)

    active = res.get("hybrid", res["mf"])

    # Metrics
    baselines = {"Multi-factor": res["mf"]}
    if "event" in res: baselines["PEAD Only"] = res["event"]
    if "hybrid" in res: baselines["Hybrid 40/60"] = res["hybrid"]
    if "mf_nofilter" in res: baselines["MF (No Filter)"] = res["mf_nofilter"]

    cols = st.columns(len(baselines))
    for col, (name, r) in zip(cols, baselines.items()):
        col.subheader(name)
        col.metric("CAGR", _f(r["cagr"]))
        col.metric("Sharpe", _f(r["sharpe"], False))
        col.metric("Max DD", _f(r["max_drawdown"]))
        col.metric("Calmar", _f(r["calmar"], False))

    # Equity curves
    fig = go.Figure()
    colors = {"Multi-factor": "#22c55e", "PEAD Only": "#3b82f6",
              "Hybrid 40/60": "#f59e0b", "MF (No Filter)": "#94a3b8"}
    for name, r in baselines.items():
        eq = r["equity"]
        fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name=name,
                                 line=dict(color=colors.get(name, "#fff"))))
    bench = res["mf"]["benchmark"]
    fig.add_trace(go.Scatter(x=bench.index, y=bench.values, name="SPY",
                             line=dict(color="#64748b", dash="dash")))
    fig.update_layout(title="Equity Curves", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # Drawdown
    dd = active["drawdown"] * 100
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=dd.index, y=dd.values, fill="tozeroy", name="DD",
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
        fig3 = go.Figure(data=go.Heatmap(
            z=pv.values*100, x=pv.columns, y=pv.index.astype(str),
            colorscale="RdYlGn", zmid=0,
            text=[[f"{v:.1f}" if not np.isnan(v) else "" for v in row] for row in pv.values*100],
            texttemplate="%{text}%"))
        fig3.update_layout(title="Monthly Returns (%)", template="plotly_dark")
        st.plotly_chart(fig3, use_container_width=True)

    # Risk events
    if "risk_events" in res and res["risk_events"]:
        st.subheader("Risk Events")
        import pandas as pd
        ev_df = pd.DataFrame([{"Date": e.date, "Type": e.event_type, "Detail": e.detail}
                              for e in res["risk_events"]])
        st.dataframe(ev_df, use_container_width=True, hide_index=True)

    # Weights
    st.subheader("최근 포트폴리오")
    wd = res["weights"].reset_index()
    wd.columns = ["Ticker", "Weight"]
    wd["Weight"] = wd["Weight"].map(lambda x: f"{x*100:.2f}%")
    st.dataframe(wd, use_container_width=True, hide_index=True)
