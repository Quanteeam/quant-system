"""Streamlit backtest rendering helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _fmt_num(value: float) -> str:
    return f"{value:.2f}"


def render_backtest(
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
    run_backtest_fn,
    cost_cfg=None,
    dynamic_universe=False,
    exchange_scope="NASDAQ",
    min_price=5.0,
    min_adv_usd=5_000_000.0,
):
    if not run:
        st.info("Set parameters and click Run.")
        return

    factor_weights = {
        "momentum": w_mom,
        "quality": w_qual,
        "value": w_val,
        "size": w_size,
        "lowvol": w_lvol,
    }
    cost_kw = {}
    if cost_cfg is not None:
        cost_kw = {"comm_pct": cost_cfg.commission_pct, "slip_pct": cost_cfg.slippage_pct}

    res = run_backtest_fn(
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
        fw=factor_weights,
        dynamic_universe=dynamic_universe,
        exchange_scope=exchange_scope,
        min_price=min_price,
        min_adv_usd=min_adv_usd,
        tf_cfg=tf_cfg,
        **cost_kw,
    )
    if not res:
        st.error("Backtest could not run with the current data and filters.")
        return

    run_info = res.get("run_info", {})
    info_cols = st.columns(5)
    info_cols[0].metric("Model", run_info.get("strategy_label", strategy_key))
    info_cols[1].metric("Eval Window", f"{pd.Timestamp(start).date()} -> {pd.Timestamp(end).date()}")
    info_cols[2].metric("Load Start", str(pd.Timestamp(run_info.get("load_start", start)).date()))
    info_cols[3].metric("Universe Mode", "Dynamic" if run_info.get("dynamic_universe") else "Legacy")
    info_cols[4].metric("Candidates", f"{run_info.get('candidate_count', 0):,}")
    missing_benchmarks = run_info.get("missing_benchmarks", [])
    benchmark_note = f"Benchmarks loaded: {', '.join(run_info.get('benchmarks', [])) or 'None'}"
    if missing_benchmarks:
        benchmark_note += f" | Missing benchmark data: {', '.join(missing_benchmarks)}"
    st.caption(
        f"Warmup trading days before evaluation: {run_info.get('warmup_days', 0)} | "
        f"Min history requirement: {run_info.get('min_history_days', 0)} days | "
        f"{benchmark_note}"
    )

    baselines = {"Multi-factor": res["mf"]}
    if "event" in res:
        baselines["PEAD Only"] = res["event"]
    if "hybrid" in res:
        baselines["Hybrid 40/60"] = res["hybrid"]
    if "mf_nofilter" in res:
        baselines["MF (No Filter)"] = res["mf_nofilter"]

    metric_cols = st.columns(len(baselines))
    for col, (name, result) in zip(metric_cols, baselines.items()):
        col.subheader(name)
        col.metric("CAGR", _fmt_pct(result["cagr"]))
        col.metric("Sharpe", _fmt_num(result["sharpe"]))
        col.metric("Max DD", _fmt_pct(result["max_drawdown"]))
        col.metric("Calmar", _fmt_num(result["calmar"]))
        col.metric("Turnover / Yr", _fmt_num(result.get("annual_turnover", 0.0)))
        col.metric("Avg Hold (days)", _fmt_num(result.get("average_holding_days", 0.0)))
        col.metric("Invested Days", _fmt_pct(result.get("invested_ratio", 0.0)))
        monthly_changes = result.get("monthly_entries", pd.Series(dtype=float)).add(
            result.get("monthly_exits", pd.Series(dtype=float)), fill_value=0.0
        )
        avg_changes = float(monthly_changes.mean()) if not monthly_changes.empty else 0.0
        col.metric("Avg Monthly Changes", _fmt_num(avg_changes))

    active = res.get("hybrid", res["mf"])

    equity_fig = go.Figure()
    strategy_colors = {
        "Multi-factor": "#22c55e",
        "PEAD Only": "#3b82f6",
        "Hybrid 40/60": "#f59e0b",
        "MF (No Filter)": "#94a3b8",
    }
    for name, result in baselines.items():
        eq = result["equity"]
        equity_fig.add_trace(
            go.Scatter(x=eq.index, y=eq.values, name=name, line=dict(color=strategy_colors.get(name, "#ffffff")))
        )

    benchmark_colors = {"SPY": "#64748b", "QQQ": "#c026d3"}
    for bench, curve in active.get("benchmark_curves", {}).items():
        equity_fig.add_trace(
            go.Scatter(
                x=curve.index,
                y=curve.values,
                name=bench,
                line=dict(color=benchmark_colors.get(bench, "#64748b"), dash="dash"),
            )
        )
    equity_fig.update_layout(title="Equity Curves", template="plotly_dark")
    st.plotly_chart(equity_fig, use_container_width=True)

    dd = active["drawdown"] * 100
    dd_fig = go.Figure()
    dd_fig.add_trace(
        go.Scatter(
            x=dd.index,
            y=dd.values,
            fill="tozeroy",
            name="Drawdown",
            line=dict(color="#ef4444"),
            fillcolor="rgba(239,68,68,0.2)",
        )
    )
    dd_fig.update_layout(title="Drawdown (%)", template="plotly_dark")
    st.plotly_chart(dd_fig, use_container_width=True)

    monthly = active["monthly"]
    if not monthly.empty:
        monthly_df = monthly.to_frame("r")
        monthly_df["year"] = monthly_df.index.year
        monthly_df["month"] = monthly_df.index.month
        pivot = monthly_df.pivot_table(values="r", index="year", columns="month", aggfunc="first")
        month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        pivot.columns = [month_labels[m - 1] for m in pivot.columns]
        heatmap = go.Figure(
            data=go.Heatmap(
                z=pivot.values * 100,
                x=pivot.columns,
                y=pivot.index.astype(str),
                colorscale="RdYlGn",
                zmid=0,
                text=[[f"{v:.1f}" if not np.isnan(v) else "" for v in row] for row in pivot.values * 100],
                texttemplate="%{text}%",
            )
        )
        heatmap.update_layout(title="Monthly Returns (%)", template="plotly_dark")
        st.plotly_chart(heatmap, use_container_width=True)

    bench_stats = active.get("benchmark_stats", {})
    if bench_stats:
        st.subheader("Benchmark Comparison")
        bench_df = pd.DataFrame(bench_stats).T.rename_axis("Benchmark").reset_index()
        for col in ["total_return", "cagr", "max_drawdown", "calmar"]:
            if col in bench_df.columns:
                bench_df[col] = bench_df[col].map(_fmt_pct if col != "calmar" else _fmt_num)
        if "sharpe" in bench_df.columns:
            bench_df["sharpe"] = bench_df["sharpe"].map(_fmt_num)
        st.dataframe(bench_df, use_container_width=True, hide_index=True)

    diag_col1, diag_col2 = st.columns(2)
    monthly_turnover = active.get("monthly_turnover", pd.Series(dtype=float))
    if not monthly_turnover.empty:
        turn_fig = go.Figure(
            data=[go.Bar(x=monthly_turnover.index, y=monthly_turnover.values, marker_color="#f59e0b")]
        )
        turn_fig.update_layout(title="Monthly Turnover", template="plotly_dark")
        diag_col1.plotly_chart(turn_fig, use_container_width=True)

    monthly_entries = active.get("monthly_entries", pd.Series(dtype=float))
    monthly_exits = active.get("monthly_exits", pd.Series(dtype=float))
    if not monthly_entries.empty or not monthly_exits.empty:
        change_fig = go.Figure()
        if not monthly_entries.empty:
            change_fig.add_trace(go.Bar(x=monthly_entries.index, y=monthly_entries.values, name="Entries", marker_color="#22c55e"))
        if not monthly_exits.empty:
            change_fig.add_trace(go.Bar(x=monthly_exits.index, y=monthly_exits.values, name="Exits", marker_color="#ef4444"))
        change_fig.update_layout(title="Monthly Position Changes", barmode="group", template="plotly_dark")
        diag_col2.plotly_chart(change_fig, use_container_width=True)

    universe_history = res.get("universe_history")
    if isinstance(universe_history, pd.DataFrame) and not universe_history.empty:
        st.subheader("Dynamic Universe Diagnostics")
        uni_fig = go.Figure()
        for col, color in [("listed", "#64748b"), ("eligible", "#22c55e"), ("selected", "#f59e0b")]:
            if col in universe_history.columns:
                uni_fig.add_trace(go.Scatter(x=universe_history.index, y=universe_history[col], name=col.title(), line=dict(color=color)))
        uni_fig.update_layout(title="Universe Size Over Time", template="plotly_dark")
        st.plotly_chart(uni_fig, use_container_width=True)

    if "risk_events" in res and res["risk_events"]:
        st.subheader("Risk Events")
        risk_df = pd.DataFrame(
            [{"Date": e.date, "Type": e.event_type, "Detail": e.detail} for e in res["risk_events"]]
        )
        st.dataframe(risk_df, use_container_width=True, hide_index=True)

    st.subheader("Latest Portfolio Weights")
    weights_df = res["weights"].reset_index()
    weights_df.columns = ["Ticker", "Weight"]
    weights_df["Weight"] = weights_df["Weight"].map(lambda x: f"{x * 100:.2f}%")
    st.dataframe(weights_df, use_container_width=True, hide_index=True)
