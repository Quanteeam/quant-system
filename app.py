"""Streamlit UI — Quant System Phase 2 (5-factor composite)."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backtest import BacktestEngine, walk_forward_split
from data import load_fundamentals, load_prices
from factors import compute_composite
from portfolio import build_multifactor_portfolio

# S&P 500 소형주 proxy universe (Phase 1 하드코딩 / Phase 6에서 교체)
UNIVERSE_TICKERS = [
    "DVA", "NWSA", "NWS", "PNR", "REG", "RHI", "SJM", "LKQ", "MHK", "IPG",
    "HII", "AIZ", "BWA", "RL", "VFC", "FRT", "UHS", "MOS", "NDSN", "SEE",
    "LNC", "AOS", "WHR", "DXC", "HSIC", "GL", "HBI", "CPB", "CAG", "TPR",
    "TAP", "IVZ", "FLS", "MAS", "LEG", "ALK", "KIM", "PVH", "NI", "LEN",
    "PHM", "NRG", "WBA", "FOXA", "FOX", "NWL", "IRM", "AIV", "BIO", "JNPR",
]
ALL_TICKERS = UNIVERSE_TICKERS + ["SPY"]


@st.cache_data(show_spinner="가격 데이터 로딩 중...")
def get_prices(start: str, end: str) -> pd.DataFrame:
    return load_prices(ALL_TICKERS, start, end)


@st.cache_data(show_spinner="펀더멘탈 데이터 로딩 중...")
def get_fundamentals() -> pd.DataFrame:
    return load_fundamentals(UNIVERSE_TICKERS)


@st.cache_data(show_spinner="백테스트 실행 중...")
def run_backtest(
    start: str, end: str, top_n: int,
    mom_lookback: int, vol_lookback: int, sector_neutral: bool,
) -> dict | None:
    prices = get_prices(start, end)
    fundamentals = get_fundamentals()

    for field in ("adj_close", "close"):
        try:
            close = prices.xs(field, level="field", axis=1)
            break
        except KeyError:
            continue

    uni_close = close[[t for t in UNIVERSE_TICKERS if t in close.columns]]
    rebal_dates = uni_close.resample("ME").last().index

    weights_rows: list[tuple] = []
    for date in rebal_dates:
        hist_close = uni_close.loc[:date].dropna(axis=1, how="all")
        if len(hist_close) < mom_lookback + 5:
            continue
        valid_tickers = hist_close.columns.tolist()
        sub_prices = prices.loc[:date, [c for c in prices.columns if c[0] in valid_tickers]]
        sub_fund = fundamentals.loc[fundamentals.index.isin(valid_tickers)]
        scores = compute_composite(
            sub_prices, sub_fund,
            momentum_lookback=mom_lookback,
            lowvol_lookback=vol_lookback,
            sector_neutral=sector_neutral,
        )
        weights = build_multifactor_portfolio(scores, top_n=top_n)
        if not weights.empty:
            weights_rows.append((date, weights))

    if not weights_rows:
        return None

    weights_history = (
        pd.DataFrame([w for _, w in weights_rows], index=[d for d, _ in weights_rows])
        .fillna(0)
        .reindex(uni_close.index, method="ffill")
        .fillna(0)
    )

    engine = BacktestEngine(prices, initial_capital=100_000,
                            commission_bps=1.0, slippage_bps=30.0)
    result = engine.run(weights_history)
    last_weights = weights_rows[-1][1]

    return {
        "equity": result.equity_curve,
        "drawdown": result.drawdown,
        "total_return": result.total_return,
        "cagr": result.cagr,
        "sharpe": result.sharpe,
        "max_drawdown": result.max_drawdown,
        "calmar": result.calmar,
        "monthly_returns": result.monthly_returns,
        "benchmark": result.benchmark_curve,
        "weights": last_weights,
    }


def _fmt(v: float, pct: bool = True) -> str:
    return f"{v * 100:.2f}%" if pct else f"{v:.2f}"


def main() -> None:
    st.set_page_config(page_title="Quant System", layout="wide")
    st.title("Quant System — 5-Factor Backtest")
    st.caption("Size · Value · Momentum · Quality · Low Vol — S&P 500 small-cap proxy")

    with st.sidebar:
        st.header("파라미터")
        start = str(st.date_input("시작일", value=pd.Timestamp("2020-01-01")))
        end = str(st.date_input("종료일", value=pd.Timestamp("2024-12-31")))
        top_n = st.slider("Top N 종목", 10, 50, 20)
        mom_lookback = st.slider("Momentum Lookback (일)", 60, 252, 252)
        vol_lookback = st.slider("Low Vol Lookback (일)", 20, 120, 60)
        sector_neutral = st.checkbox("Sector Neutral", value=True)
        run = st.button("Run backtest", type="primary", use_container_width=True)

    if not run:
        st.info("사이드바에서 파라미터 설정 후 **Run backtest**를 클릭하세요.")
        return

    result = run_backtest(start, end, top_n, mom_lookback, vol_lookback, sector_neutral)
    if result is None:
        st.error("데이터 부족. 기간을 늘리거나 lookback을 줄여주세요.")
        return

    # Metric cards
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("CAGR", _fmt(result["cagr"]))
    c2.metric("Sharpe", _fmt(result["sharpe"], pct=False))
    c3.metric("Max Drawdown", _fmt(result["max_drawdown"]))
    c4.metric("Calmar", _fmt(result["calmar"], pct=False))
    c5.metric("Total Return", _fmt(result["total_return"]))

    # Equity curve
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=result["equity"].index, y=result["equity"].values,
        name="Strategy", line=dict(color="#22c55e", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=result["benchmark"].index, y=result["benchmark"].values,
        name="SPY", line=dict(color="#94a3b8", width=1, dash="dash"),
    ))
    fig.update_layout(title="Equity Curve", xaxis_title="Date",
                      yaxis_title="Portfolio Value ($)", template="plotly_dark",
                      legend=dict(x=0.01, y=0.99))
    st.plotly_chart(fig, use_container_width=True)

    # Drawdown chart
    dd = result["drawdown"] * 100
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=dd.index, y=dd.values, fill="tozeroy",
        name="Drawdown", line=dict(color="#ef4444", width=1),
        fillcolor="rgba(239,68,68,0.2)",
    ))
    fig2.update_layout(title="Drawdown (%)", xaxis_title="Date",
                       yaxis_title="Drawdown (%)", template="plotly_dark")
    st.plotly_chart(fig2, use_container_width=True)

    # Monthly returns heatmap
    import numpy as np
    monthly = result["monthly_returns"]
    if not monthly.empty:
        mdf = monthly.to_frame("ret")
        mdf["year"] = mdf.index.year
        mdf["month"] = mdf.index.month
        pivot = mdf.pivot_table(values="ret", index="year", columns="month", aggfunc="first")
        month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                        "Jul","Aug","Sep","Oct","Nov","Dec"]
        pivot.columns = [month_labels[m - 1] for m in pivot.columns]
        fig3 = go.Figure(data=go.Heatmap(
            z=pivot.values * 100, x=pivot.columns, y=pivot.index.astype(str),
            colorscale="RdYlGn", zmid=0,
            text=[[f"{v:.1f}" if not np.isnan(v) else "" for v in row]
                  for row in pivot.values * 100],
            texttemplate="%{text}%",
        ))
        fig3.update_layout(title="Monthly Returns (%)", template="plotly_dark")
        st.plotly_chart(fig3, use_container_width=True)

    # Walk-forward validation
    wf = walk_forward_split(result["equity"])
    if wf:
        st.subheader("Walk-Forward Validation")
        st.dataframe(pd.DataFrame(wf), use_container_width=True, hide_index=True)

    # Weight table
    st.subheader("최근 포트폴리오 구성")
    w_df = result["weights"].reset_index()
    w_df.columns = ["Ticker", "Weight"]
    w_df["Weight"] = w_df["Weight"].map(lambda x: f"{x * 100:.2f}%")
    st.dataframe(w_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
