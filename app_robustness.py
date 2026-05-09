"""Streamlit robustness tab for heatmap, rolling performance, and top-k checks."""
from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
import streamlit as st

from data_backend import load_fundamentals, load_prices, load_quarterly_cache
from optimizer import ALL_TICKERS, UNIVERSE_TICKERS, backtest_with_params, best_params_to_fw
from robustness import (
    assess_robustness,
    compute_heatmap,
    plot_heatmap,
    plot_rolling_performance,
    plot_topk_consistency,
    plot_trials_distribution,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def render_robustness_tab():
    st.header("Robustness Analysis")

    try:
        study = optuna.load_study(study_name="quant_mf", storage="sqlite:///optuna.db")
    except Exception:
        st.warning("Run Optimize first to create a study.")
        return

    if not study.trials or study.best_value is None:
        st.warning("No optimization result is available yet.")
        return

    bp = study.best_params
    fw = best_params_to_fw(bp)
    st.caption(f"Best Sharpe: {study.best_value:.4f} | Trials: {len(study.trials)}")

    col1, col2 = st.columns(2)
    r_start = str(col1.date_input("Robustness Start", value=pd.Timestamp("2020-01-01"), key="r_start"))
    r_end = str(col2.date_input("Robustness End", value=pd.Timestamp("2024-12-31"), key="r_end"))

    if not st.button("Run Robustness Analysis", type="primary"):
        st.info("Run the robustness analysis to generate the heatmap and consistency checks.")
        return

    prices = load_prices(ALL_TICKERS, r_start, r_end)
    fund = load_fundamentals(UNIVERSE_TICKERS)
    qcache = load_quarterly_cache(UNIVERSE_TICKERS)

    with st.spinner("Calculating sensitivity heatmap..."):
        z, x_vals, y_vals, best_x, best_y = compute_heatmap(
            prices,
            fund,
            bp,
            quarterly_cache=qcache,
        )
        best_x_idx = int(np.argmin([abs(v - best_x) for v in x_vals]))
        best_y_idx = int(np.argmin([abs(v - best_y) for v in y_vals]))
    st.plotly_chart(plot_heatmap(z, x_vals, y_vals, best_x, best_y), use_container_width=True)

    st.plotly_chart(plot_trials_distribution(study), use_container_width=True)

    with st.spinner("Calculating rolling performance..."):
        result = backtest_with_params(
            prices,
            fund,
            bp.get("top_n", 20),
            bp.get("momentum_lb", 252),
            bp.get("low_vol_lb", 60),
            bp.get("sector_neutral", True),
            fw,
            quarterly_cache=qcache,
        )
    if result:
        fig_roll, roll_sharpe = plot_rolling_performance(result.equity_curve, result.benchmark_curve)
        st.plotly_chart(fig_roll, use_container_width=True)
    else:
        roll_sharpe = None

    fig_box, topk_data = plot_topk_consistency(study, 10)
    st.plotly_chart(fig_box, use_container_width=True)

    st.divider()
    if roll_sharpe is not None and topk_data:
        verdict, checks = assess_robustness(
            z,
            best_x_idx,
            best_y_idx,
            study.best_value,
            study,
            roll_sharpe,
            topk_data,
        )

        st.subheader(f"Overall Verdict: {verdict}")
        labels = {
            "heatmap_stable": "Heatmap stability",
            "within_95pct": "Best trial not too extreme",
            "rolling_consistent": "Rolling Sharpe consistency",
            "params_tight": "Top-K parameter consistency",
        }
        for name, passed in checks.items():
            prefix = "PASS" if passed else "FAIL"
            st.write(f"{prefix} - {labels.get(name, name)}")


