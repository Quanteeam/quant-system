"""Streamlit Robustness 탭 — 4종 시각화 + 종합 판정."""
from __future__ import annotations

import numpy as np
import streamlit as st

import optuna
from data import load_fundamentals, load_prices
from optimizer import ALL_TICKERS, UNIVERSE_TICKERS, backtest_with_params, best_params_to_fw
from robustness import (assess_robustness, compute_heatmap, plot_heatmap,
                        plot_rolling_performance, plot_topk_consistency,
                        plot_trials_distribution)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def render_robustness_tab():
    st.header("Robustness Analysis")

    try:
        study = optuna.load_study(study_name="quant_mf", storage="sqlite:///optuna.db")
    except Exception:
        st.warning("먼저 Optimize 탭에서 최적화를 실행하세요.")
        return

    if not study.trials or study.best_value is None:
        st.warning("최적화 결과가 없습니다.")
        return

    bp = study.best_params
    fw = best_params_to_fw(bp)
    st.caption(f"Best Sharpe: {study.best_value:.4f} | "
               f"Trials: {len(study.trials)}")

    # 데이터 로드
    col1, col2 = st.columns(2)
    r_start = str(col1.date_input("시작일", value="2020-01-01", key="r_start"))
    r_end = str(col2.date_input("종료일", value="2024-12-31", key="r_end"))

    if not st.button("Run Robustness Analysis", type="primary"):
        st.info("Run 클릭 시 4종 분석 실행 (1~3분 소요)")
        return

    prices = load_prices(ALL_TICKERS, r_start, r_end)
    fund = load_fundamentals(UNIVERSE_TICKERS)

    # --- 3.A. Heatmap ---
    with st.spinner("Sensitivity Heatmap 계산 중..."):
        z, x_vals, y_vals, best_x, best_y = compute_heatmap(prices, fund, bp)
        best_x_idx = int(np.argmin([abs(v - best_x) for v in x_vals]))
        best_y_idx = int(np.argmin([abs(v - best_y) for v in y_vals]))
    st.plotly_chart(plot_heatmap(z, x_vals, y_vals, best_x, best_y),
                    use_container_width=True)

    # --- 3.B. Trials Distribution ---
    st.plotly_chart(plot_trials_distribution(study), use_container_width=True)

    # --- 3.C. Rolling Performance ---
    with st.spinner("Rolling Performance 계산 중..."):
        result = backtest_with_params(prices, fund, bp.get("top_n", 20),
                                      bp.get("momentum_lb", 252),
                                      bp.get("low_vol_lb", 60),
                                      bp.get("sector_neutral", True), fw)
    if result:
        fig_roll, roll_sharpe = plot_rolling_performance(
            result.equity_curve, result.benchmark_curve)
        st.plotly_chart(fig_roll, use_container_width=True)
    else:
        roll_sharpe = None

    # --- 3.D. Top-K Consistency ---
    fig_box, topk_data = plot_topk_consistency(study, 10)
    st.plotly_chart(fig_box, use_container_width=True)

    # --- 종합 판정 ---
    st.divider()
    if roll_sharpe is not None and topk_data:
        verdict, checks = assess_robustness(
            z, best_x_idx, best_y_idx, study.best_value,
            study, roll_sharpe, topk_data)

        icons = {"Robust": "🟢", "Moderate": "🟡", "Likely Overfit": "🔴"}
        st.subheader(f"{icons.get(verdict, '❓')} 종합 판정: {verdict}")

        for name, passed in checks.items():
            label = {"heatmap_stable": "Heatmap 안정성 (주변 80%+)",
                     "within_95pct": "Trials 분포 정상 (95pct 이내)",
                     "rolling_consistent": "Rolling Sharpe 일관성 (음수 <20%)",
                     "params_tight": "Top-K 파라미터 일관성 (std <0.05)"}
            icon = "✅" if passed else "❌"
            st.write(f"{icon} {label.get(name, name)}")
