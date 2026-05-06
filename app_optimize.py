"""Streamlit Optimize 탭 — Optuna 최적화 + WFO + 분석."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from analyze import (find_robust_region, plot_parallel_coordinates,
                     plot_sensitivity, sensitivity_analysis, top_trials_df)
from data import load_fundamentals, load_prices
from optimizer import (ALL_TICKERS, UNIVERSE_TICKERS, backtest_with_params,
                       best_params_to_fw, create_objective, _normalize_weights)
from wfo import run_walk_forward, summarize_wfo

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


def render_optimize_tab():
    """Optimize 탭 렌더링."""
    st.header("Hyperparameter Optimization")

    col1, col2 = st.columns(2)
    with col1:
        opt_start = str(st.date_input("최적화 시작일", value=pd.Timestamp("2020-01-01"),
                                       key="opt_start"))
        opt_end = str(st.date_input("최적화 종료일", value=pd.Timestamp("2024-12-31"),
                                     key="opt_end"))
    with col2:
        n_trials = st.slider("Trials", 10, 200, 50, key="n_trials")
        metric = st.selectbox("목적함수", ["sharpe", "calmar", "sortino"], key="metric")

    c1, c2 = st.columns(2)
    run_opt = c1.button("Run Optimization", type="primary", use_container_width=True)
    run_wfo = c2.button("Run Walk-Forward", use_container_width=True)

    if run_opt:
        _run_optimization(opt_start, opt_end, n_trials, metric)

    if run_wfo:
        _run_wfo(opt_start, opt_end, n_trials, metric)

    # Load existing study
    try:
        study = optuna.load_study(study_name="quant_mf", storage="sqlite:///optuna.db")
        if study.trials:
            _show_results(study, opt_start, opt_end)
    except Exception:
        pass


def _run_optimization(start, end, n_trials, metric):
    """Optuna 최적화 실행."""
    prices = load_prices(ALL_TICKERS, start, end)
    fund = load_fundamentals(UNIVERSE_TICKERS)

    progress = st.progress(0, text="최적화 진행 중...")
    study = optuna.create_study(direction="maximize", study_name="quant_mf",
                                storage="sqlite:///optuna.db", load_if_exists=True)

    completed = [0]
    def callback(study, trial):
        completed[0] += 1
        pct = min(completed[0] / n_trials, 1.0)
        progress.progress(pct, text=f"Trial {completed[0]}/{n_trials} "
                          f"| Best: {study.best_value:.3f}")

    study.optimize(create_objective(prices, fund, metric),
                   n_trials=n_trials, callbacks=[callback])
    progress.empty()

    st.success(f"완료! Best {metric}: {study.best_value:.4f}")
    bp = study.best_params
    fw = best_params_to_fw(bp)
    st.json({"best_params": bp, "normalized_weights": fw})

    # session state에 best params 저장 (슬라이더 반영용)
    st.session_state["best_params"] = bp
    st.session_state["best_fw"] = fw


def _run_wfo(start, end, n_trials, metric):
    """Walk-Forward Optimization 실행."""
    with st.spinner("Walk-Forward 실행 중... (수 분 소요)"):
        results = run_walk_forward(
            full_start=start, full_end=end,
            train_years=4, test_years=1, n_trials=n_trials, metric=metric)

    if not results:
        st.warning("WFO 결과 없음 (데이터 기간 부족)")
        return

    summary = summarize_wfo(results)
    st.subheader("Walk-Forward Results")
    st.dataframe(summary, use_container_width=True, hide_index=True)

    oos_sharpes = [r.oos_sharpe for r in results]
    avg_oos = sum(oos_sharpes) / len(oos_sharpes)
    std_oos = pd.Series(oos_sharpes).std()

    c1, c2, c3 = st.columns(3)
    c1.metric("OOS Sharpe (평균)", f"{avg_oos:.3f}")
    c2.metric("OOS Sharpe (표준편차)", f"{std_oos:.3f}")
    c3.metric("IS/OOS 일관성", "Good" if std_oos < 0.3 else "Overfit 주의")


def _show_results(study, start, end):
    """기존 study 결과 표시."""
    st.divider()
    st.subheader(f"Study Results ({len(study.trials)} trials)")

    # Best params
    bp = study.best_params
    fw = best_params_to_fw(bp)
    c1, c2 = st.columns(2)
    c1.metric("Best Sharpe", f"{study.best_value:.4f}")
    c2.json(fw)

    # Apply best params 버튼
    if st.button("Best Params → 슬라이더 반영"):
        st.session_state["best_params"] = bp
        st.session_state["best_fw"] = fw
        st.success("사이드바 슬라이더에 반영됨! Backtest 탭에서 Run 클릭")

    # Parallel coordinates
    st.subheader("Top 20 Parameter Distribution")
    fig = plot_parallel_coordinates(study, 20)
    st.plotly_chart(fig, use_container_width=True)

    # Robust region
    st.subheader("Robust Region (상위 20%)")
    robust = find_robust_region(study)
    st.json(robust)

    # Sensitivity
    if st.button("Sensitivity Analysis 실행"):
        with st.spinner("민감도 분석 중..."):
            prices = load_prices(ALL_TICKERS, start, end)
            fund = load_fundamentals(UNIVERSE_TICKERS)
            sens = sensitivity_analysis(prices, fund, bp)
        if not sens.empty:
            st.plotly_chart(plot_sensitivity(sens), use_container_width=True)
            st.dataframe(sens, use_container_width=True, hide_index=True)
