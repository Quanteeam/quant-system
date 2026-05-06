"""Streamlit Optimize 탭 — Optuna 최적화 + WFO + 분석."""
from __future__ import annotations

import time

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

    # Quick / Precise 모드
    quick = st.toggle("Quick Mode (3~5분)", value=True)
    if quick:
        default_trials, default_train, default_test = 30, 3, 0.5
        st.caption("Quick: 30 trials, train 3년, test 6개월")
    else:
        default_trials, default_train, default_test = 200, 4, 1.0
        st.caption("Precise: 200 trials, train 4년, test 1년")

    col1, col2 = st.columns(2)
    with col1:
        opt_start = str(st.date_input("최적화 시작일", value=pd.Timestamp("2020-01-01"),
                                       key="opt_start"))
        opt_end = str(st.date_input("최적화 종료일", value=pd.Timestamp("2024-12-31"),
                                     key="opt_end"))
    with col2:
        n_trials = st.slider("Trials", 10, 300, int(default_trials), key="n_trials")
        metric = st.selectbox("목적함수", ["sharpe", "calmar", "sortino"], key="metric")

    c1, c2 = st.columns(2)
    run_opt = c1.button("Run Optimization", type="primary", use_container_width=True)
    run_wfo = c2.button("Run Walk-Forward", use_container_width=True)

    if run_opt:
        _run_optimization(opt_start, opt_end, n_trials, metric)
    if run_wfo:
        train_yrs = int(default_train)
        # test_years must be int for WFO window slicing
        test_yrs = max(1, int(default_test + 0.5))
        _run_wfo(opt_start, opt_end, n_trials, metric, train_yrs, test_yrs)

    # Load existing study
    try:
        study = optuna.load_study(study_name="quant_mf", storage="sqlite:///optuna.db")
        if study.trials:
            _show_results(study, opt_start, opt_end)
    except Exception:
        pass


def _run_optimization(start, end, n_trials, metric):
    """Optuna 최적화 실행 — ETA 표시."""
    prices = load_prices(ALL_TICKERS, start, end)
    fund = load_fundamentals(UNIVERSE_TICKERS)

    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    study = optuna.create_study(direction="maximize", study_name="quant_mf",
                                storage="sqlite:///optuna.db",
                                load_if_exists=True, pruner=pruner)

    progress = st.progress(0)
    status = st.empty()
    t_start = time.time()
    completed = [0]

    def callback(study, trial):
        completed[0] += 1
        pct = min(completed[0] / n_trials, 1.0)
        elapsed = time.time() - t_start
        if completed[0] > 1:
            eta = elapsed / completed[0] * (n_trials - completed[0])
            eta_str = f"{int(eta // 60)}분 {int(eta % 60)}초"
        else:
            eta_str = "계산 중..."
        progress.progress(pct)
        status.text(f"Trial {completed[0]}/{n_trials} | "
                    f"Best: {study.best_value:.3f} | "
                    f"잔여: {eta_str}")

    study.optimize(create_objective(prices, fund, metric),
                   n_trials=n_trials, callbacks=[callback])
    progress.empty()
    elapsed_total = time.time() - t_start
    status.success(f"완료! Best {metric}: {study.best_value:.4f} "
                   f"({int(elapsed_total)}초, {elapsed_total/n_trials:.1f}초/trial)")

    bp = study.best_params
    fw = best_params_to_fw(bp)
    st.json({"best_params": bp, "normalized_weights": fw})
    st.session_state["best_params"] = bp
    st.session_state["best_fw"] = fw


def _run_wfo(start, end, n_trials, metric, train_years, test_years):
    """Walk-Forward — ETA 표시."""
    status = st.empty()
    status.info("Walk-Forward 실행 중...")
    t0 = time.time()

    results = run_walk_forward(
        full_start=start, full_end=end,
        train_years=train_years, test_years=test_years,
        n_trials=n_trials, metric=metric)

    elapsed = time.time() - t0
    status.success(f"WFO 완료! ({int(elapsed)}초)")

    if not results:
        st.warning("결과 없음 (데이터 기간 부족)")
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

    bp = study.best_params
    fw = best_params_to_fw(bp)
    c1, c2 = st.columns(2)
    c1.metric("Best Sharpe", f"{study.best_value:.4f}")
    c2.json(fw)

    if st.button("Best Params → 슬라이더 반영"):
        st.session_state["best_params"] = bp
        st.session_state["best_fw"] = fw
        st.success("사이드바에 반영됨! Backtest 탭에서 Run 클릭")

    st.subheader("Top 20 Parameter Distribution")
    fig = plot_parallel_coordinates(study, 20)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Robust Region (상위 20%)")
    st.json(find_robust_region(study))

    if st.button("Sensitivity Analysis"):
        with st.spinner("민감도 분석 중..."):
            prices = load_prices(ALL_TICKERS, start, end)
            fund = load_fundamentals(UNIVERSE_TICKERS)
            sens = sensitivity_analysis(prices, fund, bp)
        if not sens.empty:
            st.plotly_chart(plot_sensitivity(sens), use_container_width=True)
            st.dataframe(sens, use_container_width=True, hide_index=True)
