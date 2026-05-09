"""Streamlit optimize tab: Optuna optimization, WFO, and analysis."""
from __future__ import annotations

import time

import optuna
import pandas as pd
import streamlit as st

from analyze import (
    find_robust_region,
    plot_parallel_coordinates,
    plot_sensitivity,
    sensitivity_analysis,
    top_trials_df,
)
from data_backend import load_fundamentals, load_prices, load_quarterly_cache
from optimizer import (
    ALL_TICKERS,
    UNIVERSE_TICKERS,
    best_params_to_fw,
    create_objective,
)
from transaction_cost import CostConfig
from wfo import run_walk_forward, summarize_wfo

optuna.logging.set_verbosity(optuna.logging.WARNING)


def render_optimize_tab():
    st.header("Hyperparameter Optimization")

    quick = st.toggle("Quick Mode (3-5 min)", value=True)
    if quick:
        default_trials, default_train, default_test = 30, 3, 0.5
        st.caption("Quick: 30 trials, train 3Y, test 6M")
    else:
        default_trials, default_train, default_test = 200, 4, 1.0
        st.caption("Precise: 200 trials, train 4Y, test 1Y")

    col1, col2 = st.columns(2)
    with col1:
        opt_start = str(
            st.date_input(
                "Optimization Start",
                value=pd.Timestamp("2020-01-01"),
                key="opt_start",
            )
        )
        opt_end = str(
            st.date_input(
                "Optimization End",
                value=pd.Timestamp("2024-12-31"),
                key="opt_end",
            )
        )
    with col2:
        n_trials = st.slider("Trials", 10, 300, int(default_trials), key="n_trials")
        metric = st.selectbox("Objective", ["sharpe", "calmar", "sortino"], key="metric")

    incl_tf = st.checkbox("Include Trend Filter In Optimization", value=True, key="incl_tf")

    c1, c2 = st.columns(2)
    run_opt = c1.button("Run Optimization", type="primary", use_container_width=True)
    run_wfo = c2.button("Run Walk-Forward", use_container_width=True)

    if run_opt:
        _run_optimization(opt_start, opt_end, n_trials, metric, incl_tf)
    if run_wfo:
        train_yrs = int(default_train)
        test_yrs = max(1, int(default_test + 0.5))
        _run_wfo(opt_start, opt_end, n_trials, metric, train_yrs, test_yrs)

    try:
        study = optuna.load_study(study_name="quant_mf", storage="sqlite:///optuna.db")
        if study.trials:
            _show_results(study, opt_start, opt_end)
    except Exception:
        pass


def _run_optimization(start, end, n_trials, metric, include_tf=False):
    prices = load_prices(ALL_TICKERS, start, end)
    fund = load_fundamentals(UNIVERSE_TICKERS)
    qcache = load_quarterly_cache(UNIVERSE_TICKERS)

    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    study_name = "quant_mf_tf" if include_tf else "quant_mf"
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage="sqlite:///optuna.db",
        load_if_exists=True,
        pruner=pruner,
    )

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
            eta_str = f"{int(eta // 60)}m {int(eta % 60)}s"
        else:
            eta_str = "Calculating..."
        progress.progress(pct)
        status.text(
            f"Trial {completed[0]}/{n_trials} | Best: {study.best_value:.3f} | ETA: {eta_str}"
        )

    cc = CostConfig()
    study.optimize(
        create_objective(
            prices,
            fund,
            metric,
            include_trend_filter=include_tf,
            cost_config=cc,
            quarterly_cache=qcache,
        ),
        n_trials=n_trials,
        callbacks=[callback],
    )
    progress.empty()
    elapsed_total = time.time() - t_start
    status.success(
        f"Done! Best {metric}: {study.best_value:.4f} ({int(elapsed_total)}s, "
        f"{elapsed_total / n_trials:.1f}s/trial)"
    )

    bp = study.best_params
    fw = best_params_to_fw(bp)
    display = {"best_params": bp, "normalized_weights": fw}
    if "tf_enable" in bp:
        display["trend_filter"] = {
            "enable": bp["tf_enable"],
            "ma_period": bp.get("tf_ma_period", "N/A"),
            "mode": bp.get("tf_mode", "N/A"),
        }
    st.json(display)
    st.session_state["best_params"] = bp
    st.session_state["best_fw"] = fw


def _run_wfo(start, end, n_trials, metric, train_years, test_years):
    progress = st.progress(0)
    status = st.empty()
    t0 = time.time()

    def on_progress(win_idx, total_wins, trial_idx, total_trials, best_val):
        pct = ((win_idx - 1) + trial_idx / total_trials) / total_wins
        pct = min(pct, 1.0)
        progress.progress(pct)
        elapsed = time.time() - t0
        total_work = total_wins * total_trials
        done_work = (win_idx - 1) * total_trials + trial_idx
        if done_work > 1:
            eta = elapsed / done_work * (total_work - done_work)
            eta_str = f"{int(eta // 60)}m {int(eta % 60)}s"
        else:
            eta_str = "Calculating..."
        best_str = f" | Best: {best_val:.3f}" if best_val is not None else ""
        status.text(
            f"Window {win_idx}/{total_wins} | Trial {trial_idx}/{total_trials}{best_str} | ETA: {eta_str}"
        )

    results = run_walk_forward(
        full_start=start,
        full_end=end,
        train_years=train_years,
        test_years=test_years,
        n_trials=n_trials,
        metric=metric,
        on_progress=on_progress,
    )

    progress.empty()
    elapsed = time.time() - t0
    status.success(f"WFO complete! ({int(elapsed)}s)")

    if not results:
        st.warning("No WFO result. Check the selected data range.")
        return

    summary = summarize_wfo(results)
    st.subheader("Walk-Forward Results")
    st.dataframe(summary, use_container_width=True, hide_index=True)

    oos_sharpes = [r.oos_sharpe for r in results]
    avg_oos = sum(oos_sharpes) / len(oos_sharpes)
    std_oos = pd.Series(oos_sharpes).std()
    c1, c2, c3 = st.columns(3)
    c1.metric("OOS Sharpe Mean", f"{avg_oos:.3f}")
    c2.metric("OOS Sharpe Std", f"{std_oos:.3f}")
    c3.metric("IS/OOS Stability", "Good" if std_oos < 0.3 else "Watch")


def _show_results(study, start, end):
    st.divider()
    st.subheader(f"Study Results ({len(study.trials)} trials)")

    bp = study.best_params
    fw = best_params_to_fw(bp)
    c1, c2 = st.columns(2)
    c1.metric("Best Sharpe", f"{study.best_value:.4f}")
    c2.json(fw)

    if st.button("Apply Best Params To Backtest"):
        st.session_state["best_params"] = bp
        st.session_state["best_fw"] = fw
        st.success("Applied. Go back to Backtest and click Run.")

    st.subheader("Top 20 Parameter Distribution")
    fig = plot_parallel_coordinates(study, 20)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Robust Region (Top 20%)")
    st.json(find_robust_region(study))

    if st.button("Sensitivity Analysis"):
        with st.spinner("Running sensitivity analysis..."):
            prices = load_prices(ALL_TICKERS, start, end)
            fund = load_fundamentals(UNIVERSE_TICKERS)
            qcache = load_quarterly_cache(UNIVERSE_TICKERS)
            sens = sensitivity_analysis(prices, fund, bp, quarterly_cache=qcache)
        if not sens.empty:
            st.plotly_chart(plot_sensitivity(sens), use_container_width=True)
            st.dataframe(sens, use_container_width=True, hide_index=True)
