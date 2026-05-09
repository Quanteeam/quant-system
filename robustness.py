"""Robustness ?쒓컖?????뚮씪誘명꽣 誘쇨컧?? 遺꾪룷, 濡ㅻ쭅 ?깃낵, ?쇨???泥댄겕."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import optuna
from backtest import _cagr, _sharpe, _drawdown
from optimizer import backtest_with_params, best_params_to_fw, _normalize_weights


# --- 3.A. Parameter Sensitivity Heatmap ---

def compute_heatmap(
    prices: pd.DataFrame, fund: pd.DataFrame, base_params: dict,
    x_range: tuple = (0.10, 0.50, 0.05), y_range: tuple = (0.10, 0.40, 0.05),
    quarterly_cache: dict | None = None,
) -> tuple[np.ndarray, list[float], list[float], float, float]:
    """Momentum(X) 횞 Quality(Y) Sharpe heatmap 怨꾩궛."""
    x_vals = list(np.arange(x_range[0], x_range[1] + 0.001, x_range[2]))
    y_vals = list(np.arange(y_range[0], y_range[1] + 0.001, y_range[2]))
    z = np.zeros((len(y_vals), len(x_vals)))

    best_fw = best_params_to_fw(base_params)
    best_x, best_y = best_fw["momentum"], best_fw["quality"]

    for i, wq in enumerate(y_vals):
        for j, wm in enumerate(x_vals):
            fw = _normalize_weights(wm, wq,
                                    base_params.get("w_value", 0.15),
                                    base_params.get("w_size", 0.10),
                                    base_params.get("w_lowvol", 0.10))
            r = backtest_with_params(
                prices, fund, base_params.get("top_n", 20),
                base_params.get("momentum_lb", 252),
                base_params.get("low_vol_lb", 60),
                base_params.get("sector_neutral", True), fw, quarterly_cache=quarterly_cache)
            z[i, j] = r.sharpe if r else 0.0

    return z, x_vals, y_vals, best_x, best_y


def plot_heatmap(z, x_vals, y_vals, best_x, best_y) -> go.Figure:
    fig = go.Figure(data=go.Heatmap(
        z=z, x=[f"{v:.2f}" for v in x_vals], y=[f"{v:.2f}" for v in y_vals],
        colorscale="Viridis", colorbar_title="Sharpe",
    ))
    # Best ?꾩튂 留덉빱
    fig.add_trace(go.Scatter(
        x=[f"{best_x:.2f}"], y=[f"{best_y:.2f}"],
        mode="markers+text", text=["??Best"], textposition="top center",
        marker=dict(size=16, color="red", symbol="star"),
        showlegend=False,
    ))
    fig.update_layout(title="Parameter Sensitivity: Momentum 횞 Quality ??Sharpe",
                      xaxis_title="Momentum Weight", yaxis_title="Quality Weight",
                      template="plotly_dark")
    return fig


# --- 3.B. Optuna Trials Distribution ---

def plot_trials_distribution(study: optuna.Study) -> go.Figure:
    vals = [t.value for t in study.trials if t.value is not None and t.value > -100]
    if not vals:
        return go.Figure()

    mean_v = np.mean(vals)
    median_v = np.median(vals)
    p95 = np.percentile(vals, 95)
    best_v = study.best_value

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=vals, nbinsx=30, marker_color="#3b82f6", name="Trials"))

    for val, name, color, dash in [
        (mean_v, "Mean", "#22c55e", "dash"),
        (median_v, "Median", "#f59e0b", "dot"),
        (p95, "95th pct", "#a855f7", "dashdot"),
        (best_v, "Best", "#ef4444", "solid"),
    ]:
        fig.add_vline(x=val, line=dict(color=color, dash=dash, width=2),
                      annotation_text=f"{name}: {val:.3f}")

    fig.update_layout(title="Optuna Trials Sharpe Distribution",
                      xaxis_title="Sharpe", yaxis_title="Count",
                      template="plotly_dark")
    return fig


# --- 3.C. Rolling Performance ---

def plot_rolling_performance(equity: pd.Series, benchmark: pd.Series,
                             window: int = 252) -> go.Figure:
    ret = equity.pct_change().dropna()
    bench_ret = benchmark.pct_change().dropna()

    roll_sharpe = ret.rolling(window).apply(lambda x: x.mean() / x.std() * np.sqrt(252)
                                            if x.std() > 0 else 0, raw=True)
    bench_sharpe = bench_ret.rolling(window).apply(lambda x: x.mean() / x.std() * np.sqrt(252)
                                                   if x.std() > 0 else 0, raw=True)
    roll_dd = _drawdown(equity).rolling(window).min()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=["1Y Rolling Sharpe", "1Y Rolling Max DD"])

    fig.add_trace(go.Scatter(x=roll_sharpe.index, y=roll_sharpe.values,
                             name="Strategy", line=dict(color="#22c55e")), row=1, col=1)
    fig.add_trace(go.Scatter(x=bench_sharpe.index, y=bench_sharpe.values,
                             name="SPY", line=dict(color="#94a3b8", dash="dash")), row=1, col=1)
    fig.add_hline(y=0, line=dict(color="white", width=0.5), row=1, col=1)

    fig.add_trace(go.Scatter(x=roll_dd.index, y=roll_dd.values * 100,
                             name="Max DD", line=dict(color="#ef4444"),
                             fill="tozeroy", fillcolor="rgba(239,68,68,0.2)"),
                  row=2, col=1)

    fig.update_layout(template="plotly_dark", height=500)
    return fig, roll_sharpe


# --- 3.D. Top-K Parameter Consistency ---

def plot_topk_consistency(study: optuna.Study, k: int = 10) -> go.Figure:
    trials = sorted([t for t in study.trials if t.value and t.value > -100],
                    key=lambda t: t.value, reverse=True)[:k]
    if not trials:
        return go.Figure()

    factors = ["w_momentum", "w_quality", "w_value", "w_size", "w_lowvol"]
    labels = ["Momentum", "Quality", "Value", "Size", "Low Vol"]

    # Normalize weights for each trial
    data = {label: [] for label in labels}
    for t in trials:
        fw = best_params_to_fw(t.params)
        for label, key in zip(labels, ["momentum", "quality", "value", "size", "lowvol"]):
            data[label].append(fw[key])

    fig = go.Figure()
    colors = ["#22c55e", "#3b82f6", "#f59e0b", "#a855f7", "#ef4444"]
    for i, (label, vals) in enumerate(data.items()):
        fig.add_trace(go.Box(y=vals, name=label, marker_color=colors[i],
                             boxmean=True))

    fig.update_layout(title=f"Top {k} Trials ??Factor Weight Distribution",
                      yaxis_title="Normalized Weight", template="plotly_dark")
    return fig, data


# --- 醫낇빀 ?먯젙 ---

def assess_robustness(
    z_heatmap: np.ndarray, best_x_idx: int, best_y_idx: int, best_sharpe: float,
    study: optuna.Study,
    roll_sharpe: pd.Series,
    topk_data: dict[str, list[float]],
) -> tuple[str, dict[str, bool]]:
    """4媛?湲곗??쇰줈 robustness ?먯젙."""
    checks = {}

    # 1. Heatmap: best 二쇰? 5횞5 ?됯퇏 > best??80%
    pad = 2
    y_s = max(0, best_y_idx - pad)
    y_e = min(z_heatmap.shape[0], best_y_idx + pad + 1)
    x_s = max(0, best_x_idx - pad)
    x_e = min(z_heatmap.shape[1], best_x_idx + pad + 1)
    neighborhood_mean = z_heatmap[y_s:y_e, x_s:x_e].mean()
    checks["heatmap_stable"] = neighborhood_mean > best_sharpe * 0.8

    # 2. Trials: best媛 95th percentile ?대궡
    vals = [t.value for t in study.trials if t.value and t.value > -100]
    if vals:
        p95 = np.percentile(vals, 95)
        checks["within_95pct"] = best_sharpe <= p95 * 1.5  # not extreme outlier
    else:
        checks["within_95pct"] = False

    # 3. Rolling: ?뚯닔 Sharpe 湲곌컙 < 20%
    valid = roll_sharpe.dropna()
    if len(valid) > 0:
        neg_ratio = (valid < 0).sum() / len(valid)
        checks["rolling_consistent"] = neg_ratio < 0.20
    else:
        checks["rolling_consistent"] = False

    # 4. Top-K 媛以묒튂 std < 0.05
    stds = [np.std(v) for v in topk_data.values()]
    checks["params_tight"] = np.mean(stds) < 0.05

    passed = sum(checks.values())
    if passed >= 3:
        verdict = "Robust"
    elif passed >= 2:
        verdict = "Moderate"
    else:
        verdict = "Likely Overfit"

    return verdict, checks


