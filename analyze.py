"""최적화 결과 분석 — 파라미터 분포, 민감도, robust 영역."""
from __future__ import annotations

import optuna
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from optimizer import (backtest_with_params, best_params_to_fw,
                       _normalize_weights)


def top_trials_df(study: optuna.Study, n: int = 20) -> pd.DataFrame:
    """상위 N개 trial의 파라미터 + 성과."""
    trials = sorted(study.trials, key=lambda t: t.value or -999, reverse=True)[:n]
    rows = []
    for t in trials:
        row = {**t.params, "value": t.value}
        fw = best_params_to_fw(t.params)
        row.update({f"norm_{k}": v for k, v in fw.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def plot_parallel_coordinates(study: optuna.Study, n: int = 20) -> go.Figure:
    """상위 N개 파라미터 조합 parallel coordinates."""
    df = top_trials_df(study, n)
    dims = []
    for col in ["top_n", "momentum_lb", "low_vol_lb",
                 "w_momentum", "w_quality", "w_value", "w_size", "w_lowvol"]:
        if col in df.columns:
            dims.append(dict(label=col, values=df[col],
                             range=[df[col].min(), df[col].max()]))
    fig = go.Figure(data=go.Parcoords(
        line=dict(color=df["value"], colorscale="Viridis", showscale=True,
                  cmin=df["value"].min(), cmax=df["value"].max()),
        dimensions=dims,
    ))
    fig.update_layout(title=f"Top {n} Trials — Parallel Coordinates",
                      template="plotly_dark")
    return fig


def sensitivity_analysis(
    prices: pd.DataFrame, fund: pd.DataFrame, base_params: dict,
    delta: float = 0.10,
) -> pd.DataFrame:
    """각 파라미터 ±delta 변화 시 성과 변화 측정."""
    fw = best_params_to_fw(base_params)
    base = backtest_with_params(
        prices, fund, base_params["top_n"], base_params["momentum_lb"],
        base_params["low_vol_lb"], base_params["sector_neutral"], fw)
    if base is None:
        return pd.DataFrame()
    base_sharpe = base.sharpe

    continuous = ["w_momentum", "w_quality", "w_value", "w_size", "w_lowvol",
                  "momentum_lb", "low_vol_lb"]
    rows = []
    for param in continuous:
        val = base_params[param]
        for direction, mult in [("-10%", 1 - delta), ("+10%", 1 + delta)]:
            modified = {**base_params, param: val * mult}
            if param.startswith("w_"):
                mfw = best_params_to_fw(modified)
            else:
                mfw = fw
                modified[param] = int(modified[param])
            r = backtest_with_params(
                prices, fund, modified.get("top_n", base_params["top_n"]),
                int(modified.get("momentum_lb", base_params["momentum_lb"])),
                int(modified.get("low_vol_lb", base_params["low_vol_lb"])),
                modified.get("sector_neutral", base_params["sector_neutral"]),
                mfw if param.startswith("w_") else fw)
            sharpe = r.sharpe if r else 0.0
            rows.append({"param": param, "change": direction,
                          "sharpe": sharpe, "delta": sharpe - base_sharpe})
    return pd.DataFrame(rows)


def plot_sensitivity(sens_df: pd.DataFrame) -> go.Figure:
    """민감도 바차트."""
    if sens_df.empty:
        return go.Figure()
    fig = make_subplots()
    colors = {"−10%": "#ef4444", "-10%": "#ef4444", "+10%": "#22c55e"}
    for _, row in sens_df.iterrows():
        fig.add_trace(go.Bar(
            x=[f"{row['param']} {row['change']}"], y=[row["delta"]],
            marker_color=colors.get(row["change"], "#888"),
            name=f"{row['param']} {row['change']}", showlegend=False,
        ))
    fig.update_layout(title="Parameter Sensitivity (Sharpe Δ from ±10%)",
                      yaxis_title="Sharpe Change", template="plotly_dark")
    return fig


def find_robust_region(study: optuna.Study, top_pct: float = 0.2) -> dict:
    """상위 top_pct 비율 trial의 파라미터 범위 (robust 영역)."""
    trials = [t for t in study.trials if t.value is not None]
    trials.sort(key=lambda t: t.value, reverse=True)
    n = max(1, int(len(trials) * top_pct))
    top = trials[:n]
    params = list(top[0].params.keys())
    robust = {}
    for p in params:
        vals = [t.params[p] for t in top if p in t.params]
        if isinstance(vals[0], bool):
            robust[p] = max(set(vals), key=vals.count)
        else:
            robust[p] = {"min": round(min(vals), 4), "max": round(max(vals), 4),
                         "mean": round(np.mean(vals), 4)}
    return robust


if __name__ == "__main__":
    study = optuna.load_study(study_name="quant_mf", storage="sqlite:///optuna.db")
    print("=== Top 5 ===")
    print(top_trials_df(study, 5).to_string(index=False))
    print("\n=== Robust Region ===")
    for k, v in find_robust_region(study).items():
        print(f"  {k}: {v}")
