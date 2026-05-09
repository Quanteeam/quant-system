"""理쒖쟻??寃곌낵 遺꾩꽍 ???뚮씪誘명꽣 遺꾪룷, 誘쇨컧?? robust ?곸뿭."""
from __future__ import annotations

import optuna
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from optimizer import (backtest_with_params, best_params_to_fw,
                       _normalize_weights)


def top_trials_df(study: optuna.Study, n: int = 20) -> pd.DataFrame:
    """?곸쐞 N媛?trial???뚮씪誘명꽣 + ?깃낵."""
    trials = sorted(study.trials, key=lambda t: t.value or -999, reverse=True)[:n]
    rows = []
    for t in trials:
        row = {**t.params, "value": t.value}
        fw = best_params_to_fw(t.params)
        row.update({f"norm_{k}": v for k, v in fw.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def plot_parallel_coordinates(study: optuna.Study, n: int = 20) -> go.Figure:
    """?곸쐞 N媛??뚮씪誘명꽣 議고빀 parallel coordinates."""
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
    fig.update_layout(title=f"Top {n} Trials ??Parallel Coordinates",
                      template="plotly_dark")
    return fig


def sensitivity_analysis(
    prices: pd.DataFrame, fund: pd.DataFrame, base_params: dict,
    delta: float = 0.10, quarterly_cache: dict | None = None,
) -> pd.DataFrame:
    """媛??뚮씪誘명꽣 짹delta 蹂?????깃낵 蹂??痢≪젙."""
    fw = best_params_to_fw(base_params)
    base = backtest_with_params(
        prices, fund, base_params["top_n"], base_params["momentum_lb"],
        base_params["low_vol_lb"], base_params["sector_neutral"], fw, quarterly_cache=quarterly_cache)
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
                mfw if param.startswith("w_") else fw, quarterly_cache=quarterly_cache)
            sharpe = r.sharpe if r else 0.0
            rows.append({"param": param, "change": direction,
                          "sharpe": sharpe, "delta": sharpe - base_sharpe})
    return pd.DataFrame(rows)


def plot_sensitivity(sens_df: pd.DataFrame) -> go.Figure:
    """誘쇨컧??諛붿감??"""
    if sens_df.empty:
        return go.Figure()
    fig = make_subplots()
    colors = {"??0%": "#ef4444", "-10%": "#ef4444", "+10%": "#22c55e"}
    for _, row in sens_df.iterrows():
        fig.add_trace(go.Bar(
            x=[f"{row['param']} {row['change']}"], y=[row["delta"]],
            marker_color=colors.get(row["change"], "#888"),
            name=f"{row['param']} {row['change']}", showlegend=False,
        ))
    fig.update_layout(title="Parameter Sensitivity (Sharpe ? from 짹10%)",
                      yaxis_title="Sharpe Change", template="plotly_dark")
    return fig


def find_robust_region(study: optuna.Study, top_pct: float = 0.2) -> dict:
    """?곸쐞 top_pct 鍮꾩쑉 trial???뚮씪誘명꽣 踰붿쐞 (robust ?곸뿭)."""
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


