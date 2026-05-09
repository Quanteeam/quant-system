"""Walk-Forward Optimization ???ㅻ쾭?쇳똿 諛⑹? 寃利?"""
from __future__ import annotations

from dataclasses import dataclass

import optuna
import pandas as pd

from data_layer.backend import load_fundamentals, load_prices, load_quarterly_cache
from research.optimizer import (ALL_TICKERS, UNIVERSE_TICKERS, backtest_with_params,
                       best_params_to_fw, create_objective)

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class WFOWindow:
    window: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    best_params: dict
    is_sharpe: float
    oos_sharpe: float
    oos_cagr: float
    oos_max_dd: float


def run_walk_forward(
    full_start: str = "2018-01-01",
    full_end: str = "2024-12-31",
    train_years: int = 4,
    test_years: int = 1,
    n_trials: int = 30,
    metric: str = "sharpe",
    on_progress=None,
) -> list[WFOWindow]:
    """Walk-forward: train N????test 1?? 1???⑥쐞 濡ㅻ쭅.

    on_progress(window_idx, total_windows, trial_idx, n_trials, best_value)
    """
    prices = load_prices(ALL_TICKERS, full_start, full_end)
    fund = load_fundamentals(UNIVERSE_TICKERS)
    qcache = load_quarterly_cache(UNIVERSE_TICKERS)

    start_year = int(full_start[:4])
    end_year = int(full_end[:4])
    results: list[WFOWindow] = []

    # ?꾩껜 ?덈룄????誘몃━ 怨꾩궛
    all_years = list(range(start_year, end_year - train_years - test_years + 2))
    total_windows = len(all_years)

    win = 0
    for y in all_years:
        train_s = f"{y}-01-01"
        train_e = f"{y + train_years - 1}-12-31"
        test_s = f"{y + train_years}-01-01"
        test_e = f"{y + train_years + test_years - 1}-12-31"
        win += 1

        # Train 援ш컙 prices ?щ씪?댁떛
        train_prices = prices.loc[train_s:train_e]
        if len(train_prices) < 252:
            continue

        # IS 理쒖쟻??
        study = optuna.create_study(direction="maximize", study_name=f"wfo_w{win}",
                                    storage=None)

        def _wfo_callback(study, trial, _w=win, _tw=total_windows):
            if on_progress:
                best_v = study.best_value if study.best_trial else None
                on_progress(_w, _tw, trial.number + 1, n_trials, best_v)

        study.optimize(create_objective(train_prices, fund, metric, quarterly_cache=qcache),
                       n_trials=n_trials, callbacks=[_wfo_callback])
        bp = study.best_params

        # IS ?깃낵
        fw = best_params_to_fw(bp)
        is_result = backtest_with_params(
            train_prices, fund, bp["top_n"], bp["momentum_lb"], bp["low_vol_lb"],
            bp["sector_neutral"], fw, quarterly_cache=qcache)
        is_sharpe = is_result.sharpe if is_result else 0.0

        # OOS ?깃낵 (best params 怨좎젙)
        test_prices = prices.loc[test_s:test_e]
        if len(test_prices) < 60:
            continue
        oos_result = backtest_with_params(
            test_prices, fund, bp["top_n"], bp["momentum_lb"], bp["low_vol_lb"],
            bp["sector_neutral"], fw, quarterly_cache=qcache)

        if oos_result is None:
            continue

        results.append(WFOWindow(
            window=win, train_start=train_s, train_end=train_e,
            test_start=test_s, test_end=test_e, best_params=bp,
            is_sharpe=is_sharpe, oos_sharpe=oos_result.sharpe,
            oos_cagr=oos_result.cagr, oos_max_dd=oos_result.max_drawdown,
        ))

    return results


def summarize_wfo(results: list[WFOWindow]) -> pd.DataFrame:
    """WFO 寃곌낵 ?붿빟 ?뚯씠釉?"""
    if not results:
        return pd.DataFrame()
    rows = []
    for r in results:
        rows.append({
            "Window": r.window,
            "Train": f"{r.train_start[:4]}~{r.train_end[:4]}",
            "Test": f"{r.test_start[:4]}~{r.test_end[:4]}",
            "IS Sharpe": round(r.is_sharpe, 3),
            "OOS Sharpe": round(r.oos_sharpe, 3),
            "OOS CAGR": f"{r.oos_cagr:.2%}",
            "OOS MaxDD": f"{r.oos_max_dd:.2%}",
            "IS/OOS Gap": f"{abs(r.is_sharpe - r.oos_sharpe):.3f}",
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    results = run_walk_forward(n_trials=10, train_years=3, test_years=1)
    print(summarize_wfo(results).to_string(index=False))
    if results:
        oos_sharpes = [r.oos_sharpe for r in results]
        print(f"\nOOS Sharpe mean: {sum(oos_sharpes)/len(oos_sharpes):.3f}")
