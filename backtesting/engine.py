"""Backtest engine and evaluation utilities."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    drawdown: pd.Series
    total_return: float
    cagr: float
    sharpe: float
    max_drawdown: float
    calmar: float
    monthly_returns: pd.Series
    benchmark_curve: pd.Series
    annual_turnover: float = 0.0
    total_cost: float = 0.0
    cost_drag: float = 0.0
    benchmark_curves: dict[str, pd.Series] = field(default_factory=dict)
    benchmark_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    monthly_turnover: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    monthly_entries: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    monthly_exits: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    average_holding_days: float = 0.0
    invested_ratio: float = 0.0


def _get_close(prices: pd.DataFrame) -> pd.DataFrame:
    for field in ("adj_close", "close"):
        try:
            return prices.xs(field, level="field", axis=1)
        except KeyError:
            continue
    raise KeyError("prices must include close or adj_close columns")


def _drawdown(equity: pd.Series) -> pd.Series:
    return (equity - equity.cummax()) / equity.cummax()


def _cagr(equity: pd.Series) -> float:
    years = len(equity) / 252
    if years == 0 or equity.empty or equity.iloc[0] == 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1)


def _sharpe(daily_ret: pd.Series, rf_annual: float = 0.0) -> float:
    excess = daily_ret - rf_annual / 252
    std = excess.std()
    return float(excess.mean() / std * np.sqrt(252)) if std and std > 0 else 0.0


def _calmar(cagr_val: float, max_dd: float) -> float:
    return abs(cagr_val / max_dd) if max_dd != 0 else 0.0


def _average_holding_days(holdings: pd.DataFrame) -> float:
    durations: list[int] = []
    for ticker in holdings.columns:
        active = holdings[ticker].astype(bool)
        if active.empty:
            continue
        start = None
        for idx, value in enumerate(active.tolist()):
            if value and start is None:
                start = idx
            elif not value and start is not None:
                durations.append(idx - start)
                start = None
        if start is not None:
            durations.append(len(active) - start)
    return float(np.mean(durations)) if durations else 0.0


def _build_benchmark_curve(series: pd.Series, initial_capital: float) -> tuple[pd.Series, dict[str, float]]:
    ret = series.pct_change(fill_method=None).fillna(0.0)
    equity = (1 + ret).cumprod() * initial_capital
    dd = _drawdown(equity)
    stats = {
        "total_return": float(equity.iloc[-1] / initial_capital - 1),
        "cagr": _cagr(equity),
        "sharpe": _sharpe(ret),
        "max_drawdown": float(dd.min()),
        "calmar": _calmar(_cagr(equity), float(dd.min())),
    }
    return equity, stats


class BacktestEngine:
    def __init__(
        self,
        prices: pd.DataFrame,
        initial_capital: float = 100_000,
        commission_bps: float = 1.0,
        slippage_bps: float = 30.0,
        cost_config=None,
    ):
        self.prices = prices
        self.initial_capital = initial_capital
        if cost_config is not None:
            self.cost_rate = cost_config.one_way_cost
        else:
            self.cost_rate = (commission_bps + slippage_bps) / 10_000

    def run(self, weights_history: pd.DataFrame, eval_start: str | pd.Timestamp | None = None) -> BacktestResult:
        close = _get_close(self.prices)

        w = weights_history.reindex(close.index, method="ffill").fillna(0.0)
        w = w.reindex(columns=close.columns, fill_value=0.0)

        daily_ret = close.pct_change(fill_method=None).fillna(0.0)
        w_prev = w.shift(1).fillna(0.0)
        gross_exposure = w_prev.abs().sum(axis=1)
        raw_port_ret = (w_prev * daily_ret).sum(axis=1)

        turnover = w.diff().abs().sum(axis=1).fillna(0.0)
        port_ret = raw_port_ret - turnover * self.cost_rate
        if not port_ret.empty:
            port_ret.iloc[0] = 0.0

        full_equity = (1 + port_ret).cumprod() * self.initial_capital
        full_dd = _drawdown(full_equity)

        if eval_start is not None:
            eval_start_ts = pd.Timestamp(eval_start)
            eval_index = close.index[close.index >= eval_start_ts]
        else:
            eval_index = close.index
        if len(eval_index) == 0:
            raise ValueError("No evaluation dates available after eval_start.")

        port_ret_eval = port_ret.loc[eval_index]
        turnover_eval = turnover.loc[eval_index]
        equity = (1 + port_ret_eval).cumprod() * self.initial_capital
        dd = _drawdown(equity)
        monthly = equity.resample("ME").last().pct_change().dropna()

        monthly_turnover = turnover_eval.resample("ME").sum().fillna(0.0)

        holdings = (w_prev.abs() > 1e-12).astype(bool).loc[eval_index]
        holdings_prev = holdings.shift(1, fill_value=False).astype(bool)
        daily_entries = (holdings & ~holdings_prev).sum(axis=1)
        daily_exits = (~holdings & holdings_prev).sum(axis=1)
        monthly_entries = daily_entries.resample("ME").sum().astype(float)
        monthly_exits = daily_exits.resample("ME").sum().astype(float)
        avg_holding_days = _average_holding_days(holdings)
        invested_ratio = float((gross_exposure.loc[eval_index] > 1e-12).mean()) if len(eval_index) else 0.0

        years = len(equity) / 252
        total_turnover = float(turnover_eval.sum())
        annual_turnover = total_turnover / years if years > 0 else 0.0
        total_cost_abs = float(turnover_eval.sum() * self.cost_rate * self.initial_capital)

        port_ret_nocost = raw_port_ret.copy()
        if not port_ret_nocost.empty:
            port_ret_nocost.iloc[0] = 0.0
        port_ret_nocost_eval = port_ret_nocost.loc[eval_index]
        equity_nocost = (1 + port_ret_nocost_eval).cumprod() * self.initial_capital
        cagr_val = _cagr(equity)
        cagr_nocost = _cagr(equity_nocost)
        max_dd = float(dd.min())
        cost_drag = cagr_nocost - cagr_val

        benchmark_curves: dict[str, pd.Series] = {}
        benchmark_stats: dict[str, dict[str, float]] = {}
        for bench in ("SPY", "QQQ"):
            if bench not in close.columns:
                continue
            curve, stats = _build_benchmark_curve(close[bench].loc[eval_index], self.initial_capital)
            benchmark_curves[bench] = curve
            benchmark_stats[bench] = stats

        benchmark_curve = benchmark_curves.get(
            "SPY",
            pd.Series(self.initial_capital, index=equity.index),
        )

        return BacktestResult(
            equity_curve=equity,
            drawdown=dd,
            total_return=float(equity.iloc[-1] / self.initial_capital - 1),
            cagr=cagr_val,
            sharpe=_sharpe(port_ret_eval),
            max_drawdown=max_dd,
            calmar=_calmar(cagr_val, max_dd),
            monthly_returns=monthly,
            benchmark_curve=benchmark_curve,
            annual_turnover=annual_turnover,
            total_cost=total_cost_abs,
            cost_drag=cost_drag,
            benchmark_curves=benchmark_curves,
            benchmark_stats=benchmark_stats,
            monthly_turnover=monthly_turnover,
            monthly_entries=monthly_entries,
            monthly_exits=monthly_exits,
            average_holding_days=avg_holding_days,
            invested_ratio=invested_ratio,
        )


def walk_forward_split(
    equity: pd.Series,
    train_years: int = 5,
    test_years: int = 1,
) -> list[dict]:
    """Split an equity curve into rolling walk-forward windows."""
    start_year = equity.index[0].year
    end_year = equity.index[-1].year
    results = []

    for i, y in enumerate(range(start_year, end_year - train_years - test_years + 2)):
        train_eq = equity[f"{y}-01-01":f"{y + train_years - 1}-12-31"]
        test_eq = equity[f"{y + train_years}-01-01":f"{y + train_years + test_years - 1}-12-31"]

        if len(train_eq) < 100 or len(test_eq) < 50:
            continue

        results.append(
            {
                "window": i + 1,
                "period": f"{y}-{y + train_years + test_years - 1}",
                "train_sharpe": round(_sharpe(train_eq.pct_change().dropna()), 3),
                "test_sharpe": round(_sharpe(test_eq.pct_change().dropna()), 3),
                "test_cagr": round(_cagr(test_eq), 4),
            }
        )

    return results
