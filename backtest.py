"""백테스트 엔진.

Phase 1: daily rebalance, commission/slippage 무시.
weights → daily returns → equity curve → metrics
"""
from __future__ import annotations

from dataclasses import dataclass

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
    benchmark_curve: pd.Series


def _get_close(prices: pd.DataFrame) -> pd.DataFrame:
    for field in ("adj_close", "close"):
        try:
            return prices.xs(field, level="field", axis=1)
        except KeyError:
            continue
    raise KeyError("prices에 close/adj_close 컬럼 없음")


def _drawdown(equity: pd.Series) -> pd.Series:
    return (equity - equity.cummax()) / equity.cummax()


def _cagr(equity: pd.Series) -> float:
    years = len(equity) / 252
    if years == 0 or equity.iloc[0] == 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1)


def _sharpe(daily_ret: pd.Series, risk_free_annual: float = 0.0) -> float:
    excess = daily_ret - risk_free_annual / 252
    std = excess.std()
    if std == 0:
        return 0.0
    return float(excess.mean() / std * np.sqrt(252))


class BacktestEngine:
    def __init__(self, prices: pd.DataFrame, initial_capital: float = 100_000):
        """
        Args:
            prices: MultiIndex columns (ticker, field).
            initial_capital: 초기 자본금.
        """
        self.prices = prices
        self.initial_capital = initial_capital

    def run(self, weights_history: pd.DataFrame) -> BacktestResult:
        """weights_history: index=date, columns=tickers, values=target weight.

        Phase 1: daily rebalance, commission/slippage 무시.
        Look-ahead 방지: t일 weight → t+1일 수익에 적용.
        """
        close = _get_close(self.prices)

        # weights를 close 날짜에 맞춤
        w = weights_history.reindex(close.index, method="ffill").fillna(0)
        w = w.reindex(columns=close.columns, fill_value=0)

        daily_ret = close.pct_change()
        # shift(1): t일 weight를 t+1일 수익에 적용 (look-ahead 방지)
        port_ret = (w.shift(1) * daily_ret).sum(axis=1)
        port_ret.iloc[0] = 0.0

        equity = (1 + port_ret).cumprod() * self.initial_capital
        dd = _drawdown(equity)

        # Benchmark (SPY)
        if "SPY" in close.columns:
            bench_ret = close["SPY"].pct_change().fillna(0)
            bench_curve = (1 + bench_ret).cumprod() * self.initial_capital
        else:
            bench_curve = pd.Series(self.initial_capital, index=equity.index)

        return BacktestResult(
            equity_curve=equity,
            drawdown=dd,
            total_return=float(equity.iloc[-1] / self.initial_capital - 1),
            cagr=_cagr(equity),
            sharpe=_sharpe(port_ret),
            max_drawdown=float(dd.min()),
            benchmark_curve=bench_curve,
        )
