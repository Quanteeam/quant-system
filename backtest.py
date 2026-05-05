"""백테스트 엔진.

Phase 3: commission/slippage, Calmar, monthly returns, walk-forward split.
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
    calmar: float
    monthly_returns: pd.Series
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


def _sharpe(daily_ret: pd.Series, rf_annual: float = 0.0) -> float:
    excess = daily_ret - rf_annual / 252
    std = excess.std()
    return float(excess.mean() / std * np.sqrt(252)) if std > 0 else 0.0


def _calmar(cagr_val: float, max_dd: float) -> float:
    return abs(cagr_val / max_dd) if max_dd != 0 else 0.0


class BacktestEngine:
    def __init__(
        self,
        prices: pd.DataFrame,
        initial_capital: float = 100_000,
        commission_bps: float = 1.0,
        slippage_bps: float = 30.0,
    ):
        """
        Args:
            prices: MultiIndex columns (ticker, field).
            initial_capital: 초기 자본금.
            commission_bps: 편도 수수료 (bps). ~$0.005/share ≈ 1bps.
            slippage_bps: 편도 슬리피지 (bps). Small-cap 보수적 30bps.
        """
        self.prices = prices
        self.initial_capital = initial_capital
        self.cost_rate = (commission_bps + slippage_bps) / 10_000

    def run(self, weights_history: pd.DataFrame) -> BacktestResult:
        """weights_history: index=date, columns=tickers, values=target weight.

        Look-ahead 방지: t일 weight → t+1일 수익에 적용.
        Transaction cost: turnover × (commission + slippage).
        """
        close = _get_close(self.prices)

        w = weights_history.reindex(close.index, method="ffill").fillna(0)
        w = w.reindex(columns=close.columns, fill_value=0)

        daily_ret = close.pct_change()
        w_prev = w.shift(1).fillna(0)
        port_ret = (w_prev * daily_ret).sum(axis=1)

        # Transaction cost: 턴오버 × cost_rate
        turnover = w.diff().abs().sum(axis=1)
        port_ret = port_ret - turnover * self.cost_rate
        port_ret.iloc[0] = 0.0

        equity = (1 + port_ret).cumprod() * self.initial_capital
        dd = _drawdown(equity)

        # Benchmark (SPY)
        if "SPY" in close.columns:
            bench_ret = close["SPY"].pct_change().fillna(0)
            bench_curve = (1 + bench_ret).cumprod() * self.initial_capital
        else:
            bench_curve = pd.Series(self.initial_capital, index=equity.index)

        cagr_val = _cagr(equity)
        max_dd = float(dd.min())
        monthly = equity.resample("ME").last().pct_change().dropna()

        return BacktestResult(
            equity_curve=equity,
            drawdown=dd,
            total_return=float(equity.iloc[-1] / self.initial_capital - 1),
            cagr=cagr_val,
            sharpe=_sharpe(port_ret),
            max_drawdown=max_dd,
            calmar=_calmar(cagr_val, max_dd),
            monthly_returns=monthly,
            benchmark_curve=bench_curve,
        )


def walk_forward_split(
    equity: pd.Series,
    train_years: int = 5,
    test_years: int = 1,
) -> list[dict]:
    """Equity curve를 walk-forward 윈도우로 분할.

    IS/OS Sharpe 비교로 과적합 점검.
    Returns list of {window, period, train_sharpe, test_sharpe, test_cagr}.
    """
    start_year = equity.index[0].year
    end_year = equity.index[-1].year
    results = []

    for i, y in enumerate(range(start_year, end_year - train_years - test_years + 2)):
        train_eq = equity[f"{y}-01-01":f"{y + train_years - 1}-12-31"]
        test_eq = equity[f"{y + train_years}-01-01":f"{y + train_years + test_years - 1}-12-31"]

        if len(train_eq) < 100 or len(test_eq) < 50:
            continue

        results.append({
            "window": i + 1,
            "period": f"{y}–{y + train_years + test_years - 1}",
            "train_sharpe": round(_sharpe(train_eq.pct_change().dropna()), 3),
            "test_sharpe": round(_sharpe(test_eq.pct_change().dropna()), 3),
            "test_cagr": round(_cagr(test_eq), 4),
        })

    return results
