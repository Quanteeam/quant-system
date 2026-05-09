"""CLI entrypoint.

Usage:
    python main.py backtest
    python main.py ui
    python main.py kill
"""
from __future__ import annotations

import subprocess
import sys


def cmd_backtest() -> None:
    import pandas as pd
    from backtest import BacktestEngine
    from config import DEFAULT_CONFIG
    from data_backend import load_prices
    from factors import compute_momentum
    from portfolio import build_multifactor_portfolio

    cfg = DEFAULT_CONFIG

    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "SPY"]
    print(f"Loading prices {cfg.backtest.start_date} ~ {cfg.backtest.end_date} ...")
    prices = load_prices(tickers, cfg.backtest.start_date, cfg.backtest.end_date)

    scores = compute_momentum(prices)
    weights = build_multifactor_portfolio(scores, top_n=cfg.portfolio.mf_num_stocks)

    try:
        close = prices.xs("adj_close", level="field", axis=1)
    except KeyError:
        close = prices.xs("close", level="field", axis=1)
    weights_df = pd.DataFrame([weights], index=[close.index[-1]])
    weights_df = weights_df.reindex(close.index, method="ffill").fillna(0)

    engine = BacktestEngine(prices, initial_capital=cfg.backtest.initial_capital)
    result = engine.run(weights_df)

    print(f"CAGR:         {result.cagr:.2%}")
    print(f"Sharpe:       {result.sharpe:.2f}")
    print(f"Max Drawdown: {result.max_drawdown:.2%}")
    print(f"Total Return: {result.total_return:.2%}")


def cmd_ui() -> None:
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "app.py"],
        check=True,
    )


def cmd_kill() -> None:
    from pathlib import Path

    kill_file = Path("KILL_SWITCH")
    kill_file.touch()
    print("KILL SWITCH ACTIVATED")
    print("  - KILL_SWITCH file created")
    print("  - All live processes should halt when broker integration is enabled")
    print("  - To resume: remove the KILL_SWITCH file")


COMMANDS = {"backtest": cmd_backtest, "ui": cmd_ui, "kill": cmd_kill}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(f"Usage: python main.py [{' | '.join(COMMANDS)}]")
        sys.exit(1)
    COMMANDS[sys.argv[1]]()


if __name__ == "__main__":
    main()


