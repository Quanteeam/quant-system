"""CLI 진입점.

Usage:
    python main.py backtest    → 콘솔 백테스트 (기본 파라미터)
    python main.py ui          → streamlit run app.py
    python main.py kill        → NotImplementedError (Phase 5+)
"""
from __future__ import annotations

import subprocess
import sys


def cmd_backtest() -> None:
    import pandas as pd
    from config import DEFAULT_CONFIG
    from data import load_prices
    from factors import compute_momentum
    from portfolio import build_multifactor_portfolio
    from backtest import BacktestEngine

    cfg = DEFAULT_CONFIG
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "SPY"]
    print(f"Loading prices {cfg.backtest.start_date} ~ {cfg.backtest.end_date} ...")
    prices = load_prices(tickers, cfg.backtest.start_date, cfg.backtest.end_date)

    scores = compute_momentum(prices)
    weights = build_multifactor_portfolio(scores, top_n=cfg.portfolio.mf_num_stocks)

    close = prices.xs("close", level="field", axis=1, errors="ignore")
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
    """Emergency kill switch: 전 포지션 청산 + 신규 주문 차단."""
    from pathlib import Path
    kill_file = Path("KILL_SWITCH")
    kill_file.touch()
    print("⚠ KILL SWITCH ACTIVATED")
    print("  - KILL_SWITCH 파일 생성됨")
    print("  - 모든 시스템 halt (Phase 7에서 broker 연동)")
    print("  - 해제: KILL_SWITCH 파일 삭제")


COMMANDS = {"backtest": cmd_backtest, "ui": cmd_ui, "kill": cmd_kill}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(f"Usage: python main.py [{' | '.join(COMMANDS)}]")
        sys.exit(1)
    COMMANDS[sys.argv[1]]()


if __name__ == "__main__":
    main()
