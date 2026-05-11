# Quant System - Multi-Factor Only

This repository is currently a local Sharadar data based multi-factor backtesting system.
Keep the active product scope narrow: monthly multi-factor equity selection, research, and Streamlit/CLI backtests.

## Current Scope

- Strategy: multi-factor only.
- Data: local Sharadar parquet bundle by default.
- Benchmarks: SPY and QQQ must come from the local data bundle.
- UI tabs: Backtest, Optimize, Robustness.
- Execution: research/backtest only. No live broker execution in the current scope.

## Out Of Scope Unless Explicitly Requested

Do not add or revive these features without a fresh user request:

- PEAD event sleeve.
- Hybrid 40/60 or 60/40 allocation.
- Earnings surprise controls in the UI.
- IBKR live trading, broker order routing, kill switch, or paper trading flows.
- Polygon migration.
- yfinance fallback.

Some legacy compatibility code may still exist. Treat it as dormant unless the user asks for cleanup.

## Directory Map

```text
quant-system/
├── core/                 # shared config and universe construction
├── data_layer/           # local parquet and Sharadar data backends
├── backtesting/          # common backtest engine, costs, trend filter
├── strategies/           # strategy registry and strategy implementations
├── strategies/multifactor/
│   ├── factors.py        # factor calculations
│   ├── portfolio.py      # multi-factor portfolio construction
│   └── strategy.py       # strategy adapter
├── research/             # optimization, robustness, walk-forward helpers
├── trading/              # legacy/inactive execution-related modules
├── ui/                   # Streamlit tab renderers
├── tests/                # pytest tests
├── preprocess.py         # raw Sharadar ZIP to local parquet bundle
├── app.py                # Streamlit entrypoint
└── main.py               # CLI entrypoint
```

Root files such as `data.py`, `backtest.py`, `factors.py`, and `portfolio.py` are compatibility wrappers. Prefer package paths for new code.

## Data Configuration

Local data is the default path.

```env
QUANT_DATA_BACKEND=local
NASDAQ_DATA_DIR=C:\Users\womin\quant_data
```

Optional direct Sharadar API access may use:

```env
QUANT_DATA_BACKEND=sharadar
NASDAQ_DATA_LINK_API_KEY=your_key
```

Do not use `yfinance`. Do not add personal local paths or API keys to tracked files.

## Preprocessing Contract

`preprocess.py` reads raw Sharadar ZIP exports and writes:

- `tickers.parquet`
- `sep/ticker=<TICKER>/data.parquet`
- `sf1.parquet`

SPY and QQQ are read from SFP and written into the same `sep/` layout as price data. If their data starts later than the requested backtest window, warn the user instead of silently inventing data.

## Strategy Rules

- New strategy code goes under `strategies/{strategy_name}/`.
- Register selectable strategies in `strategies/registry.py`.
- Shared backtest mechanics belong in `backtesting/`.
- Multi-factor-specific scoring and portfolio logic stays in `strategies/multifactor/`.
- Add new third-party packages to `requirements.txt`.

## Active Multi-Factor Behavior

- Monthly rebalance.
- Equal-weight selected portfolio.
- Default universe filtering uses price, ADV, exchange scope, and metadata.
- Factors: momentum, quality, value, size, and low volatility where data is available.
- Factor weights must sum to 1.0 in the UI.
- Benchmarks are comparison curves only, not data fallbacks.

## Verification

Run focused checks after code changes:

```bash
pytest
python main.py backtest
```

For UI-affecting changes, also run:

```bash
streamlit run app.py
```

Success means tests pass, the CLI smoke backtest runs, and the Streamlit UI exposes only multi-factor controls.

## Model Guidance For Claude Code

Use Opus for:

- Factor math and point-in-time data alignment.
- Backtest fill timing, metrics, and bias checks.
- Optimization or walk-forward methodology.

Use Sonnet for:

- UI edits.
- CLI plumbing.
- Documentation.
- Requirements and small test updates.
- Straightforward local data loading fixes.

Before making changes, confirm the task still fits the active multi-factor-only scope.
