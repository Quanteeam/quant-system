# Quant System

Local Sharadar data based multi-factor backtesting system.

The active scope is multi-factor only. PEAD event sleeves, hybrid allocation, yfinance fallback, and live broker execution are intentionally out of scope unless the team explicitly reopens them.

## Install

### pip

```bash
pip install -r requirements.txt
```

### Poetry

```bash
poetry install
```

## Run

### Streamlit UI

```bash
streamlit run app.py
# or
python main.py ui
```

### CLI Backtest

```bash
python main.py backtest
```

### Tests

```bash
pytest
```

### Check Config

```bash
python -c "from config import DEFAULT_CONFIG; print(DEFAULT_CONFIG.data.backend)"
```

## Data Backend

The default backend is `local`. Create `.env.local` in the repo root for personal paths and keys. `.env.local` is gitignored.

### Local Preprocessed Data

```env
QUANT_DATA_BACKEND=local
NASDAQ_DATA_DIR=C:\Users\womin\quant_data
```

`NASDAQ_DATA_DIR` should point to the output folder created by `preprocess.py`.

### Sharadar API

```env
QUANT_DATA_BACKEND=sharadar
NASDAQ_DATA_LINK_API_KEY=your_api_key
```

`QUANDL_API_KEY` is also supported as a legacy alias.

Do not use `yfinance`. SPY and QQQ benchmark data must come from the local Sharadar bundle.

## Preprocess Raw Sharadar ZIPs

```bash
python preprocess.py --raw-dir "C:/Users/womin/OneDrive/바탕 화면/quant_data" --out-dir "C:/Users/womin/quant_data"
```

Expected outputs:

- `tickers.parquet`
- `sep/ticker=<TICKER>/data.parquet`
- `sf1.parquet`

## Directory Structure

| Path | Purpose |
|---|---|
| `core/` | Shared config and universe construction |
| `data_layer/` | Local parquet and Sharadar data backends |
| `backtesting/` | Common backtest engine, transaction costs, trend filter |
| `strategies/` | Strategy registry and implementations |
| `strategies/multifactor/` | Active multi-factor strategy |
| `trading/` | Legacy/inactive execution-related modules |
| `research/` | Optimization, robustness, and walk-forward helpers |
| `ui/` | Streamlit tab renderers |
| `tests/` | pytest tests |

Root files such as `data.py`, `backtest.py`, `factors.py`, and `portfolio.py` remain as compatibility wrappers. New code should prefer package imports.

## Strategy Rules

New strategies belong under `strategies/{strategy_name}/` and must be registered in `strategies/registry.py` before they appear in the UI or backtest flows.

Shared backtest mechanics belong in `backtesting/`. Strategy-specific logic should stay inside its strategy folder.

When adding a package, update `requirements.txt`. Do not add local paths or personal PC dependencies to tracked files.

## Current State

- Folder structure is organized into `core`, `data_layer`, `backtesting`, `strategies`, `trading`, `research`, and `ui`.
- Legacy root modules are compatibility wrappers.
- Active strategy is `strategies/multifactor`.
- Strategy registry is available for controlled strategy selection.
- `requirements.txt` is tracked.
- Local Sharadar backend is the default.
- SPY and QQQ benchmarks are expected in the local data bundle.
- Test suite passes with `pytest`.
