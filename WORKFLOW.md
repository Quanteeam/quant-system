# Workflow Guide

This guide keeps Codex, Claude Code, and humans aligned on the current repository direction.

## Product Direction

The project is multi-factor only.

Do not ask Claude Code to implement PEAD, hybrid 40/60 allocation, IBKR live trading, Polygon migration, or yfinance fallback unless the team explicitly changes scope.

## Daily Work Loop

1. Sync latest code.

```bash
git pull --ff-only
```

Verify: working tree is clean or only contains your intentional local edits.

2. Confirm local data config.

```env
QUANT_DATA_BACKEND=local
NASDAQ_DATA_DIR=C:\Users\womin\quant_data
```

Verify: `.env.local` exists locally and is not committed.

3. Rebuild local data only when raw Sharadar ZIPs changed.

```bash
python preprocess.py --raw-dir "C:/Users/womin/OneDrive/바탕 화면/quant_data" --out-dir "C:/Users/womin/quant_data"
```

Verify: `tickers.parquet`, `sf1.parquet`, and `sep/ticker=SPY/data.parquet` exist.

4. Run tests.

```bash
pytest
```

Verify: all tests pass before pushing.

5. Run a CLI smoke check when backtest logic changes.

```bash
python main.py backtest
```

Verify: command completes without data backend or import errors.

6. Run Streamlit when UI changes.

```bash
streamlit run app.py
```

Verify: sidebar shows only multi-factor controls and no PEAD/hybrid/live controls.

7. Commit and push.

```bash
git status --short
git add <changed-files>
git commit -m "<concise message>"
git push origin master
git push org master
```

## Claude Code Prompt Template

Send this context at the start of a new Claude Code task:

```text
Project scope is multi-factor only using local Sharadar parquet data.
Do not add or revive PEAD, hybrid allocation, yfinance, Polygon migration, or live trading.
Use CLAUDE.md as the source of truth.

Task:
[write the specific task here]

Verification required:
- pytest
- python main.py backtest if backtest/data logic changed
- streamlit run app.py if UI changed
```

## Work Split With Claude Code

Codex can own:

- Git workflow and pushes.
- Cross-file consistency checks.
- Documentation alignment.
- Final review and test runs.

Claude Code can own:

- A bounded implementation task in one area.
- Focused tests for that task.
- UI cleanup inside `app.py` and `ui/`.

Avoid assigning both agents to the same files at the same time. If both need the same file, serialize the work and pull before continuing.

## Model Guidance

Use Opus for high-risk reasoning:

- Point-in-time factor correctness.
- Backtest fill timing.
- Metric methodology.
- Optimization and robustness logic.

Use Sonnet for lower-risk implementation:

- UI edits.
- Docs.
- CLI plumbing.
- Requirements updates.
- Simple loader fixes.

## Stop Conditions

Pause and ask before continuing if:

- A task requires reintroducing PEAD, hybrid allocation, yfinance, live trading, or broker execution.
- The local data path or raw Sharadar ZIP location is unknown.
- Tests fail in an area unrelated to your changes.
- The working tree contains unrelated user edits in files you need to modify.
