"""Optuna 기반 하이퍼파라미터 최적화 (Multi-factor sleeve)."""
from __future__ import annotations

import hashlib
import os
import pickle
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from backtest import BacktestEngine, BacktestResult
from data import load_fundamentals, load_prices
from data_sharadar import (
    load_fundamentals_history as load_sharadar_fundamentals_history,
    load_prices as load_sharadar_prices,
    select_fundamentals_snapshot,
)
from factors import compute_composite
from portfolio import build_multifactor_portfolio

optuna.logging.set_verbosity(optuna.logging.WARNING)
DATA_BACKEND = os.environ.get("QUANT_DATA_BACKEND", "yfinance").lower()

UNIVERSE_TICKERS = [
    "CRWD", "DDOG", "NET", "ZS", "HUBS", "PAYC", "FTNT", "SNAP",
    "MRVL", "SWKS", "MPWR", "ON", "ENTG", "MKSI",
    "ALGN", "DXCM", "HOLX", "TECH", "NBIX", "EXAS",
    "DECK", "POOL", "WSM", "DPZ", "WING", "BURL",
    "AXON", "GNRC", "TREX", "RBC", "FND", "SITE",
    "LPLA", "RGA", "EWBC", "KNSL", "WBS", "CFR",
    "TRGP", "AR", "CLF", "ATI", "GPK", "UFPI",
    "ELS", "AMH",
]
ALL_TICKERS = UNIVERSE_TICKERS + ["SPY"]

# --- Factor cache ---
_FACTOR_CACHE_DIR = Path.home() / ".cache" / "quant-system" / "factors"
_factor_mem: dict[str, pd.Series] = {}


def _factor_cache_key(
    date_str: str, mom_lb: int, vol_lb: int, sec: bool, fw_str: str, backend: str
) -> str:
    raw = f"{backend}|{date_str}|{mom_lb}|{vol_lb}|{sec}|{fw_str}"
    return hashlib.md5(raw.encode()).hexdigest()


def _get_cached_scores(key: str) -> pd.Series | None:
    if key in _factor_mem:
        return _factor_mem[key]
    path = _FACTOR_CACHE_DIR / f"{key}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            s = pickle.load(f)
        _factor_mem[key] = s
        return s
    return None


def _set_cached_scores(key: str, scores: pd.Series) -> None:
    _factor_mem[key] = scores
    _FACTOR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_FACTOR_CACHE_DIR / f"{key}.pkl", "wb") as f:
        pickle.dump(scores, f)


def backtest_with_params(
    prices: pd.DataFrame,
    fund: pd.DataFrame,
    top_n: int = 20,
    mom_lb: int = 252,
    vol_lb: int = 60,
    sec_neutral: bool = True,
    fw: dict[str, float] | None = None,
    use_cache: bool = True,
) -> BacktestResult | None:
    """순수 함수: 파라미터 → BacktestResult. 팩터 캐싱 지원."""
    for f_name in ("adj_close", "close"):
        try:
            close = prices.xs(f_name, level="field", axis=1)
            break
        except KeyError:
            continue

    uni = close[[t for t in UNIVERSE_TICKERS if t in close.columns]]
    rebal = uni.resample("ME").last().index
    fw_str = str(sorted((fw or {}).items()))

    rows = []
    for d in rebal:
        h = uni.loc[:d].dropna(axis=1, how="all")
        if len(h) < mom_lb + 5:
            continue

        cache_key = _factor_cache_key(
            str(d.date()), mom_lb, vol_lb, sec_neutral, fw_str, DATA_BACKEND
        )
        sc = _get_cached_scores(cache_key) if use_cache else None

        if sc is None:
            vt = h.columns.tolist()
            sp = prices.loc[:d, [c for c in prices.columns if c[0] in vt]]
            if DATA_BACKEND == "sharadar":
                sf = select_fundamentals_snapshot(fund, vt, d)
            else:
                sf = fund.loc[fund.index.isin(vt)]
            sc = compute_composite(sp, sf, momentum_lookback=mom_lb,
                                   lowvol_lookback=vol_lb, sector_neutral=sec_neutral,
                                   weights=fw)
            if use_cache and not sc.empty:
                _set_cached_scores(cache_key, sc)

        w = build_multifactor_portfolio(sc, top_n=top_n)
        if not w.empty:
            rows.append((d, w))

    if not rows:
        return None

    mf_wh = pd.DataFrame([w for _, w in rows], index=[d for d, _ in rows]).fillna(0)
    mf_wh = mf_wh.reindex(uni.index, method="ffill").fillna(0)
    engine = BacktestEngine(prices, commission_bps=1, slippage_bps=30)
    return engine.run(mf_wh)


def _normalize_weights(w_mom, w_qual, w_val, w_size, w_lvol) -> dict[str, float]:
    """가중치 합 = 1.0 정규화. 음수 방지."""
    total = w_mom + w_qual + w_val + w_size + w_lvol
    if total <= 0:
        return {"momentum": 0.2, "quality": 0.2, "value": 0.2, "size": 0.2, "lowvol": 0.2}
    return {
        "momentum": w_mom / total, "quality": w_qual / total,
        "value": w_val / total, "size": w_size / total, "lowvol": w_lvol / total,
    }


def create_objective(prices: pd.DataFrame, fund: pd.DataFrame, metric: str = "sharpe"):
    """Optuna objective factory with pruning support."""

    def objective(trial: optuna.Trial) -> float:
        top_n = trial.suggest_int("top_n", 20, 50, step=10)
        mom_lb = trial.suggest_int("momentum_lb", 60, 252)
        vol_lb = trial.suggest_int("low_vol_lb", 30, 180)
        sec_neutral = trial.suggest_categorical("sector_neutral", [True, False])

        w_mom = trial.suggest_float("w_momentum", 0.15, 0.45)
        w_qual = trial.suggest_float("w_quality", 0.15, 0.35)
        w_val = trial.suggest_float("w_value", 0.05, 0.25)
        w_size = trial.suggest_float("w_size", 0.05, 0.20)
        w_lvol = trial.suggest_float("w_lowvol", 0.05, 0.25)
        fw = _normalize_weights(w_mom, w_qual, w_val, w_size, w_lvol)

        result = backtest_with_params(prices, fund, top_n, mom_lb, vol_lb, sec_neutral, fw)
        if result is None:
            return -999.0

        if metric == "sharpe":
            return result.sharpe
        elif metric == "calmar":
            return result.calmar
        elif metric == "sortino":
            daily = result.equity_curve.pct_change().dropna()
            down = daily[daily < 0].std()
            return float(daily.mean() / down * (252 ** 0.5)) if down > 0 else 0.0
        return result.sharpe

    return objective


def run_optimization(
    start: str = "2020-01-01",
    end: str = "2024-12-31",
    n_trials: int = 50,
    metric: str = "sharpe",
    storage: str | None = "sqlite:///optuna.db",
    study_name: str = "quant_mf",
    n_jobs: int = 1,
) -> optuna.Study:
    """최적화 실행. MedianPruner + JournalStorage 지원."""
    if DATA_BACKEND == "sharadar":
        prices = load_sharadar_prices(ALL_TICKERS, start, end)
        fund = load_sharadar_fundamentals_history(UNIVERSE_TICKERS)
    else:
        prices = load_prices(ALL_TICKERS, start, end)
        fund = load_fundamentals(UNIVERSE_TICKERS)

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=10, n_warmup_steps=5)

    study = optuna.create_study(
        direction="maximize", study_name=study_name,
        storage=storage, load_if_exists=True, pruner=pruner,
    )
    study.optimize(create_objective(prices, fund, metric),
                   n_trials=n_trials, n_jobs=n_jobs)
    return study


def best_params_to_fw(params: dict) -> dict[str, float]:
    """study.best_params → factor weights dict."""
    return _normalize_weights(
        params["w_momentum"], params["w_quality"], params["w_value"],
        params["w_size"], params["w_lowvol"],
    )


if __name__ == "__main__":
    study = run_optimization(n_trials=10, storage=None)
    print(f"Best {study.best_value:.3f}")
    print(f"Params: {study.best_params}")
