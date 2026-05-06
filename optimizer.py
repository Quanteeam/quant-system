"""Optuna 기반 하이퍼파라미터 최적화 (Multi-factor sleeve)."""
from __future__ import annotations

import optuna
import pandas as pd

from backtest import BacktestEngine, BacktestResult
from data import load_fundamentals, load_prices
from factors import compute_composite
from portfolio import build_multifactor_portfolio

optuna.logging.set_verbosity(optuna.logging.WARNING)

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


def backtest_with_params(
    prices: pd.DataFrame,
    fund: pd.DataFrame,
    top_n: int = 20,
    mom_lb: int = 252,
    vol_lb: int = 60,
    sec_neutral: bool = True,
    fw: dict[str, float] | None = None,
) -> BacktestResult:
    """순수 함수: 파라미터 → BacktestResult. UI/캐시 의존 없음."""
    for f in ("adj_close", "close"):
        try:
            close = prices.xs(f, level="field", axis=1)
            break
        except KeyError:
            continue

    uni = close[[t for t in UNIVERSE_TICKERS if t in close.columns]]
    rebal = uni.resample("ME").last().index

    rows = []
    for d in rebal:
        h = uni.loc[:d].dropna(axis=1, how="all")
        if len(h) < mom_lb + 5:
            continue
        vt = h.columns.tolist()
        sp = prices.loc[:d, [c for c in prices.columns if c[0] in vt]]
        sf = fund.loc[fund.index.isin(vt)]
        sc = compute_composite(sp, sf, momentum_lookback=mom_lb,
                               lowvol_lookback=vol_lb, sector_neutral=sec_neutral,
                               weights=fw)
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
        "momentum": w_mom / total,
        "quality": w_qual / total,
        "value": w_val / total,
        "size": w_size / total,
        "lowvol": w_lvol / total,
    }


def create_objective(
    prices: pd.DataFrame,
    fund: pd.DataFrame,
    metric: str = "sharpe",
):
    """Optuna objective factory."""

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
) -> optuna.Study:
    """최적화 실행. 결과는 SQLite에 영속화."""
    prices = load_prices(ALL_TICKERS, start, end)
    fund = load_fundamentals(UNIVERSE_TICKERS)

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(create_objective(prices, fund, metric), n_trials=n_trials)
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
