"""Quant trading system configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


def _load_local_env() -> None:
    """Load repo-local env file for developer-specific paths."""
    env_path = Path(__file__).resolve().parents[1] / ".env.local"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip().lstrip("\ufeff")
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_local_env()


@dataclass
class UniverseConfig:
    indices: list[str] = field(default_factory=lambda: ["RUSSELL_2000", "RUSSELL_MIDCAP"])
    min_price_usd: float = 5.0
    min_adv_usd: float = 5_000_000
    exclude_otc: bool = True
    exclude_adr: bool = True


@dataclass
class FactorConfig:
    size_weight: float = 0.20
    value_weight: float = 0.20
    momentum_weight: float = 0.20
    quality_weight: float = 0.20
    lowvol_weight: float = 0.20
    momentum_lookback_months: int = 12
    momentum_skip_months: int = 1
    lowvol_lookback_days: int = 60
    sector_neutral: bool = True

    def __post_init__(self):
        total = (
            self.size_weight
            + self.value_weight
            + self.momentum_weight
            + self.quality_weight
            + self.lowvol_weight
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Factor weights must sum to 1.0, got {total}")


@dataclass
class PEADConfig:
    sue_threshold: float = 1.5
    entry_delay_days: int = 1
    max_holding_days: int = 45
    pre_next_earnings_exit_days: int = 3
    stop_loss: float = -0.10
    quality_min_percentile: float = 0.50
    value_min_percentile: float = 0.50


@dataclass
class PortfolioConfig:
    multifactor_allocation: float = 0.40
    event_allocation: float = 0.60
    mf_num_stocks: int = 20
    mf_rebalance_freq: Literal["monthly", "quarterly"] = "monthly"
    mf_weighting: Literal["equal", "inverse_vol"] = "equal"
    event_max_stocks: int = 40
    event_position_size: float = 0.015
    portfolio_vol_target: float = 0.15

    def __post_init__(self):
        total = self.multifactor_allocation + self.event_allocation
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Allocations must sum to 1.0, got {total}")


@dataclass
class RiskConfig:
    max_single_position: float = 0.03
    max_sector_exposure: float = 0.30
    max_daily_var_95: float = 0.02
    drawdown_alert: float = 0.10
    drawdown_halt: float = 0.15
    daily_loss_halt: float = 0.03


@dataclass
class ExecutionConfig:
    broker: str = "IBKR"
    algo: str = "Adaptive"
    allow_market_orders: bool = False
    use_paper: bool = True


@dataclass
class BacktestConfig:
    start_date: str = "2014-01-01"
    end_date: str = "2024-12-31"
    initial_capital: float = 100_000.0
    commission_per_share: float = 0.005
    slippage_bps: float = 30
    benchmark: str = "SPY"
    train_years: int = 5
    test_years: int = 1


@dataclass
class DataConfig:
    backend: Literal["yfinance", "local"] = field(
        default_factory=lambda: os.getenv("QUANT_DATA_BACKEND", "yfinance")
    )
    local_data_dir: str = field(
        default_factory=lambda: os.getenv(
            "NASDAQ_DATA_DIR",
            str(Path.home() / "nasdaq_data" / "processed"),
        )
    )


@dataclass
class Config:
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    factors: FactorConfig = field(default_factory=FactorConfig)
    pead: PEADConfig = field(default_factory=PEADConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    data: DataConfig = field(default_factory=DataConfig)


DEFAULT_CONFIG = Config()


if __name__ == "__main__":
    cfg = DEFAULT_CONFIG
    print(f"Multi-factor allocation: {cfg.portfolio.multifactor_allocation}")
    print(f"Event allocation:        {cfg.portfolio.event_allocation}")
    print(f"SUE threshold:           {cfg.pead.sue_threshold}")
    print(f"Backtest period:         {cfg.backtest.start_date} ~ {cfg.backtest.end_date}")
    print(f"Data backend:            {cfg.data.backend}")
    print(f"Local data dir:          {cfg.data.local_data_dir}")
    print("Config OK.")
