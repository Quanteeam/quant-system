"""
Quant trading system configuration.

모든 파라미터를 이 파일에 모아둠. 알고리즘 수정 시 대부분의 변경은
여기서 일어남. 함수 로직 자체를 바꿀 때만 다른 파일을 건드린다.

사용 예:
    from config import DEFAULT_CONFIG, Config
    cfg = DEFAULT_CONFIG
    print(cfg.pead.sue_threshold)  # 1.5

    # 변형:
    custom = Config()
    custom.pead.sue_threshold = 2.0
"""
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class UniverseConfig:
    """종목 universe 필터링 기준."""
    indices: list[str] = field(
        default_factory=lambda: ["RUSSELL_2000", "RUSSELL_MIDCAP"]
    )
    min_price_usd: float = 5.0           # 페니스톡 제외
    min_adv_usd: float = 5_000_000       # 최소 20일 평균 거래대금
    exclude_otc: bool = True
    exclude_adr: bool = True


@dataclass
class FactorConfig:
    """5팩터 가중치 및 계산 파라미터.

    가중치 합은 1.0이어야 함. multi-factor sleeve에만 적용 (event sleeve는
    별도 quality/value loose filter 사용 — PEADConfig 참조).
    """
    size_weight: float = 0.20
    value_weight: float = 0.20
    momentum_weight: float = 0.20
    quality_weight: float = 0.20
    lowvol_weight: float = 0.20

    # 계산 lookback
    momentum_lookback_months: int = 12
    momentum_skip_months: int = 1        # 12-1 momentum (직전 1개월 제외)
    lowvol_lookback_days: int = 60

    # Sector neutralization (sector 내 ranking)
    sector_neutral: bool = True

    def __post_init__(self):
        total = (self.size_weight + self.value_weight + self.momentum_weight
                 + self.quality_weight + self.lowvol_weight)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Factor weights must sum to 1.0, got {total}")


@dataclass
class PEADConfig:
    """PEAD event-driven sleeve 파라미터."""
    sue_threshold: float = 1.5           # 진입 기준 SUE z-score
    entry_delay_days: int = 1            # D+1 진입 (D+0 noise 회피)
    max_holding_days: int = 45
    pre_next_earnings_exit_days: int = 3 # 다음 실적 D-3 청산
    stop_loss: float = -0.10             # 종목별 stop

    # Loose multi-factor filter (event sleeve 진입 시)
    quality_min_percentile: float = 0.50 # sector median 이상
    value_min_percentile: float = 0.50


@dataclass
class PortfolioConfig:
    """60/40 sleeve 구성."""
    multifactor_allocation: float = 0.40
    event_allocation: float = 0.60

    # Multi-factor sleeve
    mf_num_stocks: int = 20
    mf_rebalance_freq: Literal["monthly", "quarterly"] = "monthly"
    mf_weighting: Literal["equal", "inverse_vol"] = "equal"

    # Event sleeve
    event_max_stocks: int = 40
    event_position_size: float = 0.015   # 1.5% per stock

    # Portfolio-level vol target (선택적)
    portfolio_vol_target: float = 0.15

    def __post_init__(self):
        total = self.multifactor_allocation + self.event_allocation
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Allocations must sum to 1.0, got {total}")


@dataclass
class RiskConfig:
    """Pre-trade + monitoring 한도."""
    max_single_position: float = 0.03    # 단일 종목 max 3%
    max_sector_exposure: float = 0.30    # sector max 30%
    max_daily_var_95: float = 0.02       # daily VaR 95% max 2%
    drawdown_alert: float = 0.10         # -10% 알람
    drawdown_halt: float = 0.15          # -15% event sleeve 50% 축소
    daily_loss_halt: float = 0.03        # -3% 일일 손실 시 신규 주문 정지


@dataclass
class ExecutionConfig:
    """체결 설정. 라이브 거래에만 적용 (백테스트는 별도)."""
    broker: str = "IBKR"
    algo: str = "Adaptive"               # IBKR Adaptive algo
    allow_market_orders: bool = False    # 시장가 금지
    use_paper: bool = True               # paper trading 우선


@dataclass
class BacktestConfig:
    """백테스트 환경."""
    start_date: str = "2014-01-01"
    end_date: str = "2024-12-31"
    initial_capital: float = 100_000.0
    commission_per_share: float = 0.005  # IBKR commission 추정
    slippage_bps: float = 30             # small-cap 보수적 추정 (30bps)
    benchmark: str = "SPY"

    # Walk-forward
    train_years: int = 5
    test_years: int = 1


@dataclass
class Config:
    """전체 시스템 설정. 모든 sub-config의 컨테이너."""
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    factors: FactorConfig = field(default_factory=FactorConfig)
    pead: PEADConfig = field(default_factory=PEADConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)


# 기본 인스턴스 — 다른 파일에서 import해서 사용
DEFAULT_CONFIG = Config()


if __name__ == "__main__":
    # 빠른 검증: dataclass __post_init__이 정상 동작하는지
    cfg = DEFAULT_CONFIG
    print(f"Multi-factor allocation: {cfg.portfolio.multifactor_allocation}")
    print(f"Event allocation:        {cfg.portfolio.event_allocation}")
    print(f"SUE threshold:           {cfg.pead.sue_threshold}")
    print(f"Backtest period:         {cfg.backtest.start_date} ~ "
          f"{cfg.backtest.end_date}")
    print("Config OK.")
