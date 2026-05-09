"""Risk Engine.

Pre-trade checks, drawdown monitoring, sanity checks.
Phase 5: backtest integration. Phase 7: live broker integration.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.config import DEFAULT_CONFIG, RiskConfig

FULL_HALT_DD = 0.20  # -20% ?????쒖뒪??halt (config ???곸닔)


@dataclass
class RiskEvent:
    date: pd.Timestamp
    event_type: str
    detail: str


class RiskEngine:
    """Pre-trade / real-time risk checks + backtest drawdown rules."""

    def __init__(self, config: RiskConfig | None = None):
        self.config = config or DEFAULT_CONFIG.risk

    # -- Pre-trade --

    def check_position_limits(self, weights: pd.Series) -> list[str]:
        """?⑥씪 醫낅ぉ max 3% 珥덇낵 寃異?"""
        lim = self.config.max_single_position
        return [f"{t}: {w:.1%} > {lim:.1%}"
                for t, w in weights.items() if abs(w) > lim]

    def check_sector_limits(
        self, weights: pd.Series, sectors: pd.Series,
    ) -> list[str]:
        """Sector ?몄텧 max 30% 珥덇낵 寃異?"""
        common = weights.index.intersection(sectors.index)
        if common.empty:
            return []
        lim = self.config.max_sector_exposure
        exp = weights[common].groupby(sectors[common]).sum()
        return [f"{s}: {e:.1%} > {lim:.1%}"
                for s, e in exp.items() if abs(e) > lim]

    # -- Sanity --

    def clean_weights(self, weights: pd.Series) -> pd.Series:
        """NaN, inf ?쒓굅."""
        return weights.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    def flag_extreme_returns(self, daily_ret: pd.DataFrame) -> pd.DataFrame:
        """?꾩씪 ?鍮?짹30% 蹂????True (?댁긽). False = ?뺤긽."""
        return daily_ret.abs() > 0.30

    # -- Backtest drawdown rules (iterative) --

    def apply_risk_to_backtest(
        self,
        weights_history: pd.DataFrame,
        prices: pd.DataFrame,
        initial_capital: float = 100_000,
        cost_rate: float = 0.0031,
    ) -> tuple[pd.DataFrame, list[RiskEvent]]:
        """Drawdown/loss rules??weights???곸슜.

        - Daily loss ??-3%: 24h ?좉퇋 二쇰Ц ?뺤?
        - DD ??-15%: weights 50% 異뺤냼
        - DD ??-20%: ???쒖뒪??halt (all ??0)

        Returns: (modified_weights, risk_events)
        """
        for f in ("adj_close", "close"):
            try:
                close = prices.xs(f, level="field", axis=1)
                break
            except KeyError:
                continue

        w = weights_history.reindex(close.index, method="ffill").fillna(0)
        w = w.reindex(columns=close.columns, fill_value=0)
        daily_ret = close.pct_change().fillna(0)

        modified = w.copy()
        events: list[RiskEvent] = []
        equity = initial_capital
        peak = initial_capital
        halt = False
        loss_halt_until: pd.Timestamp | None = None
        reduced = False

        for i in range(1, len(w)):
            date = w.index[i]

            if halt:
                modified.iloc[i] = 0
                continue
            if loss_halt_until is not None and date <= loss_halt_until:
                modified.iloc[i] = 0
                loss_halt_until = None  # reset after skip
                continue

            port_ret = (modified.iloc[i - 1] * daily_ret.iloc[i]).sum()
            equity *= (1 + port_ret)
            peak = max(peak, equity)
            dd = (equity - peak) / peak  # ??0

            # Daily loss check
            if port_ret <= -self.config.daily_loss_halt:
                loss_halt_until = date + pd.Timedelta(days=1)
                events.append(RiskEvent(date, "daily_loss_halt", f"{port_ret:.2%}"))

            # Full halt (DD ??-20%)
            if dd <= -FULL_HALT_DD:
                halt = True
                modified.iloc[i:] = 0
                events.append(RiskEvent(date, "system_halt", f"DD={dd:.2%}"))
                break

            # Reduce (DD ??-15%)
            if dd <= -self.config.drawdown_halt:
                if not reduced:
                    reduced = True
                    events.append(RiskEvent(date, "reduce_50pct", f"DD={dd:.2%}"))
                modified.iloc[i] *= 0.5

            # Alert (DD ??-10%)
            elif dd <= -self.config.drawdown_alert:
                if not any(e.event_type == "alert" for e in events):
                    events.append(RiskEvent(date, "alert", f"DD={dd:.2%}"))

        return modified, events
