"""嫄곕옒鍮꾩슜 紐⑤뜽 ???섏닔猷?+ ?щ━?쇱? + ?쒖옣異⑷꺽."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostConfig:
    commission_pct: float = 0.00005   # 0.005% (?몃룄)
    slippage_pct: float = 0.0005      # 0.05% (?몃룄)

    @property
    def one_way_cost(self) -> float:
        """?몃룄 珥?鍮꾩슜瑜?"""
        return self.commission_pct + self.slippage_pct

    @property
    def round_trip_cost(self) -> float:
        """?뺣났 珥?鍮꾩슜瑜?"""
        return self.one_way_cost * 2
