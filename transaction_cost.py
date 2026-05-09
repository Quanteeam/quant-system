"""거래비용 모델 — 수수료 + 슬리피지 + 시장충격."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostConfig:
    commission_pct: float = 0.00005   # 0.005% (편도)
    slippage_pct: float = 0.0005      # 0.05% (편도)

    @property
    def one_way_cost(self) -> float:
        """편도 총 비용률."""
        return self.commission_pct + self.slippage_pct

    @property
    def round_trip_cost(self) -> float:
        """왕복 총 비용률."""
        return self.one_way_cost * 2
