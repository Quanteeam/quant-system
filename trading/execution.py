"""IBKR ?ㅽ뻾 ?붿쭊 (Phase 7).

ib_async 湲곕컲 二쇰Ц ?ㅽ뻾, ?ъ????숆린?? kill switch ?곕룞.

?붽뎄 ?ы빆:
    - TWS ?먮뒗 IB Gateway ?ㅽ뻾 以?(paper: port 7497, live: 7496)
    - pip install ib_async
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from core.config import DEFAULT_CONFIG, ExecutionConfig, RiskConfig
from trading.risk import RiskEngine

logger = logging.getLogger(__name__)

KILL_SWITCH_FILE = Path("KILL_SWITCH")


@dataclass
class OrderResult:
    ticker: str
    side: str  # "BUY" or "SELL"
    qty: int
    fill_price: float | None = None
    status: str = "pending"  # pending, filled, cancelled, rejected


@dataclass
class ExecutionEngine:
    """IBKR 二쇰Ц ?ㅽ뻾 + ?ъ????숆린??"""

    config: ExecutionConfig = field(default_factory=lambda: DEFAULT_CONFIG.execution)
    risk_config: RiskConfig = field(default_factory=lambda: DEFAULT_CONFIG.risk)
    _ib: object = field(default=None, repr=False)

    async def connect(self) -> None:
        """TWS/Gateway ?곌껐."""
        try:
            from ib_async import IB
        except ImportError:
            raise ImportError("pip install ib_async")

        port = 7497 if self.config.use_paper else 7496
        self._ib = IB()
        await self._ib.connectAsync("127.0.0.1", port, clientId=1)
        logger.info(f"Connected to IBKR (port={port}, paper={self.config.use_paper})")

    async def disconnect(self) -> None:
        if self._ib:
            self._ib.disconnect()

    def is_kill_switch_active(self) -> bool:
        return KILL_SWITCH_FILE.exists()

    async def get_positions(self) -> dict[str, int]:
        """?꾩옱 ?ъ???議고쉶 ??{ticker: qty}."""
        positions = self._ib.positions()
        return {
            p.contract.symbol: int(p.position)
            for p in positions if p.position != 0
        }

    async def get_account_value(self) -> float:
        """怨꾩쥖 ?쒖옄??"""
        summary = self._ib.accountSummary()
        for item in summary:
            if item.tag == "NetLiquidation":
                return float(item.value)
        return 0.0

    def compute_target_shares(
        self, weights: pd.Series, account_value: float, prices: dict[str, float],
    ) -> dict[str, int]:
        """紐⑺몴 weight ??紐⑺몴 二쇱떇 ??"""
        targets: dict[str, int] = {}
        for ticker, w in weights.items():
            if ticker not in prices or prices[ticker] <= 0:
                continue
            dollar_alloc = account_value * w
            shares = int(dollar_alloc / prices[ticker])
            if shares != 0:
                targets[ticker] = shares
        return targets

    def compute_orders(
        self, current: dict[str, int], target: dict[str, int],
    ) -> list[tuple[str, str, int]]:
        """(?꾩옱 - 紐⑺몴) 李⑥씠 ??二쇰Ц 由ъ뒪??[(ticker, side, qty)]."""
        orders: list[tuple[str, str, int]] = []
        all_tickers = set(current) | set(target)
        for ticker in all_tickers:
            cur = current.get(ticker, 0)
            tgt = target.get(ticker, 0)
            diff = tgt - cur
            if diff > 0:
                orders.append((ticker, "BUY", diff))
            elif diff < 0:
                orders.append((ticker, "SELL", abs(diff)))
        return orders

    async def execute_orders(
        self, orders: list[tuple[str, str, int]],
    ) -> list[OrderResult]:
        """Adaptive algo濡?二쇰Ц ?ㅽ뻾. ?쒖옣媛 湲덉?."""
        from ib_async import Stock, LimitOrder, Order

        if self.is_kill_switch_active():
            logger.warning("KILL SWITCH active ??all orders rejected")
            return [OrderResult(t, s, q, status="rejected") for t, s, q in orders]

        results: list[OrderResult] = []
        for ticker, side, qty in orders:
            contract = Stock(ticker, "SMART", "USD")
            await self._ib.qualifyContractsAsync(contract)

            # Adaptive algo (IBKR)
            order = Order()
            order.action = side
            order.totalQuantity = qty
            order.orderType = "LMT"
            order.algoStrategy = "Adaptive"
            order.algoParams = [{"tag": "adaptivePriority", "value": "Normal"}]
            # Get current price for limit
            ticker_data = self._ib.reqMktData(contract)
            await asyncio.sleep(2)
            mid = (ticker_data.bid + ticker_data.ask) / 2 if ticker_data.bid > 0 else 0
            order.lmtPrice = round(mid * (1.001 if side == "BUY" else 0.999), 2)
            self._ib.cancelMktData(contract)

            trade = self._ib.placeOrder(contract, order)
            results.append(OrderResult(ticker, side, qty, status="pending"))
            logger.info(f"Order placed: {side} {qty} {ticker} @ {order.lmtPrice}")

        return results

    async def rebalance(self, target_weights: pd.Series) -> list[OrderResult]:
        """?꾩껜 由щ갭?곗뒪 ?뚮줈?? ?ъ???議고쉶 ??李⑥씠 怨꾩궛 ??二쇰Ц."""
        if self.is_kill_switch_active():
            logger.warning("KILL SWITCH ??rebalance aborted")
            return []

        # Pre-trade risk check
        risk = RiskEngine()
        violations = risk.check_position_limits(target_weights)
        if violations:
            logger.error(f"Position limit violations: {violations}")
            return []

        current_pos = await self.get_positions()
        account_val = await self.get_account_value()

        # Get current prices
        from ib_async import Stock
        prices: dict[str, float] = {}
        for ticker in set(target_weights.index) | set(current_pos.keys()):
            contract = Stock(ticker, "SMART", "USD")
            await self._ib.qualifyContractsAsync(contract)
            data = self._ib.reqMktData(contract)
            await asyncio.sleep(1)
            if data.last > 0:
                prices[ticker] = data.last
            elif data.close > 0:
                prices[ticker] = data.close
            self._ib.cancelMktData(contract)

        target_shares = self.compute_target_shares(target_weights, account_val, prices)
        orders = self.compute_orders(current_pos, target_shares)

        if not orders:
            logger.info("No orders needed")
            return []

        return await self.execute_orders(orders)

    async def emergency_liquidate(self) -> list[OrderResult]:
        """Kill switch: ???ъ???泥?궛."""
        current = await self.get_positions()
        orders = [(t, "SELL" if q > 0 else "BUY", abs(q)) for t, q in current.items()]
        if not orders:
            return []
        # Bypass kill switch check for liquidation
        from ib_async import Stock, MarketOrder
        results: list[OrderResult] = []
        for ticker, side, qty in orders:
            contract = Stock(ticker, "SMART", "USD")
            await self._ib.qualifyContractsAsync(contract)
            # Emergency: market order allowed for liquidation only
            order = MarketOrder(side, qty)
            self._ib.placeOrder(contract, order)
            results.append(OrderResult(ticker, side, qty, status="pending"))
            logger.warning(f"EMERGENCY: {side} {qty} {ticker} @ MKT")
        return results
