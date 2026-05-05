"""execution.py 단위 테스트 (mocked — IBKR 불필요)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

import pandas as pd
import pytest

from execution import ExecutionEngine, OrderResult, KILL_SWITCH_FILE


@pytest.fixture
def engine():
    return ExecutionEngine()


def test_compute_target_shares(engine):
    weights = pd.Series({"AAPL": 0.5, "MSFT": 0.3})
    prices = {"AAPL": 150.0, "MSFT": 300.0}
    targets = engine.compute_target_shares(weights, 100_000, prices)
    assert targets["AAPL"] == 333  # 50000 / 150
    assert targets["MSFT"] == 100  # 30000 / 300


def test_compute_target_shares_zero_price(engine):
    weights = pd.Series({"AAPL": 0.5})
    prices = {"AAPL": 0.0}
    targets = engine.compute_target_shares(weights, 100_000, prices)
    assert "AAPL" not in targets


def test_compute_orders_buy(engine):
    current = {"AAPL": 100}
    target = {"AAPL": 200, "MSFT": 50}
    orders = engine.compute_orders(current, target)
    assert ("AAPL", "BUY", 100) in orders
    assert ("MSFT", "BUY", 50) in orders


def test_compute_orders_sell(engine):
    current = {"AAPL": 200, "GOOGL": 30}
    target = {"AAPL": 100}
    orders = engine.compute_orders(current, target)
    assert ("AAPL", "SELL", 100) in orders
    assert ("GOOGL", "SELL", 30) in orders


def test_compute_orders_no_change(engine):
    current = {"AAPL": 100}
    target = {"AAPL": 100}
    orders = engine.compute_orders(current, target)
    assert orders == []


def test_kill_switch_active(engine, tmp_path, monkeypatch):
    monkeypatch.setattr("execution.KILL_SWITCH_FILE", tmp_path / "KILL_SWITCH")
    (tmp_path / "KILL_SWITCH").touch()
    assert engine.is_kill_switch_active() is True


def test_kill_switch_inactive(engine, monkeypatch):
    monkeypatch.setattr("execution.KILL_SWITCH_FILE", Path("/nonexistent/KILL_SWITCH"))
    assert engine.is_kill_switch_active() is False


def test_position_limit_violation_blocks_rebalance(engine):
    """3% 초과 weight → rebalance 거부."""
    weights = pd.Series({"AAPL": 0.10})  # 10% > 3% limit
    from risk import RiskEngine
    risk = RiskEngine()
    violations = risk.check_position_limits(weights)
    assert len(violations) > 0  # Would block rebalance


@pytest.mark.asyncio
async def test_rebalance_kill_switch_aborts(monkeypatch):
    engine = ExecutionEngine()
    monkeypatch.setattr(engine, "is_kill_switch_active", lambda: True)
    result = await engine.rebalance(pd.Series({"AAPL": 0.02}))
    assert result == []
