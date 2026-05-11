"""Scope guard tests for the active multi-factor-only workflow."""
from __future__ import annotations

import inspect
from dataclasses import fields

import app
from strategies.base import StrategyDataAccess
from strategies.multifactor import strategy as multifactor_strategy
from strategies.registry import STRATEGIES
from ui.backtest import render_backtest


def test_only_multifactor_strategy_is_registered():
    assert list(STRATEGIES) == ["multifactor"]


def test_active_strategy_data_access_has_no_event_loader():
    field_names = {field.name for field in fields(StrategyDataAccess)}
    assert "get_earnings" not in field_names


def test_active_multifactor_runner_has_no_event_controls():
    params = set(inspect.signature(multifactor_strategy.run_backtest).parameters)
    assert {"ev_on", "sue_th", "max_hold"}.isdisjoint(params)


def test_streamlit_backtest_flow_has_no_event_controls():
    app_params = set(inspect.signature(app.run_backtest).parameters)
    ui_params = set(inspect.signature(render_backtest).parameters)
    forbidden = {"ev_on", "sue_th", "max_hold"}

    assert forbidden.isdisjoint(app_params)
    assert forbidden.isdisjoint(ui_params)
