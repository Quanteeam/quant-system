"""Strategy registry used by the UI and future model runners."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from strategies.base import StrategyDataAccess, StrategyDefinition
from strategies.multifactor import strategy as multifactor


@dataclass(frozen=True)
class RegisteredStrategy:
    definition: StrategyDefinition
    run_backtest: Callable


STRATEGIES: dict[str, RegisteredStrategy] = {
    multifactor.DEFINITION.key: RegisteredStrategy(
        definition=multifactor.DEFINITION,
        run_backtest=multifactor.run_backtest,
    ),
}


def strategy_options() -> list[StrategyDefinition]:
    return [registered.definition for registered in STRATEGIES.values()]


def get_strategy(key: str) -> RegisteredStrategy:
    try:
        return STRATEGIES[key]
    except KeyError as exc:
        available = ", ".join(sorted(STRATEGIES))
        raise ValueError(f"Unknown strategy '{key}'. Available strategies: {available}") from exc
