from abc import ABC, abstractmethod
import datetime
import pandas as pd
from typing import Union, TYPE_CHECKING, List
from loguru import logger

if TYPE_CHECKING:
    from .option_manager import OptionBacktester

class EntryConditionChecker(ABC):
    """
    Abstract base class for entry condition checkers.

    Methods:
        should_enter() -> bool:
            Check if the entry conditions are met for the option strategy.
    """
    @abstractmethod
    def should_enter(self, strategy, manager: 'OptionBacktester', time: Union[datetime, str, pd.Timestamp]) -> bool:
        pass

class CapitalRequirementCondition(EntryConditionChecker):
    """Checks if there is sufficient capital for the trade."""
    def should_enter(self, strategy, manager: 'OptionBacktester', time: Union[datetime, str, pd.Timestamp]) -> bool:
        max_capital = min(manager.allocation * manager.config.position_size, manager.capital)
        original_contracts = strategy.contracts
        strategy.contracts = min(
            original_contracts,
            int(max_capital // strategy.get_required_capital_per_contract())
        )

        if strategy.contracts == 0:
            logger.warning("Strategy requires more capital than allowed by position size")
            return False

        if strategy.contracts != original_contracts:
            logger.info(f"Adjusted contracts from {original_contracts} to {strategy.contracts} to fit position size")

        return (strategy.get_required_capital() <= max_capital and 
                strategy.get_required_capital() <= manager.available_to_trade)

class PositionLimitCondition(EntryConditionChecker):
    """Checks if position limits are respected."""
    def should_enter(self, strategy, manager: 'OptionBacktester', time: Union[datetime, str, pd.Timestamp]) -> bool:
        conditions = [
            len(manager.active_trades) < manager.config.max_positions,
            manager.config.max_positions_per_day is None or 
            manager.trades_entered_today < manager.config.max_positions_per_day,
            manager.config.max_positions_per_week is None or 
            manager.trades_entered_this_week < manager.config.max_positions_per_week
        ]
        return all(conditions)

class RORThresholdCondition(EntryConditionChecker):
    """Checks if return over risk meets the threshold."""
    def should_enter(self, strategy, manager: 'OptionBacktester', time: Union[datetime, str, pd.Timestamp]) -> bool:
        return (manager.config.ror_threshold is None or 
                strategy.return_over_risk() >= manager.config.ror_threshold)

class ConflictCondition(EntryConditionChecker):
    """Checks for conflicts with existing positions."""
    def should_enter(self, strategy, manager: 'OptionBacktester', time: Union[datetime, str, pd.Timestamp]) -> bool:
        return not any(existing.conflicts_with(strategy) for existing in manager.active_trades)

class CompositeEntryCondition(EntryConditionChecker):
    """Combines multiple entry conditions."""
    def __init__(self, conditions: List[EntryConditionChecker]):
        self.conditions = conditions

    def should_enter(self, strategy, manager: 'OptionBacktester', time: Union[datetime, str, pd.Timestamp]) -> bool:
        for condition in self.conditions:
            if not condition.should_enter(strategy, manager, time):
                return False
        return True

class DefaultEntryCondition(EntryConditionChecker):
    """
    Default entry condition that combines all standard checks.
    """
    def __init__(self):
        self.composite = CompositeEntryCondition([
            CapitalRequirementCondition(),
            PositionLimitCondition(),
            RORThresholdCondition(),
            ConflictCondition()
        ])

    def should_enter(self, strategy, manager: 'OptionBacktester', time: Union[datetime, str, pd.Timestamp]) -> bool:
        return self.composite.should_enter(strategy, manager, time)
