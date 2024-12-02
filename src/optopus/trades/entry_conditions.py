from abc import ABC, abstractmethod
import datetime
import pandas as pd
from typing import Union, TYPE_CHECKING
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

class DefaultEntryCondition(EntryConditionChecker):
    """
    Default entry condition that checks position limits, capital requirements,
    and conflicts with existing positions.
    """
    def should_enter(self, strategy, manager: 'OptionBacktester', time: Union[datetime, str, pd.Timestamp]) -> bool:
        """
        Check if entry conditions are met.
        
        Args:
            strategy: The option strategy to check
            manager: The option manager instance
            time: Current time
            
        Returns:
            bool: True if all entry conditions are met
        """
        # Check capital requirements
        max_capital = min(manager.allocation * manager.config.position_size, manager.capital)
        original_contracts = strategy.contracts
        strategy.contracts = min(
            original_contracts,
            int(max_capital // strategy.get_required_capital_per_contract())
        )

        if strategy.contracts == 0:
            logger.warning(f"Strategy requires more capital than allowed by position size")
            return False

        if strategy.contracts != original_contracts:
            logger.info(f"Adjusted contracts from {original_contracts} to {strategy.contracts} to fit position size")

        # Check all entry conditions
        conditions = [
            ("No conflict", not any(existing.conflicts_with(strategy) for existing in manager.active_trades)),
            ("Meets ROR threshold", 
             manager.config.ror_threshold is None or strategy.return_over_risk() >= manager.config.ror_threshold),
            ("Within max positions", len(manager.active_trades) < manager.config.max_positions),
            ("Within max positions per day",
             manager.config.max_positions_per_day is None or 
             manager.trades_entered_today < manager.config.max_positions_per_day),
            ("Within max positions per week",
             manager.config.max_positions_per_week is None or 
             manager.trades_entered_this_week < manager.config.max_positions_per_week),
            ("Within max capital", strategy.get_required_capital() <= max_capital),
            ("Sufficient capital", strategy.get_required_capital() <= manager.available_to_trade)
        ]

        for condition_name, condition_result in conditions:
            if not condition_result:
                logger.info(f"Entry condition not met: {condition_name}")
                return False
            logger.debug(f"Entry condition met: {condition_name}")

        return True
