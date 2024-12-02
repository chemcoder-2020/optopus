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
            logger.warning("Capital requirement not met: Strategy requires more capital than allowed by position size")
            return False

        if strategy.contracts != original_contracts:
            logger.info(f"Adjusted contracts from {original_contracts} to {strategy.contracts} to fit position size")

        required_capital = strategy.get_required_capital()
        has_sufficient_capital = required_capital <= max_capital and required_capital <= manager.available_to_trade
        
        if has_sufficient_capital:
            logger.info(f"Capital requirement met: Required ${required_capital:.2f} <= Available ${manager.available_to_trade:.2f}")
        else:
            logger.warning(f"Capital requirement not met: Required ${required_capital:.2f} > Available ${manager.available_to_trade:.2f}")
        
        return has_sufficient_capital

class PositionLimitCondition(EntryConditionChecker):
    """Checks if position limits are respected."""
    def should_enter(self, strategy, manager: 'OptionBacktester', time: Union[datetime, str, pd.Timestamp]) -> bool:
        position_limit_ok = len(manager.active_trades) < manager.config.max_positions
        if not position_limit_ok:
            logger.warning(f"Position limit reached: {len(manager.active_trades)} active trades >= limit of {manager.config.max_positions}")
        else:
            logger.info(f"Position limit ok: {len(manager.active_trades)} active trades < limit of {manager.config.max_positions}")

        daily_limit_ok = (manager.config.max_positions_per_day is None or 
                         manager.trades_entered_today < manager.config.max_positions_per_day)
        if not daily_limit_ok:
            logger.warning(f"Daily position limit reached: {manager.trades_entered_today} trades today >= limit of {manager.config.max_positions_per_day}")
        elif manager.config.max_positions_per_day is not None:
            logger.info(f"Daily position limit ok: {manager.trades_entered_today} trades today < limit of {manager.config.max_positions_per_day}")

        weekly_limit_ok = (manager.config.max_positions_per_week is None or 
                          manager.trades_entered_this_week < manager.config.max_positions_per_week)
        if not weekly_limit_ok:
            logger.warning(f"Weekly position limit reached: {manager.trades_entered_this_week} trades this week >= limit of {manager.config.max_positions_per_week}")
        elif manager.config.max_positions_per_week is not None:
            logger.info(f"Weekly position limit ok: {manager.trades_entered_this_week} trades this week < limit of {manager.config.max_positions_per_week}")

        return all([position_limit_ok, daily_limit_ok, weekly_limit_ok])

class RORThresholdCondition(EntryConditionChecker):
    """Checks if return over risk meets the threshold."""
    def should_enter(self, strategy, manager: 'OptionBacktester', time: Union[datetime, str, pd.Timestamp]) -> bool:
        if manager.config.ror_threshold is None:
            logger.info("ROR threshold check skipped: No threshold configured")
            return True
            
        ror = strategy.return_over_risk()
        meets_threshold = ror >= manager.config.ror_threshold
        
        if meets_threshold:
            logger.info(f"ROR threshold met: {ror:.2%} >= {manager.config.ror_threshold:.2%}")
        else:
            logger.warning(f"ROR threshold not met: {ror:.2%} < {manager.config.ror_threshold:.2%}")
            
        return meets_threshold

class ConflictCondition(EntryConditionChecker):
    """Checks for conflicts with existing positions."""
    def should_enter(self, strategy, manager: 'OptionBacktester', time: Union[datetime, str, pd.Timestamp]) -> bool:
        return not any(existing.conflicts_with(strategy) for existing in manager.active_trades)

class CompositeEntryCondition(EntryConditionChecker):
    """Combines multiple entry conditions."""
    def __init__(self, conditions: List[EntryConditionChecker]):
        self.conditions = conditions

    def should_enter(self, strategy, manager: 'OptionBacktester', time: Union[datetime, str, pd.Timestamp]) -> bool:
        logger.info(f"Checking {len(self.conditions)} composite conditions")
        
        for i, condition in enumerate(self.conditions, 1):
            condition_name = condition.__class__.__name__
            if not condition.should_enter(strategy, manager, time):
                logger.warning(f"Composite condition {i}/{len(self.conditions)} ({condition_name}) failed")
                return False
            logger.info(f"Composite condition {i}/{len(self.conditions)} ({condition_name}) passed")
            
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
