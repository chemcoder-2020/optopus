from .base import EntryConditionChecker
from datetime import datetime
from typing import TYPE_CHECKING, Union
from loguru import logger
import pandas as pd

if TYPE_CHECKING:
    from ..option_manager import OptionBacktester


class ConflictCondition(EntryConditionChecker):
    """
    Checks for conflicts with existing positions.

    Args:
        check_closed_trades (bool): Whether to also check for conflicts with closed trades.

    Method:
        should_enter(strategy, manager, time) -> bool:
            Checks for conflicts with existing active and closed trades if enabled.
    """

    def __init__(self, check_closed_trades: bool = False):
        """
        Initialize the conflict condition checker.

        Args:
            check_closed_trades (bool): Whether to also check for conflicts with closed trades.
        """
        self.check_closed_trades = check_closed_trades

    def should_enter(
        self,
        strategy,
        manager: "OptionBacktester",
        time: Union[datetime, str, pd.Timestamp],
    ) -> bool:
        # Check active trades
        for existing_spread in manager.active_trades:
            if existing_spread.conflicts_with(strategy):
                logger.warning(
                    f"Position conflict detected with existing active {existing_spread.strategy_type} trade"
                )
                return False

        # Check closed trades if enabled
        if self.check_closed_trades:
            for closed_spread in manager.closed_trades:
                if closed_spread.conflicts_with(strategy):
                    logger.warning(
                        f"Position conflict detected with closed {closed_spread.strategy_type} trade"
                    )
                    return False

            if manager.closed_trades:
                logger.info(
                    f"No conflicts found with {len(manager.closed_trades)} closed trades"
                )

        if manager.active_trades:
            logger.info(
                f"No conflicts found with {len(manager.active_trades)} active trades"
            )
        else:
            logger.info("No existing active trades to check for conflicts")

        return True
