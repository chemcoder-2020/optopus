from .base import EntryConditionChecker
from datetime import datetime
from typing import TYPE_CHECKING, Union
from loguru import logger
import pandas as pd

if TYPE_CHECKING:
    from ..option_manager import OptionBacktester


class PositionLimitCondition(EntryConditionChecker):
    """
    Checks if position limits are respected.

    Method:
        should_enter(strategy, manager, time) -> bool:
            Checks if the number of active trades, daily trades, and weekly trades are within the configured limits.
    """

    def should_enter(
        self,
        strategy,
        manager: "OptionBacktester",
        time: Union[datetime, str, pd.Timestamp],
    ) -> bool:
        position_limit_ok = len(manager.active_trades) < manager.config.max_positions
        if not position_limit_ok:
            logger.warning(
                f"Position limit reached: {len(manager.active_trades)} active trades >= limit of {manager.config.max_positions}"
            )
        else:
            logger.info(
                f"Position limit ok: {len(manager.active_trades)} active trades < limit of {manager.config.max_positions}"
            )

        daily_limit_ok = (
            manager.config.max_positions_per_day is None
            or manager.trades_entered_today < manager.config.max_positions_per_day
        )
        if not daily_limit_ok:
            logger.warning(
                f"Daily position limit reached: {manager.trades_entered_today} trades today >= limit of {manager.config.max_positions_per_day}"
            )
        elif manager.config.max_positions_per_day is not None:
            logger.info(
                f"Daily position limit ok: {manager.trades_entered_today} trades today < limit of {manager.config.max_positions_per_day}"
            )

        weekly_limit_ok = (
            manager.config.max_positions_per_week is None
            or manager.trades_entered_this_week < manager.config.max_positions_per_week
        )
        if not weekly_limit_ok:
            logger.warning(
                f"Weekly position limit reached: {manager.trades_entered_this_week} trades this week >= limit of {manager.config.max_positions_per_week}"
            )
        elif manager.config.max_positions_per_week is not None:
            logger.info(
                f"Weekly position limit ok: {manager.trades_entered_this_week} trades this week < limit of {manager.config.max_positions_per_week}"
            )

        return all([position_limit_ok, daily_limit_ok, weekly_limit_ok])
