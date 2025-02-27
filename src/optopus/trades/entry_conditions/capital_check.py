from .base import EntryConditionChecker
from typing import Union, TYPE_CHECKING
from datetime import datetime
from loguru import logger
import pandas as pd

if TYPE_CHECKING:
    from ..option_manager import OptionBacktester


class CapitalRequirementCondition(EntryConditionChecker):
    """
    Checks if there is sufficient capital for the trade.

    Method:
        should_enter(strategy, manager, time) -> bool:
            Adjusts the number of contracts based on the capital requirement and checks if the required capital is available.
    """

    def should_enter(
        self,
        strategy,
        manager: "OptionBacktester",
        time: Union[datetime, str, pd.Timestamp],
    ) -> bool:
        max_capital = min(
            manager.allocation * manager.config.position_size,
            manager.available_to_trade,
        )
        original_contracts = strategy.contracts
        strategy.contracts = min(
            original_contracts,
            int(max_capital // strategy.get_required_capital_per_contract()),
        )

        if strategy.contracts == 0:
            logger.warning(
                "Capital requirement not met: Strategy requires more capital than allowed by position size"
            )
            return False

        if strategy.contracts != original_contracts:
            logger.info(
                f"Adjusted contracts from {original_contracts} to {strategy.contracts} to fit position size"
            )

        required_capital = strategy.get_required_capital()
        has_sufficient_capital = (
            required_capital <= max_capital
            and required_capital <= manager.available_to_trade
        )

        if has_sufficient_capital:
            logger.info(
                f"Capital requirement met: Required ${required_capital:.2f} <= Available ${manager.available_to_trade:.2f}"
            )
        else:
            logger.warning(
                f"Capital requirement not met: Required ${required_capital:.2f} > Available ${manager.available_to_trade:.2f}"
            )

        return has_sufficient_capital
