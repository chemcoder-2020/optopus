from .base import EntryConditionChecker
from datetime import datetime
from typing import TYPE_CHECKING, Union
from loguru import logger
import pandas as pd

if TYPE_CHECKING:
    from ..option_manager import OptionBacktester


class RORThresholdCondition(EntryConditionChecker):
    """
    Checks if return over risk meets the threshold.

    Method:
        should_enter(strategy, manager, time) -> bool:
            Checks if the return over risk meets the configured threshold.
    """

    def should_enter(
        self,
        strategy,
        manager: "OptionBacktester",
        time: Union[datetime, str, pd.Timestamp],
    ) -> bool:
        if manager.config.ror_threshold is None:
            logger.info("ROR threshold check skipped: No threshold configured")
            return True

        ror = strategy.return_over_risk()
        meets_threshold = ror >= manager.config.ror_threshold

        if meets_threshold:
            logger.info(
                f"ROR threshold met: {ror:.2%} >= {manager.config.ror_threshold:.2%}"
            )
        else:
            logger.warning(
                f"ROR threshold not met: {ror:.2%} < {manager.config.ror_threshold:.2%}"
            )

        return meets_threshold
