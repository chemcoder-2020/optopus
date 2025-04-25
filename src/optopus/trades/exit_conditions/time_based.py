from .base import BaseComponent
import pandas as pd
from loguru import logger
from typing import Union, List
import datetime


class TimeBasedCondition(BaseComponent):
    """
    Exit condition based on a specific time before expiration.

    Attributes:
        exit_time_before_expiration (pd.Timedelta): The time before expiration to exit the trade.
    """

    def __init__(self, exit_time_before_expiration: pd.Timedelta):
        """
        Initialize the TimeBasedCondition.

        Args:
            exit_time_before_expiration (pd.Timedelta): The time before expiration to exit the trade.
        """
        self.exit_time_before_expiration = exit_time_before_expiration

    def __repr__(self):
        """
        Return a string representation of the time-based condition.

        Returns:
            str: String representation of the time-based condition.
        """
        return f"{self.__class__.__name__}(exit_time_before_expiration={self.exit_time_before_expiration})"

    def update(self, **kwargs):
        """
        Update the attributes of the time-based condition.

        Args:
            **kwargs: Keyword arguments for the attributes to update.
        """
        exit_time_before_expiration = pd.Timedelta(
            kwargs.get("exit_time_before_expiration", self.exit_time_before_expiration)
        )
        self.exit_time_before_expiration = exit_time_before_expiration

    def should_exit(
        self,
        strategy,
        current_time: Union[datetime, str, pd.Timestamp],
        option_chain_df: pd.DataFrame,
        manager=None,
    ) -> bool:
        """
        Check if the current time is within the specified time before expiration.

        Args:
            strategy (OptionStrategy): The option strategy to check.
            current_time (datetime): The current time for evaluation.
            option_chain_df (pd.DataFrame): The updated option chain data.
            manager (Optional[OptionBacktester]): The backtester instance managing the strategy. Defaults to None.

        Returns:
            bool: True if the current time is within the specified time before expiration, False otherwise.
        """
        current_time = pd.Timestamp(current_time)
        expiration_time = pd.Timestamp(strategy.legs[0].expiration).replace(
            hour=16, minute=0, second=0, microsecond=0
        )
        logger.info(
            f"Current time: {current_time}, Expiration time: {expiration_time}, Exit time before expiration: {self.exit_time_before_expiration}"
        )
        return current_time >= (expiration_time - self.exit_time_before_expiration)
