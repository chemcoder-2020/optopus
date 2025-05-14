from .base import BaseComponent
import pandas as pd
from loguru import logger
from typing import Union
import datetime


class StopLossCondition(BaseComponent):
    """
    Exit condition based on a stop loss.

    Attributes:
        stop_loss (float): The stop loss percentage.
    """

    def __init__(self, stop_loss: float, **kwargs):
        """
        Initialize the StopLossCondition.

        Args:
            stop_loss (float): The stop loss percentage
        """
        self.stop_loss = stop_loss
        self.kwargs = kwargs

    def __repr__(self):
        """
        Return a string representation of the stop loss condition.

        Returns:
            str: String representation of the stop loss condition.
        """
        return f"{self.__class__.__name__}(stop_loss={self.stop_loss})"

    def update(self, **kwargs):
        """
        Update the attributes of the stop loss condition.

        Args:
            **kwargs: Keyword arguments for the attributes to update.
        """
        stop_loss = float(kwargs.get("stop_loss", self.stop_loss))
        self.stop_loss = stop_loss

    def should_exit(
        self,
        strategy,
        current_time: Union[datetime, str, pd.Timestamp],
        option_chain_df: pd.DataFrame,
        manager=None,
    ) -> bool:
        """
        Check if the stop loss is met.

        Args:
            strategy (OptionStrategy): The option strategy to check.
            current_time (datetime): The current time for evaluation.
            option_chain_df (pd.DataFrame): The updated option chain data.
            manager (Optional[OptionBacktester]): The backtester instance managing the strategy. Defaults to None.

        Returns:
            bool: True if the stop loss is met, False otherwise.
        """
        if (
            not hasattr(strategy, "filter_return_percentage")
            or not strategy.filter_return_percentage
        ):
            return_percentage = strategy.return_percentage()
        else:
            return_percentage = strategy.filter_return_percentage
        logger.info(f"Return Percentage: {return_percentage}")
        return return_percentage <= -self.stop_loss
