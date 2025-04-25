from .base import BaseComponent
import pandas as pd
from loguru import logger
from typing import Union, List
import datetime


class ProfitTargetCondition(BaseComponent):
    """
    Exit condition based on a profit target.

    Attributes:
        profit_target (float): The profit target percentage.
    """

    def __init__(self, profit_target: float, **kwargs):
        """
        Initialize the ProfitTargetCondition.

        Args:
            profit_target (float): The profit target percentage
            **kwargs: Additional keyword arguments for future extension
        """
        self.profit_target = profit_target
        self.kwargs = kwargs

    def __repr__(self):
        """
        Return a string representation of the profit target condition.

        Returns:
            str: String representation of the profit target condition.
        """
        return f"{self.__class__.__name__}(profit_target={self.profit_target})"

    def update(self, **kwargs):
        """
        Update the attributes of the profit target condition.

        Args:
            **kwargs: Keyword arguments for the attributes to update.
        """
        profit_target = float(kwargs.get("profit_target", self.profit_target))
        self.profit_target = profit_target

    def should_exit(
        self,
        strategy,
        current_time: Union[datetime, str, pd.Timestamp],
        option_chain_df: pd.DataFrame,
        manager=None,
    ) -> bool:
        """
        Check if the profit target is met.

        Args:
            strategy (OptionStrategy): The option strategy to check.
            current_time (datetime): The current time for evaluation.
            option_chain_df (pd.DataFrame): The updated option chain data.
            manager (Optional[OptionBacktester]): The backtester instance managing the strategy. Defaults to None.

        Returns:
            bool: True if the profit target is met, False otherwise.
        """
        if not hasattr(strategy, "filter_return_percentage") or not strategy.filter_return_percentage:
            return_percentage = strategy.return_percentage()
        else:
            return_percentage = strategy.filter_return_percentage
        logger.info(f"Return Percentage: {return_percentage}")
        return return_percentage >= self.profit_target
