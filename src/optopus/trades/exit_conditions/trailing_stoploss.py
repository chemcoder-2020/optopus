from .base import BaseComponent
import pandas as pd
from loguru import logger
from typing import Union, List
import datetime
import numpy as np


class TrailingStopCondition(BaseComponent):
    """
    Exit condition based on a trailing stop.

    Attributes:
        trigger (float): The trigger percentage for the trailing stop.
        stop_loss (float): The stop loss percentage.
    """

    def __init__(
        self,
        profit_target: float = 80,
        trigger: float = 40,
        stop_loss: float = 15,
        **kwargs,
    ):
        """
        Initialize the TrailingStopCondition.

        Args:
            trigger (float): The trigger percentage for the trailing stop
            stop_loss (float): The stop loss percentage
            **kwargs: Additional keyword arguments that will be set as attributes
        """
        self.profit_target = profit_target
        self.trigger = trigger
        self.stop_loss = stop_loss
        self.kwargs = kwargs

        # Set all kwargs as attributes
        for key, value in kwargs.items():
            if key != "window_size":
                setattr(self, key, value)

    def __repr__(self):
        """
        Return a string representation of the trailing stop condition.

        Returns:
            str: String representation of the trailing stop condition.
        """
        return f"{self.__class__.__name__}(trigger={self.trigger}, stop_loss={self.stop_loss})"

    def update(self, **kwargs):
        """
        Update the attributes of the trailing stop condition.

        Args:
            **kwargs: Keyword arguments for the attributes to update.
        """
        profit_target = float(kwargs.get("profit_target", self.profit_target))
        trigger = float(kwargs.get("trigger", self.trigger))
        stop_loss = float(kwargs.get("stop_loss", self.stop_loss))

        self.profit_target = profit_target
        self.trigger = trigger
        self.stop_loss = stop_loss

    def should_exit(
        self,
        strategy,
        current_time: Union[datetime, str, pd.Timestamp],
        option_chain_df: pd.DataFrame,
    ) -> bool:
        """
        Check if the trailing stop condition is met.

        Args:
            strategy (OptionStrategy): The option strategy to check.
            current_time (datetime): The current time for evaluation.
            option_chain_df (pd.DataFrame): The updated option chain data.

        Returns:
            bool: True if the trailing stop condition is met, False otherwise.
        """

        if not hasattr(strategy, "filter_return_percentage"):
            return_percentage = strategy.return_percentage()
        else:
            return_percentage = strategy.filter_return_percentage
        
        if not np.isnan(return_percentage):
            strategy.highest_return = max(strategy.highest_return, return_percentage)

        logger.info(f"Updated strategy highest return to {strategy.highest_return}")

        profit_hit = return_percentage >= self.profit_target
        if profit_hit:
            return True
        
        if strategy.highest_return >= self.trigger:
            pullback = strategy.highest_return - return_percentage
            logger.info(f"Trailing Stop Pullback: {pullback}% : {strategy.highest_return} -> {return_percentage}")
            main_condition = pullback >= self.stop_loss
            if main_condition and not self.exit_upon_positive_return:
                return True
            elif main_condition and self.exit_upon_positive_return:
                if return_percentage > 0:
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False
