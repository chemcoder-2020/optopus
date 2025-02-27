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
            profit_target (float): Percentage return to take profit at
            trigger (float): The trigger percentage for the trailing stop
            stop_loss (float): The stop loss percentage
            exit_upon_positive_return (bool): Only exit if current return is positive
            **kwargs: Additional keyword arguments that will be set as attributes
        """
        self.profit_target = profit_target
        self.trigger = trigger
        self.stop_loss = stop_loss
        self.kwargs = kwargs

        # Set all kwargs as attributes
        for key, value in kwargs.items():
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
        self.profit_target = float(kwargs.get("profit_target", self.profit_target))
        self.trigger = float(kwargs.get("trigger", self.trigger))
        self.stop_loss = float(kwargs.get("stop_loss", self.stop_loss))

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

        # Initialize highest return if needed
        if not hasattr(strategy, "highest_return"):
            strategy.highest_return = 0
            
        # Get current return percentage
        if not hasattr(strategy, "filter_return_percentage"):
            return_percentage = strategy.return_percentage()
        else:
            return_percentage = strategy.filter_return_percentage
        
        # Update highest return with safety check
        if not np.isnan(return_percentage):
            strategy.highest_return = max(strategy.highest_return, return_percentage)

        logger.debug(f"Current return: {return_percentage}% | Highest: {strategy.highest_return}%")

        # Check profit target
        if return_percentage >= self.profit_target:
            logger.info(f"Profit target {self.profit_target}% hit at {return_percentage}%")
            return True
            
        # Check trailing stop conditions
        if strategy.highest_return >= self.trigger:
            pullback = strategy.highest_return - return_percentage
            should_exit = pullback >= self.stop_loss
            if not should_exit:
                return False
                
            return True

        return False


class TrailingStopExitCondition(ExitConditionChecker):
    """
    Composite exit condition combining trailing stop logic with time-based exit,
    using premium filtering preprocessing.
    """
    
    def __init__(
        self,
        profit_target: float,
        trigger: float,
        stop_loss: float,
        exit_time_before_expiration: pd.Timedelta,
        filter_window_size: int = 5,
        filter_n_sigma: int = 3,
        **kwargs
    ):
        super().__init__()
        self.trailing_condition = TrailingStopCondition(
            profit_target=profit_target,
            trigger=trigger,
            stop_loss=stop_loss
        )
        self.time_condition = TimeBasedCondition(
            exit_time_before_expiration=exit_time_before_expiration
        )
        self.preprocessors = [
            PremiumListInit(),
            PremiumFilter(
                window_size=filter_window_size,
                n_sigma=filter_n_sigma
            )
        ]

    def should_exit(
        self,
        strategy,
        current_time: Union[datetime, str, pd.Timestamp],
        option_chain_df: pd.DataFrame,
    ) -> bool:
        # Run preprocessing steps
        for preprocessor in self.preprocessors:
            preprocessor.preprocess(strategy)
        
        # Check both conditions with OR logic
        return (self.trailing_condition.should_exit(strategy, current_time, option_chain_df) or \
               self.time_condition.should_exit(strategy, current_time, option_chain_df)

    def update(self, **kwargs):
        """Update parameters for both conditions and filter"""
        self.trailing_condition.update(
            profit_target=kwargs.get("profit_target", self.trailing_condition.profit_target),
            trigger=kwargs.get("trigger", self.trailing_condition.trigger),
            stop_loss=kwargs.get("stop_loss", self.trailing_condition.stop_loss)
        )
        
        self.time_condition.update(
            exit_time_before_expiration=kwargs.get(
                "exit_time_before_expiration",
                self.time_condition.exit_time_before_expiration
            )
        )
        
        # Update filter parameters if provided
        for preprocessor in self.preprocessors:
            if isinstance(preprocessor, PremiumFilter):
                preprocessor.update(
                    window_size=kwargs.get("filter_window_size", preprocessor.window_size),
                    n_sigma=kwargs.get("filter_n_sigma", preprocessor.n_sigma)
                )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"profit_target={self.trailing_condition.profit_target}%, "
            f"trigger={self.trailing_condition.trigger}%, "
            f"stop_loss={self.trailing_condition.stop_loss}%, "
            f"exit_time={self.time_condition.exit_time_before_expiration}, "
            f"filter={self.preprocessors[1]})"
        )
