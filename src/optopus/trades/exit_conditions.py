"""
Module for defining exit conditions for option trading strategies.

This module provides an abstract base class `ExitConditionChecker` and several concrete implementations
for different exit conditions, such as profit targets, stop losses, and time-based conditions.
"""

from abc import ABC, abstractmethod
import datetime
import pandas as pd
import numpy as np
from typing import Union, List
from loguru import logger
from ..utils.heapmedian import ContinuousMedian

class ExitConditionChecker(ABC):
    """
    Abstract base class for exit condition checkers.

    Methods:
        should_exit(strategy, current_time: Union[datetime, str, pd.Timestamp], option_chain_df: pd.DataFrame) -> bool:
            Check if the exit conditions are met for the option strategy.
    """

    def __repr__(self):
        """
        Return a string representation of the exit condition checker.

        Returns:
            str: String representation of the exit condition checker.
        """
        return f"{self.__class__.__name__}()"

    def update(self, **kwargs):
        """
        Update the attributes of the exit condition checker.

        Args:
            **kwargs: Keyword arguments for the attributes to update.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    @abstractmethod
    def should_exit(self, strategy, current_time: Union[datetime, str, pd.Timestamp], option_chain_df: pd.DataFrame) -> bool:
        """
        Check if the exit conditions are met for the option strategy.

        Args:
            strategy (OptionStrategy): The option strategy to check.
            current_time (datetime): The current time for evaluation.
            option_chain_df (pd.DataFrame): The updated option chain data.

        Returns:
            bool: True if the exit conditions are met, False otherwise.
        """
        pass

class MedianCalculator:
    """
    Class for calculating the median of a rolling window of premiums.

    Attributes:
        window_size (int): The size of the rolling window.
        premiums (list): List of premiums.
        median_calculator (ContinuousMedian): Continuous median calculator.
    """

    def __init__(self, window_size=5):
        """
        Initialize the MedianCalculator.

        Args:
            window_size (int): The size of the rolling window.
        """
        self.median_calculator = ContinuousMedian()
        self.window_size = window_size
        self.premiums = []

    def add_premium(self, mark):
        """
        Add a new premium to the rolling window.

        Args:
            mark (float): The new premium.
        """
        self.median_calculator.add(mark)
        self.premiums.append(mark)
        if len(self.premiums) > self.window_size:
            self.median_calculator.remove(self.premiums.pop(0))

    def get_median(self):
        """
        Get the current median of the rolling window.

        Returns:
            float: The current median.
        """
        return self.median_calculator.get_median()
    
    def get_median_return_percentage(self, strategy):
        """
        Calculate the median return percentage for the given strategy.

        Args:
            strategy (OptionStrategy): The option strategy to calculate the return percentage for.

        Returns:
            float: The median return percentage.
        """
        bid = strategy.current_bid
        ask = strategy.current_ask
        mark = (ask + bid) / 2
        self.add_premium(mark)
        median_net_premium = self.get_median()
        strategy.premium_log = self.premiums.copy()

        if hasattr(strategy, "strategy_side") and strategy.strategy_side == "CREDIT":
            median_pl = (
                (strategy.entry_net_premium - median_net_premium) * 100 * strategy.contracts
            ) - strategy.calculate_total_commission()
        elif hasattr(strategy, "strategy_side") and strategy.strategy_side == "DEBIT":
            median_pl = (
                (median_net_premium - strategy.entry_net_premium) * 100 * strategy.contracts
            ) - strategy.calculate_total_commission()
        else:
            raise ValueError(f"Unsupported strategy side: {strategy.strategy_side}")

        premium = abs(strategy.entry_net_premium)
        if premium == 0:
            return 0
        return (median_pl / (premium * 100 * strategy.contracts)) * 100

class ProfitTargetCondition(ExitConditionChecker):
    """
    Exit condition based on a profit target.

    Attributes:
        profit_target (float): The profit target percentage.
        median_calculator (MedianCalculator): Median calculator for rolling window.
    """

    def __init__(self, profit_target: float, **kwargs):
        """
        Initialize the ProfitTargetCondition.

        Args:
            profit_target (float): The profit target percentage.
            **kwargs: Additional keyword arguments.
        """
        self.profit_target = profit_target
        self.median_calculator = MedianCalculator(kwargs.get("window_size", 5))
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

    def should_exit(self, strategy, current_time: Union[datetime, str, pd.Timestamp], option_chain_df: pd.DataFrame) -> bool:
        """
        Check if the profit target is met.

        Args:
            strategy (OptionStrategy): The option strategy to check.
            current_time (datetime): The current time for evaluation.
            option_chain_df (pd.DataFrame): The updated option chain data.

        Returns:
            bool: True if the profit target is met, False otherwise.
        """
        current_return = strategy.return_percentage()
        logger.debug(f"Current return: {current_return}")
        current_median_return = self.median_calculator.get_median_return_percentage(strategy)
        strategy.median_return_percentage = current_median_return
        logger.debug(f"Current median return: {current_median_return}")
        return current_return >= self.profit_target and current_median_return >= self.profit_target
    
class StopLossCondition(ExitConditionChecker):
    """
    Exit condition based on a stop loss.

    Attributes:
        stop_loss (float): The stop loss percentage.
        median_calculator (MedianCalculator): Median calculator for rolling window.
    """

    def __init__(self, stop_loss: float, **kwargs):
        """
        Initialize the StopLossCondition.

        Args:
            stop_loss (float): The stop loss percentage.
            **kwargs: Additional keyword arguments.
        """
        self.stop_loss = stop_loss
        self.median_calculator = MedianCalculator(kwargs.get("window_size", 5))
        self.kwargs = kwargs

    def __repr__(self):
        """
        Return a string representation of the stop loss condition.

        Returns:
            str: String representation of the stop loss condition.
        """
        return f"{self.__class__.__name__}(stop_loss={self.stop_loss}, window_size={self.median_calculator.window_size})"

    def update(self, **kwargs):
        """
        Update the attributes of the stop loss condition.

        Args:
            **kwargs: Keyword arguments for the attributes to update.
        """
        stop_loss = float(kwargs.get("stop_loss", self.stop_loss))
        self.stop_loss = stop_loss

    def should_exit(self, strategy, current_time: Union[datetime, str, pd.Timestamp], option_chain_df: pd.DataFrame) -> bool:
        """
        Check if the stop loss is met.

        Args:
            strategy (OptionStrategy): The option strategy to check.
            current_time (datetime): The current time for evaluation.
            option_chain_df (pd.DataFrame): The updated option chain data.

        Returns:
            bool: True if the stop loss is met, False otherwise.
        """
        current_median_return = self.median_calculator.get_median_return_percentage(strategy)
        strategy.median_return_percentage = current_median_return
        return current_median_return <= -self.stop_loss

class TimeBasedCondition(ExitConditionChecker):
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
        exit_time_before_expiration = pd.Timedelta(kwargs.get("exit_time_before_expiration", self.exit_time_before_expiration))
        self.exit_time_before_expiration = exit_time_before_expiration

    def should_exit(self, strategy, current_time: Union[datetime, str, pd.Timestamp], option_chain_df: pd.DataFrame) -> bool:
        """
        Check if the current time is within the specified time before expiration.

        Args:
            strategy (OptionStrategy): The option strategy to check.
            current_time (datetime): The current time for evaluation.
            option_chain_df (pd.DataFrame): The updated option chain data.

        Returns:
            bool: True if the current time is within the specified time before expiration, False otherwise.
        """
        current_time = pd.Timestamp(current_time)
        expiration_time = pd.Timestamp(strategy.legs[0].expiration).replace(hour=16, minute=0, second=0, microsecond=0)
        return current_time >= (expiration_time - self.exit_time_before_expiration)

class TrailingStopCondition(ExitConditionChecker):
    """
    Exit condition based on a trailing stop.

    Attributes:
        trigger (float): The trigger percentage for the trailing stop.
        stop_loss (float): The stop loss percentage.
        median_calculator (MedianCalculator): Median calculator for rolling window.
    """

    def __init__(self, trigger: float, stop_loss: float, **kwargs):
        """
        Initialize the TrailingStopCondition.

        Args:
            trigger (float): The trigger percentage for the trailing stop.
            stop_loss (float): The stop loss percentage.
            **kwargs: Additional keyword arguments.
        """
        self.trigger = trigger
        self.stop_loss = stop_loss
        self.median_calculator = MedianCalculator(kwargs.get("window_size", 5))
        self.kwargs = kwargs

    def __repr__(self):
        """
        Return a string representation of the trailing stop condition.

        Returns:
            str: String representation of the trailing stop condition.
        """
        return f"{self.__class__.__name__}(trigger={self.trigger}, stop_loss={self.stop_loss}, window_size={self.median_calculator.window_size})"
    
    def update(self, **kwargs):
        """
        Update the attributes of the trailing stop condition.

        Args:
            **kwargs: Keyword arguments for the attributes to update.
        """
        trigger = float(kwargs.get("trigger", self.trigger))
        stop_loss = float(kwargs.get("stop_loss", self.stop_loss))
        self.trigger = trigger
        self.stop_loss = stop_loss

    def should_exit(self, strategy, current_time: Union[datetime, str, pd.Timestamp], option_chain_df: pd.DataFrame) -> bool:
        """
        Check if the trailing stop condition is met.

        Args:
            strategy (OptionStrategy): The option strategy to check.
            current_time (datetime): The current time for evaluation.
            option_chain_df (pd.DataFrame): The updated option chain data.

        Returns:
            bool: True if the trailing stop condition is met, False otherwise.
        """
        current_median_return = self.median_calculator.get_median_return_percentage(strategy)
        strategy.median_return_percentage = current_median_return
        highest_return = strategy.highest_return

        if highest_return >= self.trigger:
            return (highest_return - current_median_return) >= self.stop_loss

        return False

class CompositeExitCondition(ExitConditionChecker):
    """
    Composite exit condition that combines multiple exit conditions.

    Attributes:
        conditions (List[ExitConditionChecker]): List of exit conditions to combine.
        logical_operations (List[str]): List of logical operations to combine the conditions ('AND' or 'OR').
    """

    def __init__(self, conditions: List[ExitConditionChecker], logical_operations: List[str] = ['AND']):
        """
        Initialize the CompositeExitCondition.

        Args:
            conditions (List[ExitConditionChecker]): List of exit conditions to combine.
            logical_operations (List[str]): List of logical operations to combine the conditions ('AND' or 'OR').
        """
        self.conditions = conditions
        self.logical_operations = logical_operations
        for condition in conditions:
            self.__dict__.update(condition.__dict__)

    def __repr__(self):
        """
        Return a string representation of the composite exit condition.

        Returns:
            str: String representation of the composite exit condition.
        """
        return f"{self.__class__.__name__}(conditions={self.conditions}, logical_operations='{self.logical_operations}')"

    def should_exit(self, strategy, current_time: Union[datetime, str, pd.Timestamp], option_chain_df: pd.DataFrame) -> bool:
        """
        Check if the composite exit condition is met.

        Args:
            strategy (OptionStrategy): The option strategy to check.
            current_time (datetime): The current time for evaluation.
            option_chain_df (pd.DataFrame): The updated option chain data.

        Returns:
            bool: True if the composite exit condition is met, False otherwise.
        """
        if len(self.conditions) != len(self.logical_operations) + 1:
            raise ValueError("The number of logical operations must be one less than the number of conditions.")

        results = [condition.should_exit(strategy, current_time, option_chain_df) for condition in self.conditions]
        combined_result = results[0]

        for i, operation in enumerate(self.logical_operations):
            if operation == 'AND':
                combined_result = combined_result and results[i + 1]
            elif operation == 'OR':
                combined_result = combined_result or results[i + 1]
            else:
                raise ValueError("Logical operation must be 'AND' or 'OR'")

        return combined_result

    def update(self, **kwargs):
        """
        Update the attributes of the composite exit condition.

        Args:
            **kwargs: Keyword arguments for the attributes to update.
        """
        for condition in self.conditions:
            condition.update(**kwargs)
            self.__dict__.update(condition.__dict__)

    def __setattr__(self, key, value):
        """
        Override the __setattr__ method to update attributes in the conditions list.
        """
        super().__setattr__(key, value)
        for condition in self.conditions:
            if hasattr(condition, key):
                setattr(condition, key, value)

class DefaultExitCondition(ExitConditionChecker):
    """
    Default exit condition that combines a profit target and a time-based condition.

    Attributes:
        profit_target (float): The profit target percentage.
        exit_time_before_expiration (pd.Timedelta): The time before expiration to exit the trade.
        composite_condition (CompositeExitCondition): Composite exit condition combining profit target and time-based conditions.
    """

    def __init__(self, profit_target: float=40, exit_time_before_expiration: pd.Timedelta=pd.Timedelta(minutes=15), **kwargs):
        """
        Initialize the DefaultExitCondition.

        Args:
            profit_target (float): The profit target percentage.
            exit_time_before_expiration (pd.Timedelta): The time before expiration to exit the trade.
            **kwargs: Additional keyword arguments.
        """
        profit_target_condition = ProfitTargetCondition(profit_target=profit_target, window_size=kwargs.get("window_size", 5))
        time_based_condition = TimeBasedCondition(exit_time_before_expiration=exit_time_before_expiration)
        self.composite_condition = CompositeExitCondition(
            conditions=[profit_target_condition, time_based_condition],
            logical_operations=['OR']
        )
        self.__dict__.update(self.composite_condition.__dict__)

    def __repr__(self):
        """
        Return a string representation of the default exit condition.

        Returns:
            str: String representation of the default exit condition.
        """
        return f"{self.__class__.__name__}(composite_condition={self.composite_condition})"

    def should_exit(self, strategy, current_time: Union[datetime, str, pd.Timestamp], option_chain_df: pd.DataFrame) -> bool:
        """
        Check if the default exit condition is met.

        Args:
            strategy (OptionStrategy): The option strategy to check.
            current_time (datetime): The current time for evaluation.
            option_chain_df (pd.DataFrame): The updated option chain data.

        Returns:
            bool: True if the default exit condition is met, False otherwise.
        """
        return self.composite_condition.should_exit(strategy, current_time, option_chain_df)
    
    def update(self, **kwargs):
        """
        Update the attributes of the default exit condition.

        Args:
            **kwargs: Keyword arguments for the attributes to update.
        """
        self.composite_condition.update(**kwargs)
        self.__dict__.update(self.composite_condition.__dict__)
