"""
Module for defining exit conditions for option trading strategies.

This module provides an abstract base class `ExitConditionChecker` and several concrete implementations
for different exit conditions, such as profit targets, stop losses, and time-based conditions.
"""

from abc import ABC, abstractmethod
import datetime
import pandas as pd
from typing import Union, List
from loguru import logger
import numpy as np
from sktime.transformations.series.outlier_detection import HampelFilter
from sktime.transformations.series.impute import Imputer
from ...utils.filters import HampelFilterNumpy
from ...utils.heapmedian import ContinuousMedian


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
    def should_exit(
        self,
        strategy,
        current_time: Union[datetime, str, pd.Timestamp],
        option_chain_df: pd.DataFrame,
    ) -> bool:
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

    def __init__(self, window_size=5, method="ContinuousMedian", **kwargs):
        """
        Initialize the MedianCalculator.

        Args:
            window_size (int): The size of the rolling window.
            method (str): The method for calculating the median ('ContinuousMedian' or 'HampelFilter').
        Kwargs:
            **kwargs: Keyword arguments for the attributes to update.
        """
        self.window_size = window_size
        self.method = method
        self.kwargs = kwargs
        if method == "ContinuousMedian":
            self.median_calculator = ContinuousMedian()
        else:
            self.median_calculator = HampelFilterNumpy(
                window_size=window_size,
                n_sigma=self.kwargs.get("n_sigma", 3),
                k=self.kwargs.get("k", 1.4826),
                max_iterations=self.kwargs.get("max_iterations", 5),
                replace_with_na=self.kwargs.get("replace_with_na", False),
            )

        self.premiums = []

    def add_premium(self, mark):
        """
        Add a new premium to the rolling window.

        Args:
            mark (float): The new premium.
        """
        if self.method == "ContinuousMedian":
            self.median_calculator.add(mark)
        self.premiums.append(mark)

        if self.method == "ContinuousMedian":
            if len(self.premiums) > self.window_size:
                self.median_calculator.remove(self.premiums.pop(0))
        else:
            if len(self.premiums) > self.window_size + 1:
                self.premiums.pop(0)

    def get_median(self):
        """
        Get the current median of the rolling window.

        Returns:
            float: The current median.
        """
        if self.method == "ContinuousMedian":
            return self.median_calculator.get_median()
        else:
            if len(self.premiums) < self.window_size + 1:
                return 0
            else:
                return self.median_calculator.fit_transform(
                    np.array(self.premiums)
                ).flatten()[-1]

    def update(self, **kwargs):
        """
        Update the attributes of the MedianCalculator.

        Args:
            **kwargs: Keyword arguments for the attributes to update.
        """
        if "window_size" in kwargs:
            new_window_size = kwargs["window_size"]
            # Adjust the window size while maintaining existing data
            if new_window_size < self.window_size:
                # Remove oldest entries if new window is smaller
                if self.method == "ContinuousMedian":
                    self.premiums = self.premiums[-new_window_size:]
                else:
                    self.premiums = self.premiums[-new_window_size - 1 :]
            self.window_size = new_window_size

    def get_median_return_percentage(self, strategy):
        """
        Calculate the median return percentage for the given strategy.

        Args:
            strategy (OptionStrategy): The option strategy to calculate the return percentage for.

        Returns:
            float: The median return percentage or direct return if no method specified
        """
        current_return = strategy.return_percentage()

        if not hasattr(self, "method") or not self.method:  # Handle empty/null method case
            strategy.filter_return_percentage = current_return
            strategy.filter_pl = strategy.total_pl()
            strategy.premium_log = self.premiums.copy()
            return current_return

        if self.premiums == []:  # take care of first update when adding spread
            self.add_premium(0)
        self.add_premium(current_return)
        median_return = self.get_median()
        strategy.premium_log = self.premiums.copy()
        strategy.filter_return_percentage = median_return
        strategy.filter_pl = (
            strategy.entry_net_premium * strategy.contracts
        ) * strategy.filter_return_percentage
        return median_return


class PremiumFilter:
    """
    Class for detecting outliers in premium values using HampelFilter. Updates strategy's filtered PL
    and return percentage while maintaining premium history.

    Attributes:
        window_size (int): The size of the rolling window.
        filter (HampelFilterNumpy): Hampel filter for outlier detection.
        premiums (list): List of raw premiums.
    """

    def __init__(self, window_size=5, n_sigma=3, k=1.4826, max_iterations=5):
        """
        Initialize the PremiumFilter.

        Args:
            window_size (int): Size of the rolling window for outlier detection.
            n_sigma (float): Number of standard deviations for outlier threshold.
            k (float): Scale factor for MAD calculation.
            max_iterations (int): Maximum iterations for Hampel filter convergence.
        """
        self.window_size = window_size
        self.filter = HampelFilterNumpy(
            window_size=window_size,
            n_sigma=n_sigma,
            k=k,
            max_iterations=max_iterations,
            replace_with_na=True
        )
        self.premiums = []

    def add_premium(self, mark: float):
        """
        Add a new premium to the rolling window.

        Args:
            mark (float): The new premium mark to add
        """
        self.premiums.append(mark)
        if len(self.premiums) > self.window_size + 1:
            self.premiums.pop(0)

    def check_outlier(self, strategy) -> bool:
        """
        Check if current premium's return percentage is an outlier. Updates strategy's
        filtered metrics with the cleaned values.

        Args:
            strategy (OptionStrategy): The option strategy to check

        Returns:
            bool: True if NOT an outlier (valid value), False if outlier
        """
        if hasattr(strategy, "premium_log"):
            self.premiums = strategy.premium_log.copy()
        else:
            self.premiums = []
            strategy.premium_log = []
            
        current_return = strategy.return_percentage()
        self.add_premium(current_return)

        if len(self.premiums) < self.filter.window_size + 1:
            # Not enough data yet - assume valid
            strategy.filter_return_percentage = current_return
            strategy.filter_pl = strategy.total_pl()
            strategy.premium_log = self.premiums.copy()
            return True

        # Apply Hampel filter (outliers will be replaced with NaNs)
        filtered_returns = self.filter.fit_transform(np.array([self.premiums]))
        
        # Check if last value is NaN (indicates outlier)
        is_valid = not np.isnan(filtered_returns[0][-1])
        
        # Store results in strategy
        strategy.premium_log = self.premiums.copy()
        strategy.filter_return_percentage = filtered_returns[0][-1] if is_valid else filtered_returns[0][-2]
        strategy.filter_pl = strategy.entry_net_premium * strategy.contracts * strategy.filter_return_percentage
        
        return is_valid


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
            profit_target (float): The profit target percentage
            window_size (int): Window size for median calculation (default: 10)
            method (str): Median calculation method ('HampelFilter' or 'ContinuousMedian') (default: 'HampelFilter')
            **kwargs: Additional keyword arguments for future extension
        """
        self.profit_target = profit_target
        self.median_calculator = MedianCalculator(
            kwargs.get("window_size", 10), kwargs.get("method", "HampelFilter"), replace_with_na=kwargs.get("replace_with_na", True)
        )
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
        if "window_size" in kwargs:
            self.median_calculator.update(window_size=kwargs["window_size"])

    def should_exit(
        self,
        strategy,
        current_time: Union[datetime, str, pd.Timestamp],
        option_chain_df: pd.DataFrame,
    ) -> bool:
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
        current_median_return = self.median_calculator.get_median_return_percentage(
            strategy
        )
        logger.debug(f"Current median return: {current_median_return}")
        return (
            current_return >= self.profit_target
            and current_median_return >= self.profit_target
        )


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
            stop_loss (float): The stop loss percentage
            window_size (int): Window size for median calculation (default: 10)
            method (str): Median calculation method ('HampelFilter' or 'ContinuousMedian') (default: 'HampelFilter')
            **kwargs: Additional keyword arguments for future extension
        """
        self.stop_loss = stop_loss
        self.median_calculator = MedianCalculator(
            kwargs.get("window_size", 10), kwargs.get("method", "HampelFilter"), replace_with_na=kwargs.get("replace_with_na", False)
        )
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
        if "window_size" in kwargs:
            self.median_calculator.update(window_size=kwargs["window_size"])

    def should_exit(
        self,
        strategy,
        current_time: Union[datetime, str, pd.Timestamp],
        option_chain_df: pd.DataFrame,
    ) -> bool:
        """
        Check if the stop loss is met.

        Args:
            strategy (OptionStrategy): The option strategy to check.
            current_time (datetime): The current time for evaluation.
            option_chain_df (pd.DataFrame): The updated option chain data.

        Returns:
            bool: True if the stop loss is met, False otherwise.
        """
        current_median_return = self.median_calculator.get_median_return_percentage(
            strategy
        )
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
        exit_time_before_expiration = pd.Timedelta(
            kwargs.get("exit_time_before_expiration", self.exit_time_before_expiration)
        )
        self.exit_time_before_expiration = exit_time_before_expiration

    def should_exit(
        self,
        strategy,
        current_time: Union[datetime, str, pd.Timestamp],
        option_chain_df: pd.DataFrame,
    ) -> bool:
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
        expiration_time = pd.Timestamp(strategy.legs[0].expiration).replace(
            hour=16, minute=0, second=0, microsecond=0
        )
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
            trigger (float): The trigger percentage for the trailing stop
            stop_loss (float): The stop loss percentage
            window_size (int): Window size for median calculation (default: 10)
            method (str): Median calculation method ('HampelFilter' or 'ContinuousMedian') (default: 'HampelFilter')
            exit_upon_positive_return (bool): Whether to exit only if return is positive (default: False)
            **kwargs: Additional keyword arguments that will be set as attributes
        """
        self.trigger = trigger
        self.stop_loss = stop_loss
        self.median_window = kwargs.get("window_size", 10)
        # self.median_calculator = MedianCalculator(self.median_window)
        
        self.highest_return = 0
        self.kwargs = kwargs
        self.median_method = kwargs.get("method", "HampelFilter")
        self.median_calculator = MedianCalculator(
            window_size=self.median_window, method=self.median_method, n_sigma=self.kwargs.get("n_sigma", 3), k=self.kwargs.get("k", 1.4826), max_iterations=self.kwargs.get("max_iterations", 5), replace_with_na=kwargs.get("replace_with_na", False)
        )

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
        trigger = float(kwargs.get("trigger", self.trigger))
        stop_loss = float(kwargs.get("stop_loss", self.stop_loss))
        self.trigger = trigger
        self.stop_loss = stop_loss
        if "window_size" in kwargs:
            self.median_calculator.update(window_size=kwargs["window_size"])

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

        current_median_return = self.median_calculator.get_median_return_percentage(
            strategy
        )
        current_return = strategy.return_percentage()
        self.highest_return = max(self.highest_return, current_median_return)

        if self.highest_return >= self.trigger:
            main_condition = (
                self.highest_return - current_return
            ) >= self.stop_loss and (
                self.highest_return - current_median_return
            ) >= self.stop_loss

            if self.kwargs.get("exit_upon_positive_return", False):
                return main_condition
            else:
                return main_condition and current_return > 0

        return False


class CompositeExitCondition(ExitConditionChecker):
    """
    Composite exit condition that combines multiple exit conditions.

    Attributes:
        conditions (List[ExitConditionChecker]): List of exit conditions to combine.
        logical_operations (List[str]): List of logical operations to combine the conditions ('AND' or 'OR').
    """

    def __init__(
        self,
        conditions: List[ExitConditionChecker],
        logical_operations: List[str] = ["AND"],
    ):
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

    def should_exit(
        self,
        strategy,
        current_time: Union[datetime, str, pd.Timestamp],
        option_chain_df: pd.DataFrame,
    ) -> bool:
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
            raise ValueError(
                "The number of logical operations must be one less than the number of conditions."
            )

        results = [
            condition.should_exit(strategy, current_time, option_chain_df)
            for condition in self.conditions
        ]
        combined_result = results[0]

        for i, operation in enumerate(self.logical_operations):
            if operation == "AND":
                combined_result = combined_result and results[i + 1]
            elif operation == "OR":
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

    def __init__(
        self,
        profit_target: float = 40,
        exit_time_before_expiration: pd.Timedelta = pd.Timedelta(minutes=15),
        **kwargs,
    ):
        """
        Initialize the DefaultExitCondition.

        Args:
            profit_target (float): The profit target percentage
            exit_time_before_expiration (pd.Timedelta): The time before expiration to exit the trade
            window_size (int): Window size for median calculation (default: 10)
            method (str): Median calculation method ('HampelFilter' or 'ContinuousMedian') (default: 'HampelFilter')
            **kwargs: Additional keyword arguments for future extension
        """
        profit_target_condition = ProfitTargetCondition(
            profit_target=profit_target,
            window_size=kwargs.get("window_size", 10),
            method=kwargs.get("method", "HampelFilter"),
            replace_with_na=kwargs.get("replace_with_na", True),
        )
        time_based_condition = TimeBasedCondition(
            exit_time_before_expiration=exit_time_before_expiration
        )
        self.composite_condition = CompositeExitCondition(
            conditions=[profit_target_condition, time_based_condition],
            logical_operations=["OR"],
        )
        self.__dict__.update(self.composite_condition.__dict__)

    def __repr__(self):
        """
        Return a string representation of the default exit condition.

        Returns:
            str: String representation of the default exit condition.
        """
        return (
            f"{self.__class__.__name__}(composite_condition={self.composite_condition}): profit target={self.profit_target}, exit_time_before_expiration={self.exit_time_before_expiration}"
        )

    def should_exit(
        self,
        strategy,
        current_time: Union[datetime, str, pd.Timestamp],
        option_chain_df: pd.DataFrame,
    ) -> bool:
        """
        Check if the default exit condition is met.

        Args:
            strategy (OptionStrategy): The option strategy to check.
            current_time (datetime): The current time for evaluation.
            option_chain_df (pd.DataFrame): The updated option chain data.

        Returns:
            bool: True if the default exit condition is met, False otherwise.
        """
        return self.composite_condition.should_exit(
            strategy, current_time, option_chain_df
        )

    def update(self, **kwargs):
        """
        Update the attributes of the default exit condition.

        Args:
            **kwargs: Keyword arguments for the attributes to update.
        """
        self.composite_condition.update(**kwargs)
        self.__dict__.update(self.composite_condition.__dict__)


class ProfitAndTriggeredTrailingStopExitCondition(ExitConditionChecker):

    def __init__(
        self,
        profit_target: float = 80,
        trigger: float = 40,
        stop_loss: float = 15,
        exit_time_before_expiration: pd.Timedelta = pd.Timedelta("1 day"),
        **kwargs,
    ):

        profit_target_condition = ProfitTargetCondition(
            profit_target=profit_target, window_size=kwargs.get("window_size", 10), method=kwargs.get("filter_method", "HampelFilter")
        )

        tsl_condition = TrailingStopCondition(
            trigger=trigger,
            stop_loss=stop_loss,
            window_size=kwargs.get("window_size", 10),
            method=kwargs.get("filter_method", "HampelFilter"),
            exit_upon_positive_return=kwargs.get("exit_upon_positive_return", False),
        )

        time_based_condition = TimeBasedCondition(
            exit_time_before_expiration=exit_time_before_expiration
        )

        self.composite_condition = CompositeExitCondition(
            conditions=[tsl_condition, time_based_condition, profit_target_condition],
            logical_operations=["OR", "OR"],
        )
        self.__dict__.update(self.composite_condition.__dict__)

    def __repr__(self):
        """
        Return a string representation of the profit-and-triggered-trailing-stop exit condition.

        Returns:
            str: String representation of the profit-and-triggered-trailing-stop exit condition.
        """
        return (
            f"{self.__class__.__name__}(composite_condition={self.composite_condition}): profit target={self.profit_target}, trigger={self.trigger}, stop_loss={self.stop_loss}, exit_time_before_expiration={self.exit_time_before_expiration}"
        )

    def should_exit(
        self,
        strategy,
        current_time: Union[datetime, str, pd.Timestamp],
        option_chain_df: pd.DataFrame,
    ) -> bool:
        """
        Check if the default exit condition is met.

        Args:
            strategy (OptionStrategy): The option strategy to check.
            current_time (datetime): The current time for evaluation.
            option_chain_df (pd.DataFrame): The updated option chain data.

        Returns:
            bool: True if the default exit condition is met, False otherwise.
        """
        return self.composite_condition.should_exit(
            strategy, current_time, option_chain_df
        )

    def update(self, **kwargs):
        """
        Update the attributes of the default exit condition.

        Args:
            **kwargs: Keyword arguments for the attributes to update.
        """
        self.composite_condition.update(**kwargs)
        self.__dict__.update(self.composite_condition.__dict__)
