from abc import ABC, abstractmethod
import datetime
import pandas as pd
from typing import Union, List

class ExitConditionChecker(ABC):
    """
    Abstract base class for exit condition checkers.

    Methods:
        check_exit_conditions(current_time: datetime, option_chain_df: pd.DataFrame) -> bool:
            Check if the exit conditions are met for the option strategy.
    """

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    @abstractmethod
    def should_exit(self, strategy, current_time: Union[datetime, str, pd.Timestamp], option_chain_df: pd.DataFrame) -> bool:
        """
        Check if the exit conditions are met for the option strategy.

        Args:
            current_time (datetime): The current time for evaluation.
            option_chain_df (pd.DataFrame): The updated option chain data.

        Returns:
            bool: True if the exit conditions are met, False otherwise.
        """
        pass

class ProfitTargetCondition(ExitConditionChecker):
    """
    Exit condition based on a profit target.

    Attributes:
        profit_target (float): The profit target percentage.
    """

    def __init__(self, profit_target: float):
        """
        Initialize the ProfitTargetCondition.

        Args:
            profit_target (float): The profit target percentage.
        """
        self.profit_target = profit_target

    def __repr__(self):
        return f"{self.__class__.__name__}(profit_target={self.profit_target})"

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
        return current_return >= self.profit_target

class StopLossCondition(ExitConditionChecker):
    """
    Exit condition based on a stop loss.

    Attributes:
        stop_loss (float): The stop loss percentage.
    """

    def __init__(self, stop_loss: float):
        """
        Initialize the StopLossCondition.

        Args:
            stop_loss (float): The stop loss percentage.
        """
        self.stop_loss = stop_loss

    def __repr__(self):
        return f"{self.__class__.__name__}(stop_loss={self.stop_loss})"

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
        current_return = strategy.return_percentage()
        return current_return <= -self.stop_loss

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
        return f"{self.__class__.__name__}(exit_time_before_expiration={self.exit_time_before_expiration})"

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
    """

    def __init__(self, trigger: float, stop_loss: float):
        """
        Initialize the TrailingStopCondition.

        Args:
            trigger (float): The trigger percentage for the trailing stop.
            stop_loss (float): The stop loss percentage.
        """
        self.trigger = trigger
        self.stop_loss = stop_loss

    def __repr__(self):
        return f"{self.__class__.__name__}(trigger={self.trigger}, stop_loss={self.stop_loss})"

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
        current_return = strategy.return_percentage()
        highest_return = strategy.highest_return

        if highest_return >= self.trigger:
            return (highest_return - current_return) >= self.stop_loss

        return False

class CompositeExitCondition(ExitConditionChecker):
    """
    Composite exit condition that combines multiple exit conditions.

    Attributes:
        conditions (List[ExitConditionChecker]): List of exit conditions to combine.
        logical_operation (str): The logical operation to combine the conditions ('AND' or 'OR').
        attributes (dict): A dictionary containing the attributes of each condition.
    """

    def __init__(self, conditions: List[ExitConditionChecker], logical_operation: str = 'AND'):
        """
        Initialize the CompositeExitCondition.

        Args:
            conditions (List[ExitConditionChecker]): List of exit conditions to combine.
            logical_operation (str): The logical operation to combine the conditions ('AND' or 'OR').
        """
        self.conditions = conditions
        self.logical_operation = logical_operation
        for condition in conditions:
            self.__dict__.update(condition.__dict__)

    def __repr__(self):
        return f"{self.__class__.__name__}(conditions={self.conditions}, logical_operation='{self.logical_operation}')"

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
        results = [condition.should_exit(strategy, current_time, option_chain_df) for condition in self.conditions]

        if self.logical_operation == 'AND':
            return all(results)
        elif self.logical_operation == 'OR':
            return any(results)
        else:
            raise ValueError("Logical operation must be 'AND' or 'OR'")


class DefaultExitCondition(ExitConditionChecker):
    """
    Default exit condition that combines a profit target and a time-based condition.

    Attributes:
        profit_target (float): The profit target percentage.
        exit_time_before_expiration (pd.Timedelta): The time before expiration to exit the trade.
    """

    def __init__(self, profit_target: float=40, exit_time_before_expiration: pd.Timedelta=pd.Timedelta(minutes=15)):
        """
        Initialize the DefaultExitCondition.

        Args:
            profit_target (float): The profit target percentage.
            exit_time_before_expiration (pd.Timedelta): The time before expiration to exit the trade.
        """
        profit_target_condition = ProfitTargetCondition(profit_target=profit_target)
        time_based_condition = TimeBasedCondition(exit_time_before_expiration=exit_time_before_expiration)
        self.composite_condition = CompositeExitCondition(
            conditions=[profit_target_condition, time_based_condition],
            logical_operation='OR'
        )
        self.__dict__.update(self.composite_condition.__dict__)

    def __repr__(self):
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
