from abc import ABC, abstractmethod
import datetime
import pandas as pd
from typing import Union
from .option_spread import OptionStrategy

class ExitConditionChecker(ABC):
    """
    Abstract base class for exit condition checkers.

    Methods:
        check_exit_conditions(current_time: datetime, option_chain_df: pd.DataFrame) -> bool:
            Check if the exit conditions are met for the option strategy.
    """

    @abstractmethod
    def should_exit(self, strategy: OptionStrategy, current_time: Union[datetime, str, pd.Timestamp], option_chain_df: pd.DataFrame) -> bool:
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

    def should_exit(self, strategy: OptionStrategy, current_time: Union[datetime, str, pd.Timestamp], option_chain_df: pd.DataFrame) -> bool:
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
