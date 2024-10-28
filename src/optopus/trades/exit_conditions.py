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
    def check_exit_conditions(self, strategy: OptionStrategy, current_time: Union[datetime, str, pd.Timestamp], option_chain_df: pd.DataFrame) -> bool:
        """
        Check if the exit conditions are met for the option strategy.

        Args:
            current_time (datetime): The current time for evaluation.
            option_chain_df (pd.DataFrame): The updated option chain data.

        Returns:
            bool: True if the exit conditions are met, False otherwise.
        """
        pass
