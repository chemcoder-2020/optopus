from abc import ABC, abstractmethod
import datetime
import pandas as pd
from typing import Union, TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from .option_manager import OptionBacktester

class ExternalEntryConditionChecker(ABC):
    """
    Abstract base class for external entry condition checkers.
    These conditions can be used alongside standard EntryConditionCheckers
    to implement additional custom logic for trade entry decisions.

    Method:
        should_enter(strategy, manager, time) -> bool:
            Check if the external entry conditions are met for the option strategy.
    """
    @abstractmethod
    def should_enter(self, strategy, manager: 'OptionBacktester', time: Union[datetime, str, pd.Timestamp]) -> bool:
        """
        Check if the external entry conditions are met.

        Args:
            strategy: The option strategy being evaluated
            manager: The option backtester/manager instance
            time: The current time of evaluation

        Returns:
            bool: True if external conditions are met, False otherwise
        """
        pass

    def __repr__(self) -> str:
        """Return string representation of the condition checker"""
        return f"{self.__class__.__name__}()"
