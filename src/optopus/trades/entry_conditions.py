from abc import ABC, abstractmethod
import datetime
import pandas as pd

class EntryConditionChecker(ABC):
    """
    Abstract base class for entry condition checkers.

    Methods:
        should_enter(current_time: datetime, option_chain_df: pd.DataFrame) -> bool:
            Check if the entry conditions are met for the option strategy.
    """
    @abstractmethod
    def should_enter(self, current_time: datetime, option_chain_df: pd.DataFrame) -> bool:
        pass

class DefaultEntryCondition(EntryConditionChecker):
    """
    Default entry condition that always returns True.
    """
    def should_enter(self, current_time: datetime, option_chain_df: pd.DataFrame) -> bool:
        return True
