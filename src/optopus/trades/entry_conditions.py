from abc import ABC, abstractmethod
import datetime
import pandas as pd
from typing import Union

class EntryConditionChecker(ABC):
    """
    Abstract base class for entry condition checkers.

    Methods:
        should_enter() -> bool:
            Check if the entry conditions are met for the option strategy.
    """
    @abstractmethod
    def should_enter(self, time: Union[datetime, str, pd.Timestamp]) -> bool:
        pass

class DefaultEntryCondition(EntryConditionChecker):
    """
    Default entry condition that always returns True.
    """
    def should_enter(self, time: Union[datetime, str, pd.Timestamp]) -> bool:
        return True
