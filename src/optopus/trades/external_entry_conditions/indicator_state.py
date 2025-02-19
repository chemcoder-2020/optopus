from .base import BaseComponent
import pandas as pd
from loguru import logger


class IndicatorStateCheck(BaseComponent):
    """Check if short-term indicator is above long-term indicator.

    Returns True if:
    - Short-term indicator is above long-term indicator
    """

    def __init__(self, indicator, lag1=50, lag2=200, indicator1_index=-1, indicator2_index=-1, **kwargs):
        self.indicator = indicator
        self.lag1 = lag1
        self.lag2 = lag2
        self.indicator1_index = indicator1_index
        self.indicator2_index = indicator2_index

    def should_enter(self, strategy, manager, time: pd.Timestamp) -> bool:
        hist_data = manager.context["historical_data"]
        if len(hist_data) < self.lag + 1:
            return False
        
        # Compute the indicators


        # Get comparing values


        # Compare values and log results

        return

