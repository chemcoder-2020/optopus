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
        self.kwargs = kwargs  # Store kwargs for indicator function

    def should_enter(self, strategy, manager, time: pd.Timestamp) -> bool:
        hist_data = manager.context["historical_data"]
        
        # Compute indicator series with kwargs
        indicator_series = self.indicator(hist_data, **self.kwargs)
        
        # Calculate rolling indicators
        short_term = indicator_series.rolling(self.lag1).mean()
        long_term = indicator_series.rolling(self.lag2).mean()
        
        # Check if we have enough data for requested indices
        if len(short_term) < abs(self.indicator1_index) or len(long_term) < abs(self.indicator2_index):
            return False
            
        # Get values at specified positions (supports negative indexing)
        short_value = short_term.iloc[self.indicator1_index]
        long_value = long_term.iloc[self.indicator2_index]
        
        # Compare values with debug logging
        result = short_value > long_value
        logger.debug(f"IndicatorStateCheck: {short_value:.4f} > {long_value:.4f} = {result} " +
                    f"(lag={self.lag1}/{self.lag2}, indices={self.indicator1_index}/{self.indicator2_index})")
        return result

