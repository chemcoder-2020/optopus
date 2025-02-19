from .base import BaseComponent
import pandas as pd
from loguru import logger


class IndicatorStateCheck(BaseComponent):
    """Check if short-term indicator is above long-term indicator.

    Returns True if:
    - Short-term indicator is above long-term indicator
    """

    def __init__(
        self,
        indicator,
        lag1=50,
        lag2=200,
        indicator_index1=-1,
        indicator_index2=-1,
        **kwargs,
    ):
        self.indicator = indicator
        self.lag1 = lag1
        self.lag2 = lag2
        self.indicator_index1 = indicator_index1
        self.indicator_index2 = indicator_index2
        self.kwargs = kwargs  # Store kwargs for indicator function

    def should_enter(self, strategy, manager, time: pd.Timestamp) -> bool:
        hist_data = manager.context["historical_data"]

        # Compute indicator series with kwargs
        indicator_series1 = self.indicator(hist_data, self.lag1, **self.kwargs)
        indicator_series2 = self.indicator(hist_data, self.lag2, **self.kwargs)

        # Check if we have enough data for requested indices
        if len(indicator_series1) < abs(self.indicator_index1) or len(
            indicator_series2
        ) < abs(self.indicator_index2):
            return False

        # Get values at specified positions (supports negative indexing)
        short_value = indicator_series1.iloc[self.indicator_index1]
        long_value = indicator_series2.iloc[self.indicator_index2]

        # Compare values with debug logging
        result = short_value > long_value
        logger.debug(
            f"IndicatorStateCheck: {short_value:.4f} > {long_value:.4f} = {result} "
            + f"(lag={self.lag1}/{self.lag2}, indices={self.indicator1_index}/{self.indicator2_index})"
        )
        return result
