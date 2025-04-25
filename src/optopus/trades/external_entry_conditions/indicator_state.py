from .base import BaseComponent
import pandas as pd
from loguru import logger


class IndicatorStateCheck(BaseComponent):
    """Check if short-term indicator is above long-term indicator.
    Args:
        indicator: Indicator function. Params need to be passed: high, low, close, open, volume, length
        lag1: Short-term indicator lag
        lag2: Long-term indicator lag
        indicator_index1: Index of short-term indicator value
        indicator_index2: Index of long-term indicator value
    Kwargs:
        kwargs: Kwargs for indicator function

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
        logger.info(
            "IndicatorStateCheck: Starting indicator check with lag1: {}, lag2: {}".format(
                self.lag1, self.lag2
            )
        )
        try:
            if self.lag1 > 1:
                indicator_series1 = self.indicator(
                    high=hist_data["high"],
                    low=hist_data["low"],
                    close=hist_data["close"],
                    open=hist_data["open"],
                    volume=hist_data["volume"],
                    length=self.lag1,
                    **self.kwargs,
                )
            else:
                indicator_series1 = hist_data["close"].copy()

            if self.lag2 > 1:
                indicator_series2 = self.indicator(
                    high=hist_data["high"],
                    low=hist_data["low"],
                    close=hist_data["close"],
                    open=hist_data["open"],
                    volume=hist_data["volume"],
                    length=self.lag2,
                    **self.kwargs,
                )
            else:
                indicator_series2 = hist_data["close"].copy()

            if len(indicator_series1) < abs(self.indicator_index1) or len(
                indicator_series2
            ) < abs(self.indicator_index2):
                logger.warning(
                    "IndicatorStateCheck: Not enough data for indices. indicator_series1 length: {}, indicator_index1: {}, indicator_series2 length: {}, indicator_index2: {}".format(
                        len(indicator_series1),
                        self.indicator_index1,
                        len(indicator_series2),
                        self.indicator_index2,
                    )
                )
                return False

            short_value = indicator_series1.iloc[self.indicator_index1]
            long_value = indicator_series2.iloc[self.indicator_index2]

            manager.context["indicators"].update(
                {f"{self.indicator.__name__}_{self.lag1}_{self.indicator_index1}": short_value, f"{self.indicator.__name__}_{self.lag2}_{self.indicator_index2}": long_value}
            )

            logger.info(
                "IndicatorStateCheck: Comparing short_value: {} with long_value: {}".format(
                    short_value, long_value
                )
            )
            result = short_value > long_value
            logger.info("IndicatorStateCheck: Result is {}".format(result))
        except Exception as e:
            logger.error(
                f"IndicatorStateCheck: Error checking indicator state: {str(e)}"
            )
            result = False
        return result
