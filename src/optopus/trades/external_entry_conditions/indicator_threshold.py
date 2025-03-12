from .base import BaseComponent
import pandas as pd
from loguru import logger


class IndicatorThresholdCheck(BaseComponent):
    """Check if indicator is above value.
    Args:
        indicator: Indicator function. Params need to be passed: high, low, close, open, volume, length
        target: Target value
        lag: Short-term indicator lag
        indicator_index: Index of short-term indicator value
    Kwargs:
        kwargs: Kwargs for indicator function

    Returns True if:
        - Indicator is above value
    """

    def __init__(
        self,
        indicator,
        target,
        lag,
        indicator_index=-1,
        **kwargs,
    ):
        self.indicator = indicator
        self.target = target
        self.lag = lag
        self.indicator_index = indicator_index
        self.kwargs = kwargs  # Store kwargs for indicator function

    def should_enter(self, strategy, manager, time: pd.Timestamp) -> bool:
        hist_data = manager.context["historical_data"]
        logger.info(
            "IndicatorThresholdCheck: Starting indicator check with lag: {}".format(
                self.lag
            )
        )
        try:
            if self.indicator in ['close', 'open', 'high', 'low', 'volume']:
                indicator_series = hist_data[self.indicator].copy()
            else:
                if self.lag > 1:
                    indicator_series = self.indicator(
                        high=hist_data["high"],
                        low=hist_data["low"],
                        close=hist_data["close"],
                        open=hist_data["open"],
                        volume=hist_data["volume"],
                        length=self.lag,
                        **self.kwargs,
                    )
                else:
                    indicator_series = hist_data["close"].copy()

            if len(indicator_series) < abs(self.indicator_index):
                logger.warning(
                    "IndicatorThresholdCheck: Not enough data for indices. indicator_series length: {}, indicator_index: {}".format(
                        len(indicator_series),
                        self.indicator_index,
                    )
                )
                return False

            indicator_value = indicator_series.iloc[self.indicator_index]

            if self.indicator in ['close', 'open', 'high', 'low', 'volume']:
                manager.context["indicators"][
                    f"{self.indicator}_{self.indicator_index}"
                ] = indicator_value
            else:
                manager.context["indicators"][
                    f"{self.indicator.__name__}_{self.lag}_{self.indicator_index}"
                ] = indicator_value

            logger.info(
                "IndicatorThresholdCheck: Comparing indicator_value: {} with target: {}".format(
                    indicator_value, self.target
                )
            )
            if isinstance(self.target, (int, float)):
                result = indicator_value > self.target
            elif isinstance(self.target, tuple):
                result = indicator_value > self.target[0] and indicator_value < self.target[1]
            else:
                logger.error(
                    "IndicatorThresholdCheck: Invalid target type: {}".format(
                        type(self.target)
                    )
                )
                result = False
            logger.info("IndicatorThresholdCheck: Result is {}".format(result))
        except Exception as e:
            logger.error(
                f"IndicatorThresholdCheck: Error checking indicator threshold: {str(e)}"
            )
            result = False
        return result
