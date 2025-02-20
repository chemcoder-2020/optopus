from .base import BaseComponent
import pandas as pd
from loguru import logger


class StatsForecastCheck(BaseComponent):
    """Check if statsforecast forecasts a preferred trend

    Returns True if:
    - Preferred trend is indicated by statsforecast
    """

    def __init__(
        self,
        models,
        **kwargs,
    ):
        
        self.kwargs = kwargs  # Store kwargs for statsforecast

    def should_enter(self, strategy, manager, time: pd.Timestamp) -> bool:
        from statsforecast import StatsForecast
        from statsforecast.models import (
            ARIMA,
            SeasonalExponentialSmoothingOptimized,
            RandomWalkWithDrift,
            AutoARIMA,
            AutoCES,
        )
        from sktime.transformations.series.boxcox import LogTransformer
        from sktime.transformations.series.detrend import Detrender
        from sktime.transformations.series.difference import Differencer
        from sktime.forecasting.trend import TrendForecaster
        from sklearn.linear_model import Ridge
        from loguru import logger

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
