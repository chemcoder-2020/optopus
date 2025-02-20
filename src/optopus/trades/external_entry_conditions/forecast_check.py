from .base import BaseComponent
from statsforecast import StatsForecast
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
        trend_direction: str = "upward",
        **kwargs,
    ):
        self.models = models
        self.trend_direction = trend_direction.lower()
        self.kwargs = kwargs

    def should_enter(self, strategy, manager, time: pd.Timestamp) -> bool:
        hist_data = manager.context["historical_data"]
        
        if len(hist_data) < 2:  # Need at least 2 data points for forecasting
            return False

        # Prepare data for statsforecast
        df = hist_data[["CLOSE"]].reset_index()
        df.columns = ["ds", "y"]
        
        # Initialize and fit statsforecast
        sf = StatsForecast(
            models=self.models,
            freq=pd.infer_freq(hist_data.index) or "D",
            **self.kwargs
        )
        
        # Generate forecast
        forecast = sf.fit_predict(df, h=1)
        
        # Get last known value and forecasted value
        last_close = df["y"].iloc[-1]
        forecasted_value = forecast.iloc[0]["yhat"]
        
        # Determine trend direction
        detected_trend = "upward" if forecasted_value > last_close else "downward"
        
        # Compare with preferred trend
        result = detected_trend == self.trend_direction
        logger.debug(
            f"StatsForecastCheck: {detected_trend} trend detected "
            f"(Forecast: {forecasted_value:.2f} vs Last Close: {last_close:.2f})"
        )
        return result
