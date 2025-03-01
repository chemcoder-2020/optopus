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
        model: str = "ARIMA",
        context: str = "historical_data",
        trend_direction: str = "upward",
        **kwargs,
    ):
        self.model = model
        self.context = context
        self.trend_direction = trend_direction.lower()
        self.kwargs = kwargs

    def should_enter(self, strategy, manager, time: pd.Timestamp) -> bool:
        from statsforecast import StatsForecast
        import statsforecast.models as sm
        from statsforecast.models import MSTL
        import importlib

        module = importlib.import_module("statsforecast.models")

        logger.debug(
            "StatsForecastCheck: Starting forecast check using model {} and context {}.".format(
                self.model, self.context
            )
        )
        if isinstance(manager.context[self.context], pd.Series):
            hist_data = manager.context[self.context].to_frame()
        else:
            hist_data = manager.context[self.context]

        if len(hist_data) < 2:  # Need at least 2 data points for forecasting
            logger.warning(
                "StatsForecastCheck: Not enough data for forecasting. Data count: {}".format(
                    len(hist_data)
                )
            )
            return False

        # Prepare data for statsforecast
        df = pd.DataFrame(
            {
                "ds": list(hist_data.index),
                "y": list(hist_data["close"].values),
                "unique_id": "1",
            }
        )

        # Initialize and fit statsforecast
        model = getattr(module, self.model)
        if self.context == "monthly_data":
            freq = "M"
            seasonal_lengths = [12, 4]
        else:
            freq = "B"
            seasonal_lengths = [7, 30]

        trend_forecaster = model(**self.kwargs)
        if self.kwargs.get("use_mstl", False):
            mstl_scheme = MSTL(seasonal_lengths, trend_forecaster=trend_forecaster)
            sf = StatsForecast(
                models=[mstl_scheme],
                freq=freq,
            )
        else:
            sf = StatsForecast(
                models=[trend_forecaster],
                freq=freq,
            )

        # Generate forecast
        if self.kwargs.get("use_mstl", False):
            forecast = sf.fit_predict(df=df, h=1)["MSTL"]
        else:
            forecast = sf.fit_predict(df=df, h=1)[sf.models[0].__str__()]

        # Get last known value and forecasted value
        last_close = df["y"].iloc[-1]
        forecasted_value = forecast.iloc[0]

        # Determine trend direction
        detected_trend = "upward" if forecasted_value > last_close else "downward"

        # Compare with preferred trend
        result = detected_trend == self.trend_direction
        logger.debug(
            f"StatsForecastCheck: {detected_trend} trend detected "
            f"(Forecast: {forecasted_value:.2f} vs Last Close: {last_close:.2f})"
        )
        return result


class VolatilityForecastCheck(BaseComponent):
    """Check if Volatility forecast is preferable

    Returns True if:
    - Volatility increase and False otherwise
    """

    def __init__(
        self,
        model: str = "StatsForecastGARCH",
        context: str = "monthly_data",
        **kwargs,
    ):
        self.model = model
        self.context = context
        self.kwargs = kwargs

    def should_enter(self, strategy, manager, time: pd.Timestamp) -> bool:
        from sktime.forecasting.arch import StatsForecastGARCH, StatsForecastARCH
        import numpy as np

        logger.debug(
            "VolatilityForecastCheck: Starting forecast check using model {} and context {}.".format(
                self.model, self.context
            )
        )
        hist_data = manager.context[self.context]

        if len(hist_data) < 2:  # Need at least 2 data points for forecasting
            logger.warning(
                "VolatilityForecastCheck: Not enough data for forecasting. Data count: {}".format(
                    len(hist_data)
                )
            )
            return False

        # Prepare data for statsforecast
        returns = np.log(hist_data["close"] / hist_data["close"].shift(1))

        # Initialize and fit model
        sf = eval(self.model)(**self.kwargs)
        sf.fit(y=returns.dropna())

        # Generate forecast
        forecasted_value = sf.predict(fh=[1])[0]

        # Preferred?
        result = forecasted_value > 0
        pos_return = "Positive" if result else "Negative"

        logger.debug(f"VolatilityForecastCheck: {pos_return} return forecasted ")
        return result
