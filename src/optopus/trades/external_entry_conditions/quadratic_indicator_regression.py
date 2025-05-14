from .base import BaseComponent
import pandas as pd
import numpy as np
from loguru import logger


class QuadraticIndicatorRegressionCheck(BaseComponent):
    def __init__(
        self, lag=14, indicator=None, selected_indicator_column=None, **kwargs
    ):
        self.lag = lag
        self.indicator = indicator
        self.selected_indicator_column = selected_indicator_column
        self.kwargs = kwargs

    def should_enter(self, strategy, manager, time: pd.Timestamp) -> bool:
        historical_data = manager.context["historical_data"]
        if len(historical_data) < self.lag:
            return False

        if self.indicator is not None:
            indicator_series = self.indicator(
                high=historical_data["high"],
                low=historical_data["low"],
                close=historical_data["close"],
                open=historical_data["open"],
                volume=historical_data["volume"],
                **self.kwargs,
            )
            if isinstance(indicator_series, pd.Series):
                pass
            elif isinstance(indicator_series, pd.DataFrame):
                if self.selected_indicator_column is not None:
                    indicator_series = indicator_series[self.selected_indicator_column]
                else:
                    raise ValueError(
                        "QuadraticIndicatorRegressionCheck: selected_indicator_column is not specified"
                    )
            else:
                raise ValueError(
                    "QuadraticIndicatorRegressionCheck: indicator is not a pandas series or dataframe"
                )
        else:
            raise ValueError(
                "QuadraticIndicatorRegressionCheck: indicator is not specified"
            )

        recent_data = indicator_series.iloc[-self.lag :]
        if recent_data.isna().any():
            return False

        x = np.arange(self.lag)
        y = recent_data.values

        # Fit quadratic regression (degree=2)
        coefficients = np.polyfit(x, y, 2)
        a, b, c = coefficients  # ax² + bx + c
        p = np.polynomial.polynomial.Polynomial([c, b, a])
        result = p(x[-1]) - p(x[-2])
        logger.info(f"QuadraticIndicatorRegressionCheck: {result}")
        manager.context["indicators"].update(
            {f"QuadraticIndicatorRegression_{self.lag}": result}
        )

        if result > 0:
            logger.info("QuadraticIndicatorRegressionCheck passed.")
        return result > 0

    def __repr__(self):
        return f"QuadraticIndicatorRegressionCheck(lag={self.lag})"
