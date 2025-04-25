from .base import BaseComponent
import pandas as pd
import numpy as np
from loguru import logger


class CubicRegressionCheck(BaseComponent):
    def __init__(self, lag=14):
        self.lag = lag

    def should_enter(self, strategy, manager, time: pd.Timestamp) -> bool:
        historical_data = manager.context["historical_data"]
        if len(historical_data) < self.lag:
            return False

        recent_data = historical_data["close"].iloc[-self.lag :]
        x = np.arange(self.lag)
        y = recent_data.values

        # Fit Cubic regression (degree=3)
        coefficients = np.polyfit(x, y, 3)
        a, b, c, d = coefficients  # ax^3 + bx² + cx + d
        p = np.polynomial.polynomial.Polynomial([d, c, b, a])
        result = p(x[-1]) - p(x[-2])
        logger.info(f"CubicRegressionCheck: {result}")
        manager.context["indicators"].update({f"CubicRegression_{self.lag}": result})

        if result > 0:
            logger.info("CubicRegressionCheck passed.")
        return result > 0

    def __repr__(self):
        return f"CubicRegressionCheck(lag={self.lag})"
