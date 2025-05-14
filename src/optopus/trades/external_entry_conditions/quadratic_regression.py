from .base import BaseComponent
import pandas as pd
import numpy as np
from loguru import logger


class QuadraticRegressionCheck(BaseComponent):
    def __init__(self, lag=14):
        self.lag = lag

    def should_enter(self, strategy, manager, time: pd.Timestamp) -> bool:
        historical_data = manager.context["historical_data"]
        if len(historical_data) < self.lag:
            return False

        recent_data = historical_data["close"].iloc[-self.lag :]
        x = np.arange(self.lag)
        y = recent_data.values

        # Fit quadratic regression (degree=2)
        coefficients = np.polyfit(x, y, 2)
        a, b, c = coefficients  # ax² + bx + c
        p = np.polynomial.polynomial.Polynomial([c, b, a])
        result = p(x[-1]) - p(x[-2])
        logger.info(f"QuadraticRegressionCheck: {result}")
        manager.context["indicators"].update(
            {f"QuadraticRegression_{self.lag}": result}
        )

        if result > 0:
            logger.info("QuadraticRegressionCheck passed.")
        return result > 0

    def __repr__(self):
        return f"QuadraticRegressionCheck(lag={self.lag})"
