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
        
        # Parabole opens upwards if leading coefficient is positive
        if a > 0:
            logger.info("QuadraticRegressionCheck passed. Parabola opening upwards")
        return a > 0

    def __repr__(self):
        return f"QuadraticRegressionCheck(lag={self.lag})"
