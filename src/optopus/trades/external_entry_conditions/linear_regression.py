from .base import BaseComponent
import pandas as pd
from loguru import logger


class LinearRegressionCheck(BaseComponent):
    def __init__(self, lag=14):
        self.lag = lag

    def should_enter(self, strategy, manager, time: pd.Timestamp) -> bool:
        from sklearn.linear_model import LinearRegression
        import numpy as np

        historical_data = manager.context["historical_data"]
        if len(historical_data) < self.lag:
            return False

        recent_data = historical_data["close"].iloc[-self.lag :]
        X = np.arange(self.lag).reshape(-1, 1)
        y = recent_data.values
        model = LinearRegression()
        model.fit(X, y)
        if model.coef_[0] > 0:
            logger.info("LinearRegressionCheck passed. Slope > 0")

        manager.context["indicators"].update({f"LinearRegression_{self.lag}": model.coef_[0]})

        return model.coef_[0] > 0

    def __repr__(self):
        return f"LinearRegressionCheck(lag={self.lag}"
