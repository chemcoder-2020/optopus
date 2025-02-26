from .base import BaseComponent
import pandas as pd
from loguru import logger


class RidgeCheck(BaseComponent):
    def __init__(self, lag=14):
        self.lag = lag

    def should_enter(self, strategy, manager, time: pd.Timestamp) -> bool:
        from sklearn.linear_model import Ridge
        import numpy as np

        historical_data = manager.context["historical_data"]
        if len(historical_data) < self.lag:
            return False

        recent_data = historical_data["close"].iloc[-self.lag :]
        X = np.arange(self.lag).reshape(-1, 1)
        y = recent_data.values
        model = Ridge()
        model.fit(X, y)
        # Require positive slope for upward trend
        y_pred = model.predict(X)
        if y_pred[-1] - y_pred[-2] > 0:
            logger.info(f"RidgeCheck passed. Current Value > Previous Value")
            return True
        return False

    def __repr__(self):
        return f"RidgeCheck(lag={self.lag})"
