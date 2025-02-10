import numpy as np
import pandas as pd
from typing import Union
from .base_metric import BaseMetric


class WinRate(BaseMetric):
    """Calculates win rate percentage from boolean success array"""

    def calculate(self, successes: np.ndarray) -> dict:
        if successes.size == 0:
            return {"win_rate": 0.0}

        win_rate = np.mean(successes.astype(float))
        return {"win_rate": float(win_rate)}


class ProfitFactor(BaseMetric):
    """Calculates profit factor (gross profits / gross losses)"""

    def calculate(self, returns: Union[pd.Series, np.ndarray]) -> dict:
        returns = returns.copy()

        profits = returns[returns > 0].sum()
        losses = -returns[returns < 0].sum()
        return {"profit_factor": self.safe_ratio(profits, losses)}
