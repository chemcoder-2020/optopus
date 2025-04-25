from .base import BaseComponent
import pandas as pd
import numpy as np
from loguru import logger
from statsmodels.nonparametric.smoothers_lowess import lowess


def calculate_loess(series, frac=0.05, it=3):
    """Calculate LOESS smoothing with confidence interval"""
    x = np.arange(len(series))
    y = series.values
    smoothed = lowess(y, x, frac=frac, it=it, return_sorted=False)

    return smoothed


class LOESSCheck(BaseComponent):
    def __init__(self, frac=0.05, **kwargs):
        self.frac = frac
        self.kwargs = kwargs

    def should_enter(self, strategy, manager, time: pd.Timestamp) -> bool:
        historical_data = manager.context["historical_data"]

        recent_data = historical_data["close"].copy()
        smoothed = calculate_loess(recent_data, frac=self.frac, **self.kwargs)

        result = smoothed[-1] - smoothed[-2]
        logger.info(f"LOESSCheck: {result}")
        manager.context["indicators"].update({f"LOESS_{self.frac}": smoothed[-1]})

        if result > 0:
            logger.info("LOESSCheck passed.")
        return result > 0

    def __repr__(self):
        return f"LOESSCheck(frac={self.frac})"
