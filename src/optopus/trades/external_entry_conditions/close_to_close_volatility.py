from .base import BaseComponent
import pandas as pd
import numpy as np
from loguru import logger


class CloseToCloseVolatilityDecreaseCheck(BaseComponent):
    """Check if close-to-close volatility is decreasing.

    Returns True if:
    - Close-to-close volatility shows decrease from previous to current window
    """

    def __init__(self, lag=21, zero_drift=True):
        self.lag = lag
        self.zero_drift = zero_drift

    def should_enter(self, strategy, manager, time: pd.Timestamp) -> bool:
        hist_data = manager.context["historical_data"]
        required_periods = self.lag + (1 if self.zero_drift else 2)
        if len(hist_data) < required_periods:
            return False

        close_prices = hist_data["close"]

        if self.zero_drift:
            # Zero drift assumption (original Parkinson)
            log_returns = np.log(close_prices / close_prices.shift(1))
            volatility = log_returns.rolling(self.lag).var(ddof=0)
        else:
            # Non-zero drift adjustment (Yang-Zhang style)
            log_returns = np.log(close_prices / close_prices.shift(1))
            mu_cc = log_returns.rolling(self.lag - 1).mean()
            adj_returns = log_returns - mu_cc
            volatility = adj_returns.rolling(self.lag - 1).var(ddof=1)

        # Get current and previous values
        vol_current = volatility.iloc[-1]
        vol_prev = volatility.iloc[-2]

        manager.context["indicators"].update({f"c2c_vol_{self.lag}": vol_current})

        logger.info(
            f"Previous Close-to-Close Volatility: {vol_prev:.4f}; Current Close-to-Close Volatility: {vol_current:.4f}."
        )
        if vol_current < vol_prev:
            logger.info("Passed Close-to-Close Volatility Decrease Check")
        else:
            logger.info("Failed Close-to-Close Volatility Decrease Check")

        return vol_current < vol_prev

    def __repr__(self):
        return f"CloseToCloseVolatilityDecreaseCheck(lag={self.lag}, zero_drift={self.zero_drift})"
