from .base import BaseComponent
import pandas as pd
import numpy as np
from loguru import logger


class CloseToCloseVolatilityDecreaseCheck(BaseComponent):
    """Check if close-to-close volatility is decreasing.

    Returns True if:
    - Close-to-close volatility shows decrease from previous to current window
    """

    def __init__(self, lag=21):
        self.lag = lag

    def should_enter(self, strategy, manager, time: pd.Timestamp) -> bool:
        hist_data = manager.context["historical_data"]
        if len(hist_data) < self.lag + 1:
            return False

        # Calculate close-to-close log returns
        close_prices = hist_data["close"]
        log_returns = np.log(close_prices / close_prices.shift(1))
        
        # Calculate rolling volatility (std dev of log returns)
        volatility = log_returns.rolling(self.lag).std()

        # Get current and previous values
        vol_current = volatility.iloc[-1]
        vol_prev = volatility.iloc[-2]

        logger.info(
            f"Previous Close-to-Close Volatility: {vol_prev:.4f}; Current Close-to-Close Volatility: {vol_current:.4f}."
        )
        if vol_current < vol_prev:
            logger.info("Passed Close-to-Close Volatility Decrease Check")
        else:
            logger.info("Failed Close-to-Close Volatility Decrease Check")

        return vol_current < vol_prev

    def __repr__(self):
        return f"CloseToCloseVolatilityDecreaseCheck(lag={self.lag})"
