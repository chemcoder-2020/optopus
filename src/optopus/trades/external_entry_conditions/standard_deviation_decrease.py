from .base import BaseComponent
import pandas as pd
from loguru import logger


class StdevVolatilityDecreaseCheck(BaseComponent):
    """Check if Stdev Volatility is decreasing.

    Returns True if:
    - Standard Deviation Volatility shows decrease from current to previous rolling window
    """

    def __init__(self, lag=21):
        self.lag = lag

    def should_enter(self, strategy, manager, time: pd.Timestamp) -> bool:

        hist_data = manager.context["historical_data"]
        if len(hist_data) < self.lag + 1:
            return False

        df = hist_data["close"]

        # Calculate volatility measures
        volatility = df.pct_change().rolling(self.lag).std()

        # Get current and previous values
        volatility_current = volatility.iloc[-1]
        volatility_prev = volatility.iloc[-2]

        logger.info(
            f"Previous Standard Dev Volatility: {volatility_prev:.4f}; Current Standard Dev Volatility: {volatility_current:.4f}."
        )
        if volatility_current < volatility_prev:
            logger.info("Passed Standard Dev Volatility Decrease Check")
        else:
            logger.info("Failed Standard Dev Volatility Decrease Check")

        # Check both measures decreased
        return volatility_current < volatility_prev

    def __repr__(self):
        return f"StdevVolatilityDecreaseCheck(lag={self.lag})"
