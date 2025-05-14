from .base import BaseComponent
import pandas as pd
from loguru import logger


class ParkinsonVolatilityDecreaseCheck(BaseComponent):
    """Check if Parkinson Volatility is decreasing.

    Returns True if:
    - Both volatility measures show decrease from previous to current window
    """

    def __init__(self, lag=21):
        self.lag = lag

    def should_enter(self, strategy, manager, time: pd.Timestamp) -> bool:
        import numpy as np

        hist_data = manager.context["historical_data"]
        if len(hist_data) < self.lag + 1:
            return False

        df = hist_data[["high", "low", "close", "open"]]

        # Calculate volatility measures
        log_hl = np.log(df["high"] / df["low"])
        log_hl_squared = log_hl**2
        parkinson = (
            log_hl_squared.rolling(self.lag).sum() / (4 * self.lag * np.log(2))
        ).apply(np.sqrt)

        # Get current and previous values
        parkinson_current = parkinson.iloc[-1]
        parkinson_prev = parkinson.iloc[-2]

        manager.context["indicators"].update(
            {f"parkinson_vol_{self.lag}": parkinson_current}
        )

        logger.info(
            f"Previous Parkinson Volatility: {parkinson_prev:.4f}; Current Parkinson Volatility: {parkinson_current:.4f}."
        )
        if parkinson_current < parkinson_prev:
            logger.info("Passed Parkinson Volatility Decrease Check")
        else:
            logger.info("Failed Parkinson Volatility Decrease Check")

        # Check both measures decreased
        return parkinson_current < parkinson_prev

    def __repr__(self):
        return f"ParkinsonVolatilityDecreaseCheck(lag={self.lag})"
