from .base import BaseComponent
import pandas as pd
import numpy as np
from loguru import logger


class GarmanKlassVolatilityDecreaseCheck(BaseComponent):
    """Check if Garman-Klass Volatility is decreasing.

    Returns True if:
    - Garman-Klass volatility shows decrease from previous to current window
    """

    def __init__(self, lag=21):
        self.lag = lag

    def should_enter(self, strategy, manager, time: pd.Timestamp) -> bool:
        hist_data = manager.context["historical_data"]
        if len(hist_data) < self.lag + 1:
            return False

        df = hist_data[["high", "low", "close", "open"]]

        # Calculate Garman-Klass components
        log_hl = np.log(df['high']/df['low'])
        log_co = np.log(df['close']/df['open'])
        
        # Calculate volatility using Garman-Klass estimator
        gk_terms = 0.5*(log_hl**2) - (2*np.log(2)-1)*(log_co**2)
        gk_volatility = np.sqrt(gk_terms.rolling(self.lag).mean())

        # Get current and previous values
        gk_current = gk_volatility.iloc[-1]
        gk_prev = gk_volatility.iloc[-2]

        if not hasattr(manager.context, "indicators"):
            manager.context["indicators"] = {}
        else:
            manager.context["indicators"].update(
                {f"gk_vol_{self.lag}": gk_current}
            )

        logger.info(
            f"Previous Garman-Klass Volatility: {gk_prev:.4f}; Current Garman-Klass Volatility: {gk_current:.4f}."
        )
        if gk_current < gk_prev:
            logger.info("Passed Garman-Klass Volatility Decrease Check")
        else:
            logger.info("Failed Garman-Klass Volatility Decrease Check")

        return gk_current < gk_prev

    def __repr__(self):
        return f"GarmanKlassVolatilityDecreaseCheck(lag={self.lag})"
