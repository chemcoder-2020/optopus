from .base import BaseComponent
import pandas as pd
import numpy as np
from loguru import logger


class RogersSatchellVolatilityDecreaseCheck(BaseComponent):
    """Check if Rogers-Satchell Volatility is decreasing.

    Returns True if:
    - Rogers-Satchell volatility shows decrease from previous to current window
    """

    def __init__(self, lag=21):
        self.lag = lag

    def should_enter(self, strategy, manager, time: pd.Timestamp) -> bool:
        hist_data = manager.context["historical_data"]
        if len(hist_data) < self.lag + 1:
            return False

        df = hist_data[["high", "low", "close", "open"]]

        # Calculate Rogers-Satchell volatility components
        rs_terms = (
            np.log(df['high']/df['close']) * np.log(df['high']/df['open']) + 
            np.log(df['low']/df['close']) * np.log(df['low']/df['open'])
        )
        
        # Calculate volatility
        rs_volatility = np.sqrt(rs_terms.rolling(self.lag).mean())

        # Get current and previous values
        rs_current = rs_volatility.iloc[-1]
        rs_prev = rs_volatility.iloc[-2]

        if not hasattr(manager.context, "indicators"):
            manager.context["indicators"] = {}
        else:
            manager.context["indicators"].update(
                {f"rs_vol_{self.lag}": rs_current}
            )

        logger.info(
            f"Previous Rogers-Satchell Volatility: {rs_prev:.4f}; Current Rogers-Satchell Volatility: {rs_current:.4f}."
        )
        if rs_current < rs_prev:
            logger.info("Passed Rogers-Satchell Volatility Decrease Check")
        else:
            logger.info("Failed Rogers-Satchell Volatility Decrease Check")

        return rs_current < rs_prev

    def __repr__(self):
        return f"RogersSatchellVolatilityDecreaseCheck(lag={self.lag})"
