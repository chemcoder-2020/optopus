from .base import BaseComponent
import pandas as pd
import numpy as np
from loguru import logger


class YangZhangVolatilityDecreaseCheck(BaseComponent):
    """Check if Yang-Zhang Volatility is decreasing.

    Returns True if:
    - Yang-Zhang volatility shows decrease from previous to current window
    """

    def __init__(self, lag=21):
        self.lag = lag

    def should_enter(self, strategy, manager, time: pd.Timestamp) -> bool:
        hist_data = manager.context["historical_data"]
        if len(hist_data) < self.lag + 2:  # Need at least lag+1 periods for calculations
            return False

        df = hist_data[["high", "low", "close", "open"]]

        # Calculate components for Yang-Zhang volatility
        # Close-to-open returns
        close_to_open = np.log(df['open'] / df['close'].shift(1))
        mu_co = close_to_open.rolling(self.lag-1).mean()
        sigma_co_sq = close_to_open.rolling(self.lag-1).var(ddof=0)
        
        # Open-to-close returns
        open_to_close = np.log(df['close'] / df['open'])
        mu_oc = open_to_close.rolling(self.lag-1).mean()
        sigma_oc_sq = open_to_close.rolling(self.lag-1).var(ddof=0)
        
        # Rogers-Satchell component
        rs_terms = (
            np.log(df['high']/df['close']) * np.log(df['high']/df['open']) + 
            np.log(df['low']/df['close']) * np.log(df['low']/df['open'])
        )
        sigma_rs_sq = rs_terms.rolling(self.lag-1).mean()
        
        # Calculate Yang-Zhang volatility
        k = 0.34 / (1.34 + (self.lag/(self.lag-2)))
        yang_zhang = np.sqrt(sigma_co_sq + k * sigma_oc_sq + (1 - k) * sigma_rs_sq)
        
        # Get current and previous values
        yz_current = yang_zhang.iloc[-1]
        yz_prev = yang_zhang.iloc[-2]

        if not hasattr(manager.context, "indicators"):
            manager.context["indicators"] = {}
        else:
            manager.context["indicators"].update(
                {f"yz_vol_{self.lag}": yz_current}
            )

        logger.info(
            f"Previous Yang-Zhang Volatility: {yz_prev:.4f}; Current Yang-Zhang Volatility: {yz_current:.4f}."
        )
        if yz_current < yz_prev:
            logger.info("Passed Yang-Zhang Volatility Decrease Check")
        else:
            logger.info("Failed Yang-Zhang Volatility Decrease Check")

        return yz_current < yz_prev

    def __repr__(self):
        return f"YangZhangVolatilityDecreaseCheck(lag={self.lag})"
