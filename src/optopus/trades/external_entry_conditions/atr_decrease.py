from .base import BaseComponent
import pandas as pd
from loguru import logger


class VolatilityDecreaseCheck(BaseComponent):
    def __init__(self, lag=14):
        self.lag = lag

    def should_enter(self, strategy, manager, time: pd.Timestamp) -> bool:
        import pandas_ta as pt

        hist_data = manager.context["historical_data"]

        data = hist_data.copy()
        high = data["high"]
        low = data["low"]
        close = data["close"]
        data["tr0"] = abs(high - low)
        data["tr1"] = abs(high - close.shift())
        data["tr2"] = abs(low - close.shift())
        tr = data[["tr0", "tr1", "tr2"]].max(axis=1)

        atr = pt.sma(tr, self.lag)

        if not hasattr(manager.context, "indicators"):
            manager.context["indicators"] = {}
        else:
            manager.context["indicators"].update({f"atr_{self.lag}": atr.iloc[-1]})

        assert time == manager.context.get(
            "bar"
        ), "VolatilityCheck: time is not equal to bar"
        logger.info(
            f"Previous ATR: {atr.iloc[-2]:.4f}; Current ATR: {atr.iloc[-1]:.4f}."
        )
        if atr.iloc[-1] < atr.iloc[-2]:
            logger.info("Passed ATR decrease check")
        else:
            logger.info("Failed ATR decrease check")

        return atr.iloc[-1] - atr.iloc[-2] < 0

    def __repr__(self):
        return f"VolatilityDecreaseCheck(lag={self.lag}"
