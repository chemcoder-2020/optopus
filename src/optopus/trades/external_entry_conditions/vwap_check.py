from .base import BaseComponent
import pandas as pd
from loguru import logger


class VWAPCheck(BaseComponent):
    def __init__(self, anchor="W"):
        super().__init__()
        self.anchor = anchor

    def should_enter(self, strategy, manager, time: pd.Timestamp) -> bool:
        import optopus.pandas_ta as pt

        hist_data = manager.context["historical_data"]
        vwap = pt.vwap(
            high=hist_data["high"],
            low=hist_data["low"],
            close=hist_data["close"],
            volume=hist_data["volume"],
            anchor=self.anchor,
        )

        vwap_check = hist_data["close"].iloc[-1] > vwap.iloc[-1]

        if vwap_check:
            logger.info(f"VWAPCheck on {self.anchor} anchor Passed")
        else:
            logger.info(f"VWAPCheck on {self.anchor} anchor Failed")

        manager.context["indicators"].update({f"VWAP_{self.anchor}": vwap.iloc[-1]})
        return vwap_check
