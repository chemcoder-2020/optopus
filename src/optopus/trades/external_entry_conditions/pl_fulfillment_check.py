from .base import BaseComponent
import pandas as pd
from loguru import logger


class PLFulfilmentCheck(BaseComponent):
    def __init__(
        self,
        target_return=0.01,
        target_loss_prevention=None,
        freq="W",
    ):
        self.target_return = target_return
        self.target_loss_prevention = target_loss_prevention
        self.freq = freq

    def should_enter(self, strategy, manager, time: pd.Timestamp) -> bool:
        pl = (
            pd.DataFrame(manager.performance_data)
            .set_index("time")["closed_pl"]
            .resample(self.freq)
            .last()
            .diff()
        )
        current_pl = pl.iloc[-1]

        current_return = current_pl / manager.config.initial_capital
        manager.context["indicators"].update(
            {f"Current {self.freq} Return": current_return}
        )

        if self.target_loss_prevention is not None:
            return -self.target_loss_prevention < current_return < self.target_return
        else:
            return current_return < self.target_return
