from .base import BaseComponent
import pandas as pd
from loguru import logger
from typing import Union
import datetime


class TotalPLLossCheck(BaseComponent):
    def __init__(self, max_total_pl_loss: float = 0.1):
        self.max_total_pl_loss = max_total_pl_loss

    def should_exit(
        self,
        strategy,
        current_time: Union[datetime, str, pd.Timestamp],
        option_chain_df: pd.DataFrame,
        manager=None,
    ) -> bool:
        df = pd.DataFrame(manager.performance_data).set_index("time")

        todays_total_pl = df[df.index >= pd.Timestamp(df.index[-1].date())]["total_pl"]
        current_drawdown = -min((todays_total_pl.iloc[-1] - todays_total_pl.iloc[0]), 0)

        if current_drawdown / manager.config.initial_capital > self.max_total_pl_loss:
            logger.info(
                f"Total PL loss exceeded: {current_drawdown / manager.config.initial_capital:.2%} > {self.max_total_pl_loss:.2%}"
            )
            return True

        logger.info(
            f"Total PL loss: {current_drawdown / manager.config.initial_capital:.2%} < {self.max_total_pl_loss:.2%}"
        )
        return False
