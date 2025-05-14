from .base import BaseComponent
import pandas as pd
from loguru import logger
from typing import Union
import datetime


class ShotClock(BaseComponent):
    def __init__(self, checkpoints=None):
        if checkpoints is None:
            checkpoints = {
                "12:00": 20,
                "12:30": 25,
                "13:00": 30,
                "13:30": 35,
                "14:00": 40,
            }
        self.checkpoints = checkpoints

    def should_exit(
        self,
        strategy,
        current_time: Union[datetime, str, pd.Timestamp],
        option_chain_df: pd.DataFrame,
        manager=None,
        **kwargs,
    ) -> bool:
        if not hasattr(strategy, "filter_return_percentage"):
            return_percentage = strategy.return_percentage()
        else:
            return_percentage = strategy.filter_return_percentage

        entry_time = strategy.entry_time

        for checkpoint, percentage in self.checkpoints.items():
            if (
                current_time.round("15min").strftime("%H:%M") == checkpoint
                and return_percentage > percentage
                and entry_time < current_time
            ):
                logger.info(
                    f"ShotClock triggered at {current_time.strftime('%H:%M')} with return percentage: {return_percentage}"
                )
                return True
        logger.info(
            f"ShotClock not triggered at {current_time.strftime('%H:%M')} with return percentage: {return_percentage}"
        )
        return False
