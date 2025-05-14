from .base import BaseComponent
import pandas as pd
from loguru import logger
from typing import Union
import datetime


class WeeklyPLCheckForExit(BaseComponent):
    def __init__(
        self,
        target_return=0.01,
        target_loss_prevention=0.01,
        loss_prevention_to_last_week=0.5,
    ):
        self.target_return = target_return
        self.target_loss_prevention = target_loss_prevention
        self.loss_prevention_to_last_week = loss_prevention_to_last_week

    def should_exit(self, strategy, manager, time: pd.Timestamp) -> bool:
        weekly_pl = (
            pd.DataFrame(manager.performance_data)
            .set_index("time")["closed_pl"]
            .resample("W")
            .last()
            .diff()
        )
        previous_weekly_pl = weekly_pl.iloc[-2]
        current_weekly_pl = weekly_pl.iloc[-1]
        current_weekly_return = current_weekly_pl / manager.config.initial_capital

        # Prevention check: exit if current weekly PL loss exceeds the previous week's PL gain

        if previous_weekly_pl > 0:
            loss_prevention_triggered = (
                current_weekly_pl
                < -previous_weekly_pl * self.loss_prevention_to_last_week
            )
        else:
            loss_prevention_triggered = False

        return (
            -self.target_loss_prevention > current_weekly_return
            or current_weekly_return > self.target_return
            or loss_prevention_triggered
        )
