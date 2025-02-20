from optopus.trades.exit_conditions import (
    ExitConditionChecker,
    TimeBasedCondition,
    CompositeExitCondition,
    TrailingStopCondition,
    ProfitTargetCondition
)
from datetime import datetime
import pandas as pd
from typing import Union

class ExitCondition(ExitConditionChecker):

    def __init__(
        self,
        profit_target: float = 80,
        trigger: float = 40,
        stop_loss: float = 15,
        exit_time_before_expiration: pd.Timedelta = pd.Timedelta("1 day"),
        **kwargs,
    ):

        profit_target_condition = ProfitTargetCondition(
            profit_target=profit_target, window_size=kwargs.get("window_size", 10), method=kwargs.get("filter_method", "HampelFilter")
        )

        tsl_condition = TrailingStopCondition(
            trigger=trigger,
            stop_loss=stop_loss,
            window_size=kwargs.get("window_size", 10),
            method=kwargs.get("filter_method", "HampelFilter"),
            exit_upon_positive_return=kwargs.get("exit_upon_positive_return", False),
        )

        time_based_condition = TimeBasedCondition(
            exit_time_before_expiration=exit_time_before_expiration
        )

        self.composite_condition = CompositeExitCondition(
            conditions=[tsl_condition, time_based_condition, profit_target_condition],
            logical_operations=["OR", "OR"],
        )
        self.__dict__.update(self.composite_condition.__dict__)

    def __repr__(self):
        """
        Return a string representation of the default exit condition.

        Returns:
            str: String representation of the default exit condition.
        """
        return (
            f"{self.__class__.__name__}(composite_condition={self.composite_condition})"
        )

    def should_exit(
        self,
        strategy,
        current_time: Union[datetime, str, pd.Timestamp],
        option_chain_df: pd.DataFrame,
    ) -> bool:
        """
        Check if the default exit condition is met.

        Args:
            strategy (OptionStrategy): The option strategy to check.
            current_time (datetime): The current time for evaluation.
            option_chain_df (pd.DataFrame): The updated option chain data.

        Returns:
            bool: True if the default exit condition is met, False otherwise.
        """
        return self.composite_condition.should_exit(
            strategy, current_time, option_chain_df
        )

    def update(self, **kwargs):
        """
        Update the attributes of the default exit condition.

        Args:
            **kwargs: Keyword arguments for the attributes to update.
        """
        self.composite_condition.update(**kwargs)
        self.__dict__.update(self.composite_condition.__dict__)
