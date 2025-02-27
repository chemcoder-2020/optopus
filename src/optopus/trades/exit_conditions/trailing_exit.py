from .base import (
    CompositePipelineCondition,
    PremiumListInit,
    PremiumFilter,
    ExitConditionChecker,
)
from .time_based import TimeBasedCondition
from .trailing_stoploss import TrailingStopCondition
import pandas as pd
from typing import Union
from datetime import datetime


class PipelineTrailingStopExit(CompositePipelineCondition):
    """
    Composite exit condition using pipeline architecture that combines:
    - Trailing stop logic
    - Time-based exit
    - Premium filtering preprocessing
    """

    def __init__(
        self,
        profit_target: float,
        trigger: float,
        stop_loss: float,
        exit_time_before_expiration: pd.Timedelta,
        window_size: int = 5,
        n_sigma: int = 3,
        **kwargs,
    ):
        # Create pipeline components
        trailing = TrailingStopCondition(
            profit_target=profit_target, trigger=trigger, stop_loss=stop_loss
        )
        time_based = TimeBasedCondition(
            exit_time_before_expiration=exit_time_before_expiration
        )

        # Build pipeline with OR logic
        pipeline = trailing | time_based  # OR combination

        # Configure preprocessing
        preprocessors = [
            PremiumListInit(),
            PremiumFilter(window_size=window_size, n_sigma=n_sigma),
        ]

        super().__init__(pipeline=pipeline, preprocessors=preprocessors, **kwargs)

    def update(self, **kwargs):
        """Update parameters through pipeline components"""
        # Update trailing stop parameters
        if hasattr(self.pipeline.left, "update"):
            self.pipeline.left.update(
                profit_target=kwargs.get(
                    "profit_target", self.pipeline.left.profit_target
                ),
                trigger=kwargs.get("trigger", self.pipeline.left.trigger),
                stop_loss=kwargs.get("stop_loss", self.pipeline.left.stop_loss),
            )

        # Update time-based parameters
        if hasattr(self.pipeline.right, "update"):
            self.pipeline.right.update(
                exit_time_before_expiration=kwargs.get(
                    "exit_time_before_expiration",
                    self.pipeline.right.exit_time_before_expiration,
                )
            )

        # Update filter parameters
        for preprocessor in self.preprocessors:
            if isinstance(preprocessor, PremiumFilter):
                preprocessor.update(
                    window_size=kwargs.get("window_size", preprocessor.window_size),
                    n_sigma=kwargs.get("n_sigma", preprocessor.n_sigma),
                )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"trailing={self.pipeline.left}, "
            f"time_based={self.pipeline.right}, "
            f"filter={self.preprocessors[1]})"
        )


class TrailingStopExitCondition(ExitConditionChecker):
    """
    Composite exit condition combining trailing stop logic with time-based exit,
    using premium filtering preprocessing.
    """

    def __init__(
        self,
        profit_target: float,
        trigger: float,
        stop_loss: float,
        exit_time_before_expiration: pd.Timedelta,
        window_size: int = 5,
        n_sigma: int = 3,
        **kwargs,
    ):
        super().__init__()
        self.trailing_condition = TrailingStopCondition(
            profit_target=profit_target, trigger=trigger, stop_loss=stop_loss
        )
        self.time_condition = TimeBasedCondition(
            exit_time_before_expiration=exit_time_before_expiration
        )
        self.preprocessors = [
            PremiumListInit(),
            PremiumFilter(window_size=window_size, n_sigma=n_sigma),
        ]

    def should_exit(
        self,
        strategy,
        current_time: Union[datetime, str, pd.Timestamp],
        option_chain_df: pd.DataFrame,
    ) -> bool:
        # Run preprocessing steps
        for preprocessor in self.preprocessors:
            preprocessor.preprocess(strategy)

        # Check both conditions with OR logic
        return self.trailing_condition.should_exit(
            strategy, current_time, option_chain_df
        ) or self.time_condition.should_exit(strategy, current_time, option_chain_df)

    def update(self, **kwargs):
        """Update parameters for both conditions and filter"""
        self.trailing_condition.update(
            profit_target=kwargs.get(
                "profit_target", self.trailing_condition.profit_target
            ),
            trigger=kwargs.get("trigger", self.trailing_condition.trigger),
            stop_loss=kwargs.get("stop_loss", self.trailing_condition.stop_loss),
        )

        self.time_condition.update(
            exit_time_before_expiration=kwargs.get(
                "exit_time_before_expiration",
                self.time_condition.exit_time_before_expiration,
            )
        )

        # Update filter parameters if provided
        for preprocessor in self.preprocessors:
            if isinstance(preprocessor, PremiumFilter):
                preprocessor.update(
                    window_size=kwargs.get(
                        "filter_window_size", preprocessor.window_size
                    ),
                    n_sigma=kwargs.get("filter_n_sigma", preprocessor.n_sigma),
                )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"profit_target={self.trailing_condition.profit_target}%, "
            f"trigger={self.trailing_condition.trigger}%, "
            f"stop_loss={self.trailing_condition.stop_loss}%, "
            f"exit_time={self.time_condition.exit_time_before_expiration}, "
            f"filter={self.preprocessors[1]})"
        )
