from .base import EntryConditionChecker, SequentialPipelineCondition
from .time_check import TimeBasedEntryCondition
from .premium_process_check import PremiumProcessCondition
from .capital_check import CapitalRequirementCondition
from .position_limit_check import PositionLimitCondition
from .ror_threshold_check import RORThresholdCondition
from .conflict_check import ConflictCondition
from .trailing_entry_check import TrailingStopEntry


class DefaultEntryCondition(EntryConditionChecker):
    def __init__(self, **kwargs):
        self.pipeline = SequentialPipelineCondition(
            steps=[
                (
                    TimeBasedEntryCondition(
                        allowed_days=kwargs.get(
                            "allowed_days", ["Mon", "Tue", "Wed", "Thu", "Fri"]
                        ),
                        allowed_times=kwargs.get("allowed_times", ["09:45-15:45"]),
                        timezone=kwargs.get("timezone", "America/New_York"),
                    ),
                    "AND",
                ),
                (
                    PremiumProcessCondition(
                        filter_method=kwargs.get("filter_method", "HampelFilterNumpy"),
                        window_size=kwargs.get("window_size", 3),
                        bid_mark_fluctuation=kwargs.get("fluctuation", 0.1),
                        n_sigma=kwargs.get("n_sigma", 3),
                        k=kwargs.get("k", 1.4826),
                        max_iterations=kwargs.get("max_iterations", 5),
                        implementation=kwargs.get("implementation", "pandas"),
                    ),
                    "AND",
                ),
                (CapitalRequirementCondition(), "AND"),
                (PositionLimitCondition(), "AND"),
                (
                    RORThresholdCondition(
                        filter_method=kwargs.get("filter_method", "HampelFilterNumpy"),
                        window_size=kwargs.get("window_size", 3),
                        n_sigma=kwargs.get("n_sigma", 3),
                        k=kwargs.get("k", 1.4826),
                        max_iterations=kwargs.get("max_iterations", 5),
                        implementation=kwargs.get("implementation", "pandas"),
                    ),
                    "AND",
                ),
                (
                    ConflictCondition(
                        check_closed_trades=kwargs.get("check_closed_trades", True)
                    ),
                    "AND",
                ),
                (
                    TrailingStopEntry(
                        trailing_entry_direction=kwargs.get(
                            "trailing_entry_direction", "bullish"
                        ),
                        trailing_entry_threshold=kwargs.get(
                            "trailing_entry_threshold", 0
                        ),
                        method=kwargs.get("method", "percent"),
                        trailing_entry_reset_period=kwargs.get(
                            "trailing_entry_reset_period", None
                        ),
                    ),
                    "AND",
                ),
            ]
        )

    def should_enter(self, strategy, manager, time) -> bool:
        return self.pipeline.should_enter(strategy, manager, time)
