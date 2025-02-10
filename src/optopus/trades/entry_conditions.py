from abc import ABC, abstractmethod
import datetime
import pandas as pd
from typing import Union, TYPE_CHECKING, List
from loguru import logger
import numpy as np
from ..utils.heapmedian import ContinuousMedian
from ..utils.filters import HampelFilterNumpy


if TYPE_CHECKING:
    from .option_manager import OptionBacktester


class EntryConditionChecker(ABC):
    """
    Abstract base class for entry condition checkers.

    Method:
        should_enter(strategy, manager, time) -> bool:
            Check if the entry conditions are met for the option strategy.
    """

    @abstractmethod
    def should_enter(
        self,
        strategy,
        manager: "OptionBacktester",
        time: Union[datetime, str, pd.Timestamp],
    ) -> bool:
        pass


class MedianCalculator(EntryConditionChecker):
    def __init__(
        self, window_size=7, fluctuation=0.1, method="HampelFilter", **kwargs
    ):
        self.window_size = window_size
        self.fluctuation = fluctuation
        self.method = method
        self.premiums = []
        self.kwargs = kwargs
        if method == "ContinuousMedian":
            self.median_calculator = ContinuousMedian()
        else:
            self.median_calculator = HampelFilterNumpy(
                window_size=window_size,
                n_sigma=self.kwargs.get("n_sigma", 3),
                k=self.kwargs.get("k", 1.4826),
                max_iterations=self.kwargs.get("max_iterations", 5),
                replace_with_na=self.kwargs.get("replace_with_na", False),
            )

    def add_premium(self, mark):
        if self.method == "ContinuousMedian":
            self.median_calculator.add(mark)
        self.premiums.append(mark)

        if self.method == "ContinuousMedian":
            if len(self.premiums) > self.window_size:
                self.median_calculator.remove(self.premiums.pop(0))
        else:
            if len(self.premiums) > self.window_size + 1:
                self.premiums.pop(0)

    def get_median(self):
        """
        Get the current median of the rolling window.

        Returns:
            float: The current median.
        """
        if self.method == "ContinuousMedian":
            return self.median_calculator.get_median()
        else:
            if len(self.premiums) < self.window_size + 1:
                return 0
            else:
                return self.median_calculator.fit_transform(
                    np.array(self.premiums)
                ).flatten()[-1]

    def should_enter(self, strategy, manager, time) -> bool:
        bid = strategy.current_bid
        ask = strategy.current_ask
        mark = (ask + bid) / 2

        self.add_premium(mark)
        median_mark = self.get_median()

        return np.isclose(mark, median_mark, rtol=self.fluctuation) and np.isclose(
            bid, mark, rtol=self.fluctuation
        )


class CapitalRequirementCondition(EntryConditionChecker):
    """
    Checks if there is sufficient capital for the trade.

    Method:
        should_enter(strategy, manager, time) -> bool:
            Adjusts the number of contracts based on the capital requirement and checks if the required capital is available.
    """

    def should_enter(
        self,
        strategy,
        manager: "OptionBacktester",
        time: Union[datetime, str, pd.Timestamp],
    ) -> bool:
        max_capital = min(
            manager.allocation * manager.config.position_size,
            manager.available_to_trade,
        )
        original_contracts = strategy.contracts
        strategy.contracts = min(
            original_contracts,
            int(max_capital // strategy.get_required_capital_per_contract()),
        )

        if strategy.contracts == 0:
            logger.warning(
                "Capital requirement not met: Strategy requires more capital than allowed by position size"
            )
            return False

        if strategy.contracts != original_contracts:
            logger.info(
                f"Adjusted contracts from {original_contracts} to {strategy.contracts} to fit position size"
            )

        required_capital = strategy.get_required_capital()
        has_sufficient_capital = (
            required_capital <= max_capital
            and required_capital <= manager.available_to_trade
        )

        if has_sufficient_capital:
            logger.info(
                f"Capital requirement met: Required ${required_capital:.2f} <= Available ${manager.available_to_trade:.2f}"
            )
        else:
            logger.warning(
                f"Capital requirement not met: Required ${required_capital:.2f} > Available ${manager.available_to_trade:.2f}"
            )

        return has_sufficient_capital


class PositionLimitCondition(EntryConditionChecker):
    """
    Checks if position limits are respected.

    Method:
        should_enter(strategy, manager, time) -> bool:
            Checks if the number of active trades, daily trades, and weekly trades are within the configured limits.
    """

    def should_enter(
        self,
        strategy,
        manager: "OptionBacktester",
        time: Union[datetime, str, pd.Timestamp],
    ) -> bool:
        position_limit_ok = len(manager.active_trades) < manager.config.max_positions
        if not position_limit_ok:
            logger.warning(
                f"Position limit reached: {len(manager.active_trades)} active trades >= limit of {manager.config.max_positions}"
            )
        else:
            logger.info(
                f"Position limit ok: {len(manager.active_trades)} active trades < limit of {manager.config.max_positions}"
            )

        daily_limit_ok = (
            manager.config.max_positions_per_day is None
            or manager.trades_entered_today < manager.config.max_positions_per_day
        )
        if not daily_limit_ok:
            logger.warning(
                f"Daily position limit reached: {manager.trades_entered_today} trades today >= limit of {manager.config.max_positions_per_day}"
            )
        elif manager.config.max_positions_per_day is not None:
            logger.info(
                f"Daily position limit ok: {manager.trades_entered_today} trades today < limit of {manager.config.max_positions_per_day}"
            )

        weekly_limit_ok = (
            manager.config.max_positions_per_week is None
            or manager.trades_entered_this_week < manager.config.max_positions_per_week
        )
        if not weekly_limit_ok:
            logger.warning(
                f"Weekly position limit reached: {manager.trades_entered_this_week} trades this week >= limit of {manager.config.max_positions_per_week}"
            )
        elif manager.config.max_positions_per_week is not None:
            logger.info(
                f"Weekly position limit ok: {manager.trades_entered_this_week} trades this week < limit of {manager.config.max_positions_per_week}"
            )

        return all([position_limit_ok, daily_limit_ok, weekly_limit_ok])


class RORThresholdCondition(EntryConditionChecker):
    """
    Checks if return over risk meets the threshold.

    Method:
        should_enter(strategy, manager, time) -> bool:
            Checks if the return over risk meets the configured threshold.
    """

    def should_enter(
        self,
        strategy,
        manager: "OptionBacktester",
        time: Union[datetime, str, pd.Timestamp],
    ) -> bool:
        if manager.config.ror_threshold is None:
            logger.info("ROR threshold check skipped: No threshold configured")
            return True

        ror = strategy.return_over_risk()
        meets_threshold = ror >= manager.config.ror_threshold

        if meets_threshold:
            logger.info(
                f"ROR threshold met: {ror:.2%} >= {manager.config.ror_threshold:.2%}"
            )
        else:
            logger.warning(
                f"ROR threshold not met: {ror:.2%} < {manager.config.ror_threshold:.2%}"
            )

        return meets_threshold


class ConflictCondition(EntryConditionChecker):
    """
    Checks for conflicts with existing positions.

    Args:
        check_closed_trades (bool): Whether to also check for conflicts with closed trades.

    Method:
        should_enter(strategy, manager, time) -> bool:
            Checks for conflicts with existing active and closed trades if enabled.
    """

    def __init__(self, check_closed_trades: bool = False):
        """
        Initialize the conflict condition checker.

        Args:
            check_closed_trades (bool): Whether to also check for conflicts with closed trades.
        """
        self.check_closed_trades = check_closed_trades

    def should_enter(
        self,
        strategy,
        manager: "OptionBacktester",
        time: Union[datetime, str, pd.Timestamp],
    ) -> bool:
        # Check active trades
        for existing_spread in manager.active_trades:
            if existing_spread.conflicts_with(strategy):
                logger.warning(
                    f"Position conflict detected with existing active {existing_spread.strategy_type} trade"
                )
                return False

        # Check closed trades if enabled
        if self.check_closed_trades:
            for closed_spread in manager.closed_trades:
                if closed_spread.conflicts_with(strategy):
                    logger.warning(
                        f"Position conflict detected with closed {closed_spread.strategy_type} trade"
                    )
                    return False

            if manager.closed_trades:
                logger.info(
                    f"No conflicts found with {len(manager.closed_trades)} closed trades"
                )

        if manager.active_trades:
            logger.info(
                f"No conflicts found with {len(manager.active_trades)} active trades"
            )
        else:
            logger.info("No existing active trades to check for conflicts")

        return True


class TrailingStopEntry(EntryConditionChecker):
    def __init__(self, **kwargs):
        """
        Trailing stop entry condition checker.

        Args:
            trailing_entry_direction (str): 'bullish' or 'bearish' direction for trailing stop
            trailing_entry_threshold (float): Threshold value for entry trigger
            method (str): Calculation method ('percent', 'dollar', or 'atr')
            trailing_entry_reset_period (str): Reset period for trailing stop ('D', 'W', 'M', 'Y')
        """
        self.trailing_entry_direction = kwargs.get(
            "trailing_entry_direction", "bullish"
        ).lower()
        if self.trailing_entry_direction not in ("bullish", "bearish"):
            raise ValueError("trailing_entry_direction must be 'bullish' or 'bearish'")

        self.trailing_entry_threshold = abs(kwargs.get("trailing_entry_threshold", 1.0))
        self.method = kwargs.get("method", "percent").lower()
        self.trailing_entry_reset_period = kwargs.get(
            "trailing_entry_reset_period", None
        )

        # State tracking variables
        self.current_date = None
        self.current_week = None
        self.current_month = None
        self.current_year = None
        self.cum_min = None  # Track cumulative min price
        self.cum_max = None  # Track cumulative max price

    def should_enter(self, strategy, manager, time) -> bool:
        time = pd.Timestamp(time)
        if self.trailing_entry_reset_period is not None:
            if self.trailing_entry_reset_period == "D":
                if self.current_date != time.date():
                    self.current_date = time.date()
                    self.cum_min = None
                    self.cum_max = None
            elif self.trailing_entry_reset_period == "W":
                current_week = time.isocalendar()[1]  # Get ISO week number
                if self.current_week != current_week:
                    self.current_week = current_week
                    self.cum_min = None
                    self.cum_max = None
            elif self.trailing_entry_reset_period == "M":
                if self.current_month != time.month:
                    self.current_month = time.month
                    self.cum_min = None
                    self.cum_max = None
            elif self.trailing_entry_reset_period == "Y":
                if self.current_year != time.year:
                    self.current_year = time.year
                    self.cum_min = None
                    self.cum_max = None

        current_price = strategy.underlying_last

        if self.cum_max is None:
            self.cum_max = current_price

        if self.cum_min is None:
            self.cum_min = current_price

        # Update cumulative min/max
        self.cum_min = min(self.cum_min, current_price)
        self.cum_max = max(self.cum_max, current_price)

        # Calculate price movement from extreme
        if self.trailing_entry_direction == "bullish":
            if self.method == "dollar":
                price_change = current_price - self.cum_min
            elif self.method == "percent":
                price_change = ((current_price - self.cum_min) / self.cum_min) * 100
            elif self.method == "atr":
                if hasattr(manager, "atr") and manager.atr > 0:
                    price_change = (current_price - self.cum_min) / manager.atr
                else:
                    logger.error(
                        "ATR not available or invalid for trailing stop calculation"
                    )
                    return False
        else:
            if self.method == "dollar":
                price_change = current_price - self.cum_max
            elif self.method == "percent":
                price_change = ((current_price - self.cum_max) / self.cum_max) * 100
            elif self.method == "atr":
                if hasattr(manager, "atr") and manager.atr > 0:
                    price_change = (current_price - self.cum_max) / manager.atr
                else:
                    logger.error(
                        "ATR not available or invalid for trailing stop calculation"
                    )
                    return False

        if self.trailing_entry_direction == "bullish":
            return price_change >= self.trailing_entry_threshold
        else:
            return price_change <= -self.trailing_entry_threshold


class CompositeEntryCondition(EntryConditionChecker):
    """
    Combines multiple entry conditions.

    Args:
        conditions (List[EntryConditionChecker]): List of entry conditions to combine.

    Method:
        should_enter(strategy, manager, time) -> bool:
            Checks if all combined entry conditions are met.
    """

    def __init__(self, conditions: List[EntryConditionChecker]):
        self.conditions = conditions

    def should_enter(
        self,
        strategy,
        manager: "OptionBacktester",
        time: Union[datetime, str, pd.Timestamp],
    ) -> bool:
        logger.info(f"Checking {len(self.conditions)} composite conditions")

        for i, condition in enumerate(self.conditions, 1):
            condition_name = condition.__class__.__name__
            if not condition.should_enter(strategy, manager, time):
                logger.warning(
                    f"Composite condition {i}/{len(self.conditions)} ({condition_name}) failed"
                )
                return False
            logger.info(
                f"Composite condition {i}/{len(self.conditions)} ({condition_name}) passed"
            )

        return True


class DefaultEntryCondition(EntryConditionChecker):
    """
    Default entry condition that combines all standard checks.

    Method:
        should_enter(strategy, manager, time) -> bool:
            Checks if all default entry conditions are met.
    """

    def __init__(self, **kwargs):
        self.composite = CompositeEntryCondition(
            [
                CapitalRequirementCondition(),
                PositionLimitCondition(),
                RORThresholdCondition(),
                ConflictCondition(
                    check_closed_trades=kwargs.get("check_closed_trades", True)
                ),
                MedianCalculator(
                    window_size=kwargs.get("window_size", 7),
                    fluctuation=kwargs.get("fluctuation", 0.1),
                    method=kwargs.get("filter_method", "HampelFilter"),
                    n_sigma=kwargs.get("n_sigma", 3),
                    k=kwargs.get("k", 1.4826),
                    max_iterations=kwargs.get("max_iterations", 5),
                ),
                TrailingStopEntry(
                    trailing_entry_direction=kwargs.get(
                        "trailing_entry_direction", "bullish"
                    ),
                    trailing_entry_threshold=kwargs.get("trailing_entry_threshold", 0),
                    method=kwargs.get("method", "percent"),
                    trailing_entry_reset_period=kwargs.get(
                        "trailing_entry_reset_period", None
                    ),
                ),
            ]
        )

    def should_enter(self, strategy, manager, time) -> bool:
        return self.composite.should_enter(strategy, manager, time)
