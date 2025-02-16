from abc import ABC, abstractmethod
import datetime
import pandas as pd
from typing import Union, TYPE_CHECKING, List, Tuple
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


class BaseComponent:
    """Base class for all pipeline components with operator overloading"""

    _registry = {}  # Class-level component registry

    @classmethod
    def register(cls, name: str):
        def decorator(subclass):
            cls._registry[name.lower()] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs):
        """Factory method for creating components"""
        return cls._registry[name.lower()](**kwargs)

    def __mul__(self, other):
        return AndComponent(self, other)

    def __or__(self, other):
        return OrComponent(self, other)

    def __invert__(self):
        return NotComponent(self)


class AndComponent(BaseComponent):
    """AND logical operator component"""

    def __init__(self, left: BaseComponent, right: BaseComponent):
        super().__init__()
        self.left = left
        self.right = right

    def should_enter(self, strategy, manager, time: pd.Timestamp) -> bool:
        return self.left.should_enter(
            strategy=strategy, manager=manager, time=time
        ) and self.right.should_enter(
            strategy=strategy,
            manager=manager,
            time=time,
        )


class OrComponent(BaseComponent):
    """OR logical operator component"""

    def __init__(self, left: BaseComponent, right: BaseComponent):
        super().__init__()
        self.left = left
        self.right = right

    def should_enter(self, strategy, manager, time: pd.Timestamp) -> bool:
        return self.left.should_enter(
            strategy=strategy, manager=manager, time=time
        ) or self.right.should_enter(
            strategy=strategy,
            manager=manager,
            time=time,
        )


class NotComponent(BaseComponent):
    """NOT logical operator component"""

    def __init__(self, component: BaseComponent):
        super().__init__()
        self.component = component

    def should_enter(self, strategy, manager, time: pd.Timestamp) -> bool:
        return not self.component.should_enter(
            strategy=strategy,
            manager=manager,
            time=time,
        )


class CompositePipelineCondition(EntryConditionChecker):
    """Decision pipeline combining multiple bot entry conditions using logical operators"""

    def __init__(self, pipeline: BaseComponent, ohlc_data: str, ticker=None):
        """
        Args:
            pipeline: Configured pipeline using component operators
            ohlc_data: Path to OHLC data file
        """
        self.pipeline = pipeline

    def should_enter(self, strategy, manager, time: pd.Timestamp) -> bool:
        logger.debug(f"Evaluating pipeline at {time}: {self.pipeline}")

        # Evaluate the pipeline
        result = self.pipeline.should_enter(
            strategy=strategy, manager=manager, time=time
        )
        logger.debug(f"Pipeline evaluation result: {result}")
        return result


class PremiumHampelFilterCondition(BaseComponent):
    def __init__(self, window_size=7, fluctuation=0.1, **kwargs):
        self.window_size = window_size
        self.fluctuation = fluctuation
        self.kwargs = kwargs
        self.filter_operator = HampelFilterNumpy(
            window_size=window_size,
            n_sigma=self.kwargs.get("n_sigma", 3),
            k=self.kwargs.get("k", 1.4826),
            max_iterations=self.kwargs.get("max_iterations", 5),
            replace_with_na=True,
        )

    def should_enter(self, strategy, manager, time: pd.Timestamp) -> bool:
        # Get current bid/ask and calculate mark price
        bid = strategy.current_bid
        ask = strategy.current_ask
        mark = (
            (ask + bid) / 2 if bid != 0 else ask
        )  # assume mark is trading at mid price if bid is not 0, and ask price if bid is 0

        # Initialize context if needed
        if not hasattr(manager, "context"):
            logger.debug("Creating new context in manager")
            manager.context = {}

        # Initialize premiums list in context if needed
        if "premiums" not in manager.context:
            manager.context["premiums"] = []

        # Add new premium to rolling window
        manager.context["premiums"].append(mark)

        # Trim to window size
        if len(manager.context["premiums"]) > self.window_size:
            manager.context["premiums"].pop(0)

        try:
            # Apply Hampel filter through window
            filtered_values = self.filter_operator.fit_transform(
                np.array(manager.context["premiums"])
            ).flatten()

            # Get most recent filtered value
            filtered_mark = filtered_values[-1] if len(filtered_values) > 0 else mark

            if np.isnan(filtered_mark):
                logger.debug(
                    "Filtered mark is NaN, replacing premium context with previous value and returning False"
                )
                manager.context["premiums"][-1] = manager.context["premiums"][-2]
                return False

            logger.debug(f"Premium passed Hampel filter: {filtered_mark}")

            return True

        except Exception as e:
            logger.error(f"Hampel filter error: {str(e)}")
            return False

class PremiumMedianFilterCondition(BaseComponent):
    def __init__(self, window_size=7, fluctuation=0.1, **kwargs):
        self.window_size = window_size
        self.fluctuation = fluctuation
        self.median_calculator = ContinuousMedian()

    def should_enter(self, strategy, manager, time: pd.Timestamp) -> bool:
        # Get current bid/ask and calculate mark price
        bid = strategy.current_bid
        ask = strategy.current_ask
        mark = (
            (ask + bid) / 2 if bid != 0 else ask
        )

        # Initialize context if needed
        if not hasattr(manager, "context"):
            logger.debug("Creating new context in manager")
            manager.context = {}

        # Initialize median premiums tracking if needed
        if "median_premiums" not in manager.context:
            manager.context["median_premiums"] = []
            self.median_calculator = ContinuousMedian()

        # Track premiums in rolling window
        manager.context["median_premiums"].append(mark)
        self.median_calculator.add(mark)

        # Maintain window size
        if len(manager.context["median_premiums"]) > self.window_size:
            oldest = manager.context["median_premiums"].pop(0)
            self.median_calculator.remove(oldest)

        try:
            # Get current median and calculate deviation
            current_median = self.median_calculator.get_median()
            deviation = abs((mark - current_median) / current_median) if current_median != 0 else 0.0
            
            logger.debug(f"Median filter check: {deviation:.2%} vs allowed {self.fluctuation:.2%}")
            
            # Store calculated median in context
            manager.context["current_median"] = current_median
            
            return deviation <= self.fluctuation

        except Exception as e:
            logger.error(f"Median filter error: {str(e)}")
            return False


class MedianCalculator(BaseComponent):
    def __init__(self, window_size=7, fluctuation=0.1, method="HampelFilter", **kwargs):
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


class MedianCalculator(EntryConditionChecker):
    def __init__(self, window_size=7, fluctuation=0.1, method="HampelFilter", **kwargs):
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


class TimeBasedEntryCondition(EntryConditionChecker):
    """
    Checks if current time is within allowed entry times.

    Args:
        allowed_times (List[Union[Tuple[str, str], str]]): List of time ranges in "HH:MM-HH:MM" format (default: 9:45AM-3:45PM ET)
        allowed_days (List[Union[int, str]]): List of allowed days (0=Mon-6=Sun or 'Mon'-'Sun', default: Mon-Fri)
        timezone (str): Timezone string (default: "America/New_York")
    """

    def __init__(
        self,
        allowed_times: List[Union[Tuple[str, str], str]] = ["09:45-15:45"],
        allowed_days: List[Union[int, str]] = ["Mon", "Tue", "Wed", "Thu", "Fri"],
        timezone: str = "America/New_York",
    ):
        self.allowed_times = allowed_times
        self.allowed_days = allowed_days
        self.timezone = timezone

        # Convert string day names to numbers if needed
        self.allowed_day_numbers = []
        day_map = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
        for day in self.allowed_days:
            if isinstance(day, str):
                self.allowed_day_numbers.append(day_map[day.lower()[:3]])
            else:
                self.allowed_day_numbers.append(int(day))

        # Parse time ranges
        self.time_ranges = []
        for time_spec in self.allowed_times:
            if isinstance(time_spec, str):
                start_str, end_str = time_spec.split("-")
            else:
                start_str, end_str = time_spec

            start = pd.Timestamp(start_str)  # .tz_localize(self.timezone)
            end = pd.Timestamp(end_str)  # .tz_localize(self.timezone)
            self.time_ranges.append((start.time(), end.time()))

    def should_enter(
        self,
        strategy,
        manager: "OptionBacktester",
        time: Union[datetime, str, pd.Timestamp],
    ) -> bool:
        current_time = pd.Timestamp(time)  # .tz_convert(self.timezone)

        # Check allowed days
        if self.allowed_day_numbers:
            day_of_week = current_time.dayofweek
            if day_of_week not in self.allowed_day_numbers:
                logger.warning(f"Entry not allowed on {current_time.day_name()}")
                return False

        # Check allowed time ranges
        if self.time_ranges:
            in_window = False
            curr_time = current_time.time()
            for start, end in self.time_ranges:
                if start <= curr_time <= end:
                    in_window = True
                    break

            if not in_window:
                logger.warning(
                    f"Current time {curr_time.strftime('%H:%M:%S')} UTC not in allowed ranges"
                )
                return False

        logger.info("Time-based entry conditions met")
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
                TimeBasedEntryCondition(
                    allowed_days=kwargs.get(
                        "allowed_days", ["Mon", "Tue", "Wed", "Thu", "Fri"]
                    ),
                    allowed_times=kwargs.get("allowed_times", ["09:45-15:45"]),
                    timezone=kwargs.get("timezone", "America/New_York"),
                ),
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
