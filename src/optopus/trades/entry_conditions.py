from abc import ABC, abstractmethod
import datetime
import pandas as pd
from typing import Union, TYPE_CHECKING, List, Tuple, Type
from loguru import logger
import numpy as np
import importlib
from ..utils.heapmedian import ContinuousMedian
from ..utils.filters import HampelFilterNumpy, Filter


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


class Preprocessor(ABC):
    @abstractmethod
    def preprocess(self, data):
        pass


class PremiumListInit(Preprocessor):
    def preprocess(self, strategy, manager):
        if not hasattr(manager, "context"):
            manager.context = {}

        if "premiums" not in manager.context:
            manager.context["premiums"] = []

        logger.debug("Premium log initialized for manager")
        return True


class PremiumFilter(Preprocessor):
    def __init__(
        self,
        filter_method: Union[Filter, Type[Filter], str] = HampelFilterNumpy,
        **kwargs,
    ):
        if isinstance(filter_method, str):
            filter_module = importlib.import_module("optopus.utils.filters")
            filter_method = getattr(filter_module, filter_method)
        elif isinstance(filter_method, type) and issubclass(filter_method, Filter):
            filter_method = filter_method
        else:
            raise ValueError(
                "filter_method must be a Filter class or a string name of a Filter class"
            )

        self.filter_method = filter_method

        self.premium_filter = self.filter_method(**kwargs)

        for key, value in kwargs.items():
            setattr(self, key, value)

    def preprocess(self, strategy, manager):
        """
        Check if current premium's return percentage is an outlier. Updates strategy's filtered metrics with the cleaned values.

        Args:
            strategy (OptionStrategy): The option strategy to check
        """
        bid = strategy.current_bid
        ask = strategy.current_ask
        mark = (ask + bid) / 2  # if bid != 0 else ask
        manager.context["premiums"].append(mark)
        if len(manager.context["premiums"]) > self.window_size + 1:
            manager.context["premiums"].pop(0)

        if len(manager.context["premiums"]) < self.window_size + 1:
            logger.debug(
                f"Manager's premiums context has less than {self.window_size + 1} values, skipping filter - assume valid data"
            )
            return mark

        filtered_returns = self.premium_filter.fit_transform(manager.context["premiums"])

        return filtered_returns[-1]

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.filter_method.__name__}: {self.kwargs})"
        )


class MedianCalculator(EntryConditionChecker):
    def __init__(self, window_size=7, fluctuation=0.1, method="HampelFilter", **kwargs):
        self.window_size = window_size
        self.fluctuation = fluctuation
        self.method = method
        self.kwargs = kwargs
        if method == "ContinuousMedian":
            self.median_calculator = ContinuousMedian()
        else:
            self.median_calculator = HampelFilterNumpy(
                window_size=window_size,
                n_sigma=self.kwargs.get("n_sigma", 3),
                k=self.kwargs.get("k", 1.4826),
                max_iterations=self.kwargs.get("max_iterations", 5),
                replace_with_na=True,
            )

    def should_enter(self, strategy, manager, time) -> bool:
        bid = strategy.current_bid
        ask = strategy.current_ask
        mark = (ask + bid) / 2  # if bid != 0 else ask

        if not hasattr(manager, "context"):
            manager.context = {}

        if "premiums" not in manager.context:
            manager.context["premiums"] = []

        manager.context["premiums"].append(mark)
        if self.method == "ContinuousMedian":
            self.median_calculator.add(mark)

        if self.method == "ContinuousMedian":
            if len(manager.context["premiums"]) > self.window_size:
                self.median_calculator.remove(manager.context["premiums"].pop(0))
        else:
            if len(manager.context["premiums"]) > self.window_size + 1:
                manager.context["premiums"].pop(0)

        if self.method == "ContinuousMedian":
            filtered_mark = self.median_calculator.get_median()
        else:
            if len(manager.context["premiums"]) < self.window_size + 1:
                filtered_mark = 0
            else:
                filtered_mark = self.median_calculator.fit_transform(
                    np.array(manager.context["premiums"])
                ).flatten()[-1]

        if filtered_mark == 0:
            logger.warning(
                f"Filtered mark is 0. Probably not enough data for MedianCalculator's window size of {self.window_size}"
            )
            return False
        elif np.isnan(filtered_mark):
            logger.warning(
                f"Filtered mark is NaN. The filter {self.method} has detected an outlying price. Returning False and replacing premium list with previous value."
            )
            manager.context["premiums"][-1] = manager.context["premiums"][-2]
            return False

        logger.info(
            f"Passed MedianCalculator check. Filtered mark: {filtered_mark}, Bid: {bid}, Mark: {mark}, Fluctuation: {self.fluctuation}"
        )

        return np.isclose(mark, filtered_mark, rtol=self.fluctuation) and np.isclose(
            bid, mark, rtol=self.fluctuation
        )

class PremiumProcessCondition(EntryConditionChecker):
    def __init__(self, bid_mark_fluctuation=0.1, **kwargs):
        self.premium_filter = PremiumFilter(**kwargs)
        self.premium_init = PremiumListInit()
        self.bid_mark_fluctuation = bid_mark_fluctuation
        self.kwargs = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
            
    def should_enter(self, strategy, manager, time) -> bool:
        # Init the premium context
        self.premium_init.preprocess(strategy, manager)

        # Apply the filter
        filtered_mark = self.premium_filter.preprocess(strategy, manager)

        # Check if the mark is close to the bid
        bid = strategy.current_bid
        if not np.isclose(filtered_mark, bid, rtol=self.bid_mark_fluctuation):
            logger.warning(
                f"Premium is not close to the bid. Filtered mark: {filtered_mark}, Bid: {bid}, Fluctuation: {self.bid_mark_fluctuation}"
            )
            return False

        logger.info(
            f"Passed PremiumProcessCondition check. Filtered mark: {filtered_mark}, Bid: {bid}, Fluctuation: {self.bid_mark_fluctuation}"
        )

        return True


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
                logger.info(
                    f"Entry not allowed on {current_time.day_name()}. Rejecting entry on TimeBasedEntryCondition"
                )
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
                logger.info(
                    f"Current time {curr_time.strftime('%H:%M:%S')} not in allowed ranges. Rejecting entry on TimeBasedEntryCondition"
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


class SequentialPipelineCondition(EntryConditionChecker):
    """Process conditions sequentially with configurable logic operators"""

    LOGIC_MAP = {
        "AND": lambda a, b: a and b,
        "OR": lambda a, b: a or b,
        "XOR": lambda a, b: (a or b) and not (a and b),
        "NAND": lambda a, b: not (a and b),
    }

    def __init__(self, steps: List[Tuple[EntryConditionChecker, str]]):
        """
        Args:
            steps: List of (condition, logic_operator) tuples.
                   First condition's operator is ignored
        """
        self.steps = steps

    def should_enter(self, strategy, manager, time) -> bool:
        if not self.steps:
            logger.info("SequentialPipeline: No steps configured, denying entry")
            return False

        # Evaluate first step
        first_condition = self.steps[0][0]
        result = first_condition.should_enter(strategy, manager, time)
        logger.info(
            f"SequentialPipeline Step 1/{len(self.steps)} ({first_condition.__class__.__name__}): {result}"
        )
        if not result:
            if self.steps[0][1] == "AND":
                logger.info("SequentialPipeline: First step failed, denying entry")
                return False
            else:
                logger.info(
                    "SequentialPipeline: First step failed, still evaluating remaining steps"
                )

        # Evaluate remaining steps
        for i, (condition, logic) in enumerate(self.steps[1:], start=2):
            condition_name = condition.__class__.__name__

            if logic not in self.LOGIC_MAP:
                raise ValueError(f"Invalid logic operator: {logic}")

            current = condition.should_enter(strategy, manager, time)
            new_result = self.LOGIC_MAP[logic](result, current)

            logger.info(
                f"SequentialPipeline Step {i}/{len(self.steps)} ({condition_name}) "
                f"with logic '{logic}': {result} {logic} {current} => {new_result}"
            )

            result = new_result

            # Short-circuit evaluation
            if logic == "AND" and not result:
                logger.info("Short-circuiting due to AND logic with False result")
                break
            if logic == "OR" and result:
                logger.info("Short-circuiting due to OR logic with True result")
                break

        logger.info(f"Final SequentialPipeline result: {result}")
        return result


class ConditionalGate(EntryConditionChecker):
    """Only evaluate main condition if pre-condition passes"""

    def __init__(
        self,
        main_condition: EntryConditionChecker,
        pre_condition: EntryConditionChecker,
    ):
        self.main_condition = main_condition
        self.pre_condition = pre_condition

    def should_enter(self, strategy, manager, time) -> bool:
        if self.pre_condition.should_enter(strategy, manager, time):
            return self.main_condition.should_enter(strategy, manager, time)
        return True  # Bypass if pre-condition fails


class DefaultEntryCondition(EntryConditionChecker):
    def __init__(self, **kwargs):
        self.pipeline = SequentialPipelineCondition(
            steps=[
                (CapitalRequirementCondition(), "AND"),
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
                # (
                #     MedianCalculator(
                #         window_size=kwargs.get("window_size", 7),
                #         fluctuation=kwargs.get("fluctuation", 0.1),
                #         method=kwargs.get("filter_method", "HampelFilter"),
                #         n_sigma=kwargs.get("n_sigma", 3),
                #         k=kwargs.get("k", 1.4826),
                #         max_iterations=kwargs.get("max_iterations", 5),
                #     ),
                #     "AND",
                # ),
                (
                    PremiumProcessCondition(
                        window_size=kwargs.get("window_size", 3),
                        bid_mark_fluctuation=kwargs.get("fluctuation", 0.1),
                        filter_method=kwargs.get("filter_method", "HampelFilterNumpy"),
                        n_sigma=kwargs.get("n_sigma", 3),
                        k=kwargs.get("k", 1.4826),
                        max_iterations=kwargs.get("max_iterations", 5),
                        implementation=kwargs.get("implementation", "pandas"),
                    ),
                    "AND",
                ),
                (PositionLimitCondition(), "AND"),
                (RORThresholdCondition(), "AND"),
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
